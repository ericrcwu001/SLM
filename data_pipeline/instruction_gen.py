"""Instruction generation (teacher) + deterministic tag validation.

The teacher turns each accepted supported image-LUT pair into a synthetic instruction triple
``{gold_tags, concise, natural}`` (docs/data_collection_plan.md "Instruction Generation").
Its inputs are the LUT's *measured effect* — the deterministic behavior vector plus the
candidate structured tags — and, when available, the source image (the pinned teacher is
vision-capable). It never sees raw LUT floats or the target LUT.

Authority split (ADR 0004): the deterministic behavior vector is authoritative for every
measurable tag claim (:func:`validate_tags_against_behavior`); the LLM/VLM judge
(:mod:`eval.judge_client`) is a non-authoritative language / semantic gate layered on top.

Gating: :meth:`TeacherClient.generate` raises :class:`RequiresTeacher` until
``configs/model_clients.yaml`` pins ``teacher_primary`` with provider, a concrete ``model_id``
(no aliases), endpoint/api-key env-var names, prompt version, and batch id, AND the referenced
credential env vars are set at call time. The bidirectional tag<->behavior validation below is
authoritative and usable regardless of teacher availability.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Callable, Optional

import yaml

from eval import openai_compat

from .constants import (
    INSTRUCTION_STATUS_AUTHORED,
    INSTRUCTION_STATUS_GENERATED,
    INSTRUCTION_STATUS_PENDING,
    INSTRUCTION_STATUS_REJECTED,
)
from .errors import RequiresTeacher, TeacherGenerationError

_ALIASES = {"latest", "stable", "current", "default", "auto"}
_REQUIRED_PROFILE_KEYS = ("provider", "model_id", "endpoint_env", "api_key_env",
                          "prompt_version", "batch_id")

# The smallest Lab-domain move the deterministic gate treats as "measurably present". The
# behavior summary shown to the teacher and the forward tag<->behavior check MUST agree on
# this, or the teacher is told a direction the gate then rejects (wasted spend + false reject).
_MIN_MEASURABLE_MOVE = 1.0
# Reverse-coverage threshold for instruction generation. Kept equal to
# AcceptanceChecker's default (active_dataset.py) and the "predeclared coverage threshold" in
# docs/data_collection_plan.md so the money-spending path and the acceptance gate never
# diverge (a lower value here would reject rows acceptance keeps).
INSTRUCTION_COVERAGE_THRESHOLD = 5.0

# gold_tag -> (behavior key, required sign) for direction validation.
_TAG_BEHAVIOR = {
    "warmer": ("temperature_delta_b", +1), "cooler": ("temperature_delta_b", -1),
    "tint_magenta": ("tint_delta_a", +1), "tint_green": ("tint_delta_a", -1),
    "brighter": ("mean_l_delta", +1), "darker": ("mean_l_delta", -1),
    "more_contrast": ("contrast_l_spread_delta", +1), "less_contrast": ("contrast_l_spread_delta", -1),
    "more_saturated": ("chroma_delta", +1), "muted": ("chroma_delta", -1),
    "lifted_blacks": ("black_point_l_delta", +1), "crushed_blacks": ("black_point_l_delta", -1),
    "lifted_shadows": ("shadow_l_delta", +1), "brighter_highlights": ("highlight_l_delta", +1),
}

# Style-bundle tags imply a recipe of several behaviors, so their presence satisfies the
# reverse "unmentioned behavior" coverage check (the row is a style/composite, not a single
# attribute claim).
_STYLE_TAGS = {"matte", "faded", "filmic", "cinematic", "teal-orange", "sepia",
               "bleach bypass", "natural"}

# The full instruction-tag vocabulary the teacher may emit (single source of truth for the
# prompt). Directional tags first, then style bundles.
KNOWN_TAGS = sorted(_TAG_BEHAVIOR.keys()) + sorted(_STYLE_TAGS)

# behavior key -> {sign: word} for the human-readable behavior summary handed to the teacher.
_KEY_WORDS: dict[str, dict[int, str]] = {}
for _tag, (_key, _sign) in _TAG_BEHAVIOR.items():
    _KEY_WORDS.setdefault(_key, {})[_sign] = _tag

# Measurable axes summarized for the teacher (direction axes + a few context signals).
_SUMMARY_KEYS = [
    "temperature_delta_b", "tint_delta_a", "mean_l_delta", "contrast_l_spread_delta",
    "chroma_delta", "black_point_l_delta", "shadow_l_delta", "highlight_l_delta",
    "split_tone_strength", "neutral_drift_deltaE", "skin_locus_deltaE00_mean", "clip_rate",
]


def build_teacher_system_prompt() -> str:
    """Role, output-JSON contract, tag vocabulary, and the content restrictions (ADR 0004 /
    data_collection_plan.md "Instruction Generation")."""
    directional = ", ".join(sorted(_TAG_BEHAVIOR.keys()))
    styles = ", ".join(sorted(_STYLE_TAGS))
    return f"""You write synthetic natural-language editing instructions for a global \
color-grading dataset. You are given the MEASURED color effect of one 3D color LUT (a set of \
Lab-domain deltas from a deterministic probe) and, when provided, the source image the LUT \
was measured on. A 3D LUT is GLOBAL: it remaps every pixel's color the same way and cannot \
address regions, objects, or subjects.

Your job: describe the measured global look as if instructing an editor, in three forms.

OUTPUT CONTRACT — output EXACTLY ONE JSON object and nothing else (no prose, no markdown \
fences):
{{
  "gold_tags": [ ...subset of the allowed tags below, matching the measured directions... ],
  "concise": "one short, literal instruction naming the changes",
  "natural": "one more natural / creative phrasing of the same look"
}}

ALLOWED TAGS — use only these, and only when the measured behavior actually moves in that \
direction:
  directional: {directional}
  style bundles: {styles}

RULES:
- The MEASURED behavior is the ground truth. Every tag and every claim in both instructions \
must match the measured directions. Never state a change the measurements do not show (e.g. \
do not say "warmer" if temperature_delta_b is negative or negligible).
- Keep it to global tone/color only. Do NOT mention: local or object edits; scene content \
that is not relevant to the global color; impossible preservation claims (e.g. "keep skin \
exactly the same" while recoloring); aesthetic rankings such as "best" or "beautiful" unless \
they map to a named style recipe above.
- "concise" is literal and minimal (e.g. "Make it warmer, more muted, and lift the blacks."). \
"natural" is a looser phrasing of the same look (e.g. "Give it a soft warm matte feel with \
gentler colors."). They must describe the SAME edit.
- Prefer a style-bundle tag (e.g. matte, cinematic, teal-orange) when the measured effect is \
a recognizable composite look; otherwise use the directional tags."""


TEACHER_SYSTEM_PROMPT = build_teacher_system_prompt()


def summarize_behavior(behavior: dict) -> str:
    """Human-readable, signed summary of the measured behavior vector for the teacher prompt."""
    if not behavior:
        return "  (no measurable behavior recorded)"
    lines: list[str] = []
    for key in _SUMMARY_KEYS:
        if key not in behavior:
            continue
        val = float(behavior.get(key) or 0.0)
        words = _KEY_WORDS.get(key, {})
        if words:
            # Only call it a direction when it clears the SAME floor the validator uses to back
            # a tag; below that it is "negligible" so the teacher does not emit an unbackable tag.
            desc = words.get(1 if val > 0 else -1, "") if abs(val) >= _MIN_MEASURABLE_MOVE else "negligible"
        else:
            desc = ""  # context-only signal (e.g. clip_rate, neutral_drift)
        lines.append(f"  - {key} = {val:+.3f}" + (f"  -> {desc}" if desc else ""))
    return "\n".join(lines) if lines else "  (no measurable behavior recorded)"


def build_teacher_user_text(behavior: dict, candidate_tags) -> str:
    return (
        "MEASURED COLOR BEHAVIOR (ground truth; Lab-domain deltas from the deterministic probe. "
        "Your instruction MUST match these directions and MUST NOT claim any change not shown here):\n"
        + summarize_behavior(behavior)
        + "\n\nCANDIDATE STRUCTURED TAGS (from deterministic derivation; refine to the allowed "
        "vocabulary and the measured behavior):\n  "
        + json.dumps(list(candidate_tags or []))
        + "\n\nReturn ONLY the JSON object described in the system prompt."
    )


def _row_get(row) -> Callable:
    return row.get if isinstance(row, dict) else (lambda k, d=None: getattr(row, k, d))


def build_teacher_messages(row, *, attach_image: bool = True, min_image_edge: int = 256) -> list:
    """Build the OpenAI-style chat messages for one row (system + user[+image])."""
    get = _row_get(row)
    behavior = get("measured_behavior", {}) or {}
    candidate_tags = get("gold_tags", None)
    if not candidate_tags:
        candidate_tags = get("structured_tags", []) or []
    parts: list = []
    image_path = get("image_path", None)
    if attach_image and image_path and os.path.exists(str(image_path)):
        # PIL is only needed here; import lazily so the module loads without the frontier extra.
        from eval.frontier_client import encode_image

        b64, media = encode_image(str(image_path), min_image_edge)
        parts.append(openai_compat.image_part(f"data:{media};base64,{b64}"))
    parts.append(openai_compat.text_part(build_teacher_user_text(behavior, candidate_tags)))
    return [
        {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": parts},
    ]


def _norm_tag(s) -> str:
    """Case/separator-insensitive tag key ('bleach-bypass' == 'bleach bypass' == 'bleach_bypass')."""
    return re.sub(r"[\s_-]+", " ", str(s).strip().lower())


_KNOWN_TAGS_NORM = {_norm_tag(t): t for t in KNOWN_TAGS}


def _clean_tags(raw) -> list[str]:
    """Normalize teacher-emitted tags to the canonical known vocabulary (drop hallucinated tags).

    Matching is separator-insensitive so a compliant teacher that emits 'bleach-bypass' or
    'teal orange' still maps to the canonical 'bleach bypass' / 'teal-orange' instead of being
    silently dropped (which would also wrongly disable the style reverse-coverage bypass)."""
    out: list[str] = []
    for t in (raw or []):
        canon = _KNOWN_TAGS_NORM.get(_norm_tag(t))
        if canon and canon not in out:
            out.append(canon)
    return out


class TeacherClient:
    def __init__(self, model_clients_path: str | Path = "configs/model_clients.yaml",
                 *, attach_image: bool = True, min_image_edge: int = 256,
                 timeout: float = openai_compat.DEFAULT_TIMEOUT_S):
        self.path = Path(model_clients_path)
        self.attach_image = attach_image
        self.min_image_edge = min_image_edge
        self.timeout = timeout

    def _profile(self) -> Optional[dict]:
        if not self.path.exists():
            return None
        try:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            return None
        return (data.get("teacher_primary") or None)

    def is_available(self) -> bool:
        prof = self._profile()
        if not prof:
            return False
        if any(not prof.get(k) for k in _REQUIRED_PROFILE_KEYS):
            return False
        if str(prof.get("model_id", "")).lower() in _ALIASES:
            return False
        return True

    def generate(self, row: dict) -> dict:
        """Generate ``{gold_tags, concise, natural, provenance}`` for one accepted pair.

        Raises :class:`RequiresTeacher` if the profile is unpinned/aliased or its credential
        env vars are unset; :class:`TeacherGenerationError` if the API call ran but returned an
        unusable / unparseable response.
        """
        if not self.is_available():
            raise RequiresTeacher(
                "configs/model_clients.yaml teacher_primary is missing or uses an alias; "
                f"instruction text stays instruction_status={INSTRUCTION_STATUS_PENDING!r}. "
                "Pin a concrete teacher profile to enable."
            )
        prof = self._profile() or {}
        provider = str(prof.get("provider", "openai_compatible"))
        if provider not in ("openai_compatible", "openai"):
            raise RequiresTeacher(
                f"teacher provider {provider!r} is not supported by this client "
                "(openai_compatible only)."
            )
        try:
            base_url, api_key = openai_compat.resolve_endpoint(prof)
        except openai_compat.OpenAICompatError as exc:
            # Profile pinned but not runnable in this environment (env vars unset).
            raise RequiresTeacher(f"teacher credentials not available: {exc}") from exc

        messages = build_teacher_messages(
            row, attach_image=self.attach_image, min_image_edge=self.min_image_edge)
        rd = dict(prof.get("request_defaults") or {})
        try:
            client = openai_compat.build_client(base_url, api_key, timeout=self.timeout)
            res = openai_compat.chat_completion(
                client, str(prof["model_id"]), messages,
                max_tokens=int(rd.get("max_tokens", 1024)),
                temperature=rd.get("temperature", 0.7),
                reasoning_effort=openai_compat.effort_from_profile(prof))
        except openai_compat.OpenAICompatError as exc:
            raise TeacherGenerationError(f"teacher API call failed: {exc}") from exc

        if res.api_refusal or not res.text:
            raise TeacherGenerationError(
                f"teacher returned no usable text (finish_reason={res.finish_reason})")
        try:
            obj = openai_compat.parse_json_object(res.text)
        except openai_compat.OpenAICompatError as exc:
            raise TeacherGenerationError(str(exc)) from exc

        gold_tags = _clean_tags(obj.get("gold_tags") or obj.get("tags") or [])
        concise = str(obj.get("concise") or "").strip()
        natural = str(obj.get("natural") or "").strip()
        if not concise or not natural:
            raise TeacherGenerationError(
                f"teacher JSON missing concise/natural (keys={sorted(obj)})")

        provenance = {
            "teacher_provider": provider,
            "teacher_model_id": prof.get("model_id"),
            "teacher_endpoint_env": prof.get("endpoint_env"),
            "teacher_api_key_env": prof.get("api_key_env"),
            "teacher_prompt_version": prof.get("prompt_version"),
            "prompt_generation_batch_id": prof.get("batch_id"),
            "credential_profile": prof.get("credential_profile"),
            "effort": openai_compat.effort_from_profile(prof),
            "finish_reason": res.finish_reason,
            "usage": res.usage,
        }
        return {"gold_tags": gold_tags, "concise": concise, "natural": natural,
                "provenance": provenance}


def _messages_preview(messages: list) -> str:
    """Flatten chat messages to text for --dry-run inspection (image bytes elided)."""
    out: list[str] = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, str):
            body = content
        else:
            chunks = []
            for p in content or []:
                if p.get("type") == "text":
                    chunks.append(p.get("text", ""))
                elif p.get("type") == "image_url":
                    chunks.append("<image>")
            body = "\n".join(chunks)
        out.append(f"[{m.get('role')}]\n{body}")
    return "\n\n".join(out)


def generate_instructions_for_rows(
    rows, teacher: TeacherClient, *,
    judge_model_clients_path: Optional[str] = None,
    run_judge: bool = True,
    coverage_threshold: float = INSTRUCTION_COVERAGE_THRESHOLD,
    dry_run: bool = False,
    limit: Optional[int] = None,
    attach_image: bool = True,
    done_ids: Optional[set] = None,
    on_row: Optional[Callable[[dict], None]] = None,
) -> dict:
    """Run instruction generation end-to-end over ``rows`` and return a manifest.

    For each row: call the teacher, run the authoritative deterministic tag<->behavior gate,
    optionally run the non-authoritative judge language gate, and mark the row
    ``teacher_generated`` (accepted) or ``rejected_teacher``. ``dry_run`` builds the prompts
    without calling the API (so the wiring is verifiable offline). ``on_row`` is invoked with
    each per-row result dict as it completes (used by the runner for resumable JSONL output).
    Rows are dicts or objects exposing ``id``, ``gold_tags``, ``measured_behavior`` and
    (optionally) ``image_path`` / ``structured_tags``.
    """
    # Lazy: keep eval.judge_client out of this module's import surface (it is imported widely
    # via active_dataset for validate_tags_against_behavior).
    from eval import judge_client

    done_ids = set(done_ids or [])
    if not dry_run and not teacher.is_available():
        raise RequiresTeacher(
            "teacher_primary is not pinned/available; cannot generate instructions "
            "(use dry_run=True to inspect prompts offline).")

    judge_available = bool(
        run_judge and not dry_run and judge_client.is_available(judge_model_clients_path))

    prof = teacher._profile() or {}
    results: list[dict] = []
    counts = {"generated": 0, "rejected": 0, "error": 0, "dry_run": 0, "skipped": 0, "authored": 0}
    n_processed = 0

    for row in rows:
        get = _row_get(row)
        rid = get("id", None)
        if rid in done_ids:
            counts["skipped"] += 1
            continue
        # Source-authored rows (e.g. MMArt-PPR10K): the instruction is authoritative, so the
        # teacher is skipped entirely. Never spends, so it does not consume the API `limit`.
        authored = get("instruction", None)
        if authored and get("instruction_status", None) == INSTRUCTION_STATUS_AUTHORED:
            res = {"id": rid, "instruction_status": INSTRUCTION_STATUS_AUTHORED,
                   "concise": authored, "natural": get("instruction_natural", None),
                   "gold_tags": list(get("gold_tags", []) or []), "authored": True}
            counts["authored"] += 1
            results.append(res)
            if on_row:
                on_row(res)
            continue
        if limit is not None and n_processed >= limit:
            break
        n_processed += 1
        behavior = get("measured_behavior", {}) or {}

        if dry_run:
            msgs = build_teacher_messages(
                row, attach_image=attach_image, min_image_edge=teacher.min_image_edge)
            res = {"id": rid, "instruction_status": "dry_run",
                   "prompt_preview": _messages_preview(msgs)}
            counts["dry_run"] += 1
            results.append(res)
            if on_row:
                on_row(res)
            continue

        try:
            gen = teacher.generate(row)
        except (RequiresTeacher, TeacherGenerationError) as exc:
            res = {"id": rid, "instruction_status": INSTRUCTION_STATUS_REJECTED,
                   "error": f"{type(exc).__name__}: {exc}"}
            counts["error"] += 1
            results.append(res)
            if on_row:
                on_row(res)
            continue

        ok, issues = validate_tags_against_behavior(
            gen["gold_tags"], behavior, coverage_threshold=coverage_threshold)

        judge_result = None
        judge_pass = True
        if judge_available:
            lr = judge_client.score_instruction(
                gen["concise"], gen["natural"], gen["gold_tags"], behavior,
                model_clients_path=judge_model_clients_path)
            judge_result = {"status": lr.status, "reason": lr.reason, "details": lr.details}
            judge_pass = lr.status != "fail"  # non-authoritative: only a hard fail rejects

        accepted = ok and judge_pass
        status = INSTRUCTION_STATUS_GENERATED if accepted else INSTRUCTION_STATUS_REJECTED
        counts["generated" if accepted else "rejected"] += 1
        res = {
            "id": rid, "instruction_status": status,
            "concise": gen["concise"], "natural": gen["natural"], "gold_tags": gen["gold_tags"],
            "validation_ok": ok, "validation_issues": issues,
            "judge": judge_result, "provenance": gen["provenance"],
        }
        results.append(res)
        if on_row:
            on_row(res)

    return {
        "teacher_available": teacher.is_available(),
        "judge_available": judge_available,
        "dry_run": dry_run,
        "prompt_version": prof.get("prompt_version"),
        "batch_id": prof.get("batch_id"),
        "counts": counts,
        "n_processed": n_processed,
        "rows": results,
    }


def apply_instruction_result(sft_row, result: dict) -> None:
    """Apply a :func:`generate_instructions_for_rows` per-row result onto an ``SftRow``.

    Only a ``teacher_generated`` result writes instruction text + tags; a rejection records the
    status and leaves the (None) instruction so nothing is fabricated. dry_run/skipped/error
    rows leave the SftRow untouched except for the status where applicable.
    """
    status = result.get("instruction_status")
    if status == INSTRUCTION_STATUS_GENERATED:
        sft_row.instruction = result.get("concise")
        sft_row.instruction_natural = result.get("natural")
        sft_row.instruction_status = INSTRUCTION_STATUS_GENERATED
        if result.get("gold_tags"):
            sft_row.gold_tags = list(result["gold_tags"])
    elif status == INSTRUCTION_STATUS_REJECTED:
        sft_row.instruction_status = INSTRUCTION_STATUS_REJECTED
    # INSTRUCTION_STATUS_AUTHORED: leave the row untouched — assemble_active already wrote the
    # authoritative source instruction; the teacher was skipped.


def validate_tags_against_behavior(gold_tags: list[str], behavior: dict,
                                   coverage_threshold: float = 2.0) -> tuple[bool, list[str]]:
    """Bidirectional check: every explicit tag is backed by measured behavior, and every
    major measured behavior is mentioned (or would need allowed-unmentioned marking).
    Returns (ok, issues).
    """
    issues: list[str] = []
    # forward: tag -> behavior direction
    for tag in gold_tags:
        if tag in _TAG_BEHAVIOR:
            key, sign = _TAG_BEHAVIOR[tag]
            if sign * behavior.get(key, 0.0) < _MIN_MEASURABLE_MOVE:  # below a minimal measurable move
                issues.append(f"tag_not_backed:{tag}")
    # reverse: major measured behaviors that no tag mentions (skipped for style/composite rows)
    if set(gold_tags) & _STYLE_TAGS:
        return (len(issues) == 0), issues
    tagged_keys = {_TAG_BEHAVIOR[t][0] for t in gold_tags if t in _TAG_BEHAVIOR}
    major = {
        "temperature_delta_b": abs(behavior.get("temperature_delta_b", 0.0)),
        "tint_delta_a": abs(behavior.get("tint_delta_a", 0.0)),
        "mean_l_delta": abs(behavior.get("mean_l_delta", 0.0)),
        "chroma_delta": abs(behavior.get("chroma_delta", 0.0)),
        "contrast_l_spread_delta": abs(behavior.get("contrast_l_spread_delta", 0.0)),
    }
    for key, mag in major.items():
        if mag >= coverage_threshold and key not in tagged_keys:
            issues.append(f"unmentioned_behavior:{key}")
    return (len(issues) == 0), issues
