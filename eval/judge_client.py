"""LLM/VLM-as-judge (L8) — gated by configs/model_clients.yaml.

Two entry points, both non-authoritative (they never convert a deterministic L0-L7 fail into
a pass — docs/eval_harness_implementation.md "LLM/VLM Judge", ADR 0004):

* :func:`score` — the eval-harness L8 judge that scores a model's produced LUT output. It needs
  a decoded LUT / graded image, so in this decode-disabled spine it stays ``not_evaluated``.
* :func:`score_instruction` — the language / semantic quality gate on a teacher-generated
  instruction (concision, no local/semantic claims, no content leakage, tag<->prompt
  consistency). This is TEXT-only and runs as part of instruction generation
  (:mod:`data_pipeline.instruction_gen`) when ``judge_primary`` is pinned and its credential
  env vars are set.

Both are blocked until ``configs/model_clients.yaml`` pins ``judge_primary`` with a concrete
(non-alias) ``model_id`` + endpoint/api-key env vars + prompt/batch versions, and those env
vars are actually set.
"""

from __future__ import annotations

import os
from typing import Optional

import yaml

from . import openai_compat
from .schemas import LayerResult

# Kept in sync with data_pipeline.instruction_gen._ALIASES so the same model_id is judged
# concrete-or-alias identically by teacher and judge (both hit the same gateway).
_ALIASES = {"latest", "stable", "current", "default", "auto"}

_INSTRUCTION_JUDGE_LAYER = "L8_instruction_judge"


class JudgeUnavailable(RuntimeError):
    pass


def _load_profile(model_clients_path: Optional[str]) -> dict:
    if not model_clients_path or not os.path.exists(model_clients_path):
        return {}
    try:
        with open(model_clients_path, "r", encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:  # noqa: BLE001
        return {}
    return cfg.get("judge_primary") or {}


def is_available(model_clients_path: Optional[str]) -> bool:
    """True iff judge_primary is fully pinned in model_clients.yaml (concrete model id,
    endpoint/api-key env vars set, prompt/batch versions present)."""
    jp = _load_profile(model_clients_path)
    if not jp:
        return False
    model_id = str(jp.get("model_id", "")).strip()
    if not model_id or model_id.lower() in _ALIASES:
        return False
    for key in ("provider", "endpoint_env", "api_key_env", "prompt_version", "batch_id"):
        if not jp.get(key):
            return False
    # secrets referenced by env var name only; env vars must actually be set
    if not os.environ.get(str(jp.get("endpoint_env"))):
        return False
    if not os.environ.get(str(jp.get("api_key_env"))):
        return False
    return True


# --- instruction-quality judge (runs during instruction generation) ----------------
def build_instruction_judge_system_prompt() -> str:
    return """You are a language and semantic quality reviewer for GLOBAL color-grading edit \
instructions in a training dataset. You are NOT the authority on whether the color direction \
is correct — that is verified deterministically elsewhere. Review ONLY the language:

Flag an instruction (pass=false) if any of these hold:
- it references a LOCAL, regional, object, or subject edit (a global LUT cannot do these);
- it mentions scene content that is not relevant to the global color look;
- it makes an impossible preservation claim (e.g. "keep skin identical" while recoloring);
- it uses an aesthetic ranking such as "best"/"beautiful" that is not a named style;
- the "concise" and "natural" phrasings describe DIFFERENT edits, or contradict the gold_tags.

Output EXACTLY ONE JSON object and nothing else:
{"pass": true|false, "issues": ["short_slug", ...], "notes": "one short sentence"}
Use pass=true with an empty issues list when the instruction is clean."""


def score_instruction(concise: str, natural: str, gold_tags, measured_behavior: dict,
                      model_clients_path: Optional[str] = None) -> LayerResult:
    """Non-authoritative language/semantic gate on a teacher-generated instruction.

    Returns ``not_evaluated`` when the judge is unpinned/unavailable; otherwise ``pass`` or
    ``fail`` with the parsed verdict recorded in ``details``. A ``fail`` NEVER overrides the
    deterministic tag<->behavior gate; callers treat it as advisory (rejecting only on a hard
    fail, per data_pipeline.instruction_gen.generate_instructions_for_rows)."""
    if not is_available(model_clients_path):
        return LayerResult(
            layer=_INSTRUCTION_JUDGE_LAYER, status="not_evaluated",
            reason="judge_gated:model_clients.yaml judge_primary not pinned or env unset")
    prof = _load_profile(model_clients_path)
    provider = str(prof.get("provider", "openai_compatible"))
    if provider not in ("openai_compatible", "openai"):
        return LayerResult(
            layer=_INSTRUCTION_JUDGE_LAYER, status="not_evaluated",
            reason=f"judge_provider_unsupported:{provider}")

    # Import the teacher's behavior summarizer lazily to avoid a hard eval->data_pipeline dep.
    try:
        from data_pipeline.instruction_gen import summarize_behavior
        behavior_text = summarize_behavior(measured_behavior or {})
    except Exception:  # noqa: BLE001
        behavior_text = str(measured_behavior or {})

    user_text = (
        f'gold_tags: {list(gold_tags or [])}\n'
        f'concise: "{concise}"\n'
        f'natural: "{natural}"\n\n'
        "measured behavior (for consistency checking only):\n" + behavior_text +
        "\n\nReturn ONLY the JSON verdict object.")
    messages = [
        {"role": "system", "content": build_instruction_judge_system_prompt()},
        {"role": "user", "content": [openai_compat.text_part(user_text)]},
    ]
    rd = dict(prof.get("request_defaults") or {})
    try:
        base_url, api_key = openai_compat.resolve_endpoint(prof)
        client = openai_compat.build_client(base_url, api_key)
        res = openai_compat.chat_completion(
            client, str(prof["model_id"]), messages,
            max_tokens=int(rd.get("max_tokens", 512)),
            temperature=rd.get("temperature"),  # None -> omit (some models deprecate/forbid it)
            reasoning_effort=openai_compat.effort_from_profile(prof))
        verdict = openai_compat.parse_json_object(res.text)
    except openai_compat.OpenAICompatError as exc:
        # A judge transport/parse failure must not fabricate a pass or a fail.
        return LayerResult(layer=_INSTRUCTION_JUDGE_LAYER, status="not_evaluated",
                           reason=f"judge_error:{exc}")

    passed = bool(verdict.get("pass", False))
    issues = list(verdict.get("issues", []) or [])
    return LayerResult(
        layer=_INSTRUCTION_JUDGE_LAYER,
        status="pass" if passed else "fail",
        reason=None if passed else ("judge_flagged:" + ",".join(str(i) for i in issues)),
        details={
            "verdict": verdict,
            "authority": "non_authoritative",
            "provenance": _provenance(prof, res.finish_reason, res.usage),
        },
    )


# --- eval-harness L8 output judge (decode-gated) ------------------------------------
def _graded_image_path(row) -> Optional[str]:
    """A decoded/graded image to show the output judge, if the pipeline produced one.

    None in the decode-disabled spine (no decoder -> no graded artifact), which is why
    :func:`score` returns ``not_evaluated`` there."""
    get = row.get if isinstance(row, dict) else (lambda k, d=None: getattr(row, k, d))
    return get("graded_image_path", None)


def score(row, parsed_output, deterministic_results, model_clients_path=None) -> LayerResult:  # noqa: ANN001
    """Return the L8 judge LayerResult for a model output (recorded, non-authoritative).

    In this decode-disabled spine there is no graded image to score, so this returns
    ``not_evaluated``. The HTTP path (:func:`_run_output_judge`) is wired and taken only once a
    decoder produces a graded image."""
    if not is_available(model_clients_path):
        return LayerResult(
            layer="L8_judge",
            status="not_evaluated",
            reason="judge_gated:model_clients.yaml judge_primary not pinned",
        )
    graded = _graded_image_path(row)
    if not graded or not os.path.exists(str(graded)):
        # Available but nothing to score yet (no decoded artifact).
        return LayerResult(
            layer="L8_judge",
            status="not_evaluated",
            reason="judge_not_run_in_decode_disabled_spine",
        )
    return _run_output_judge(row, parsed_output, deterministic_results, model_clients_path, graded)


def build_output_judge_system_prompt() -> str:
    return """You are a non-authoritative quality reviewer for a global color-grade applied to \
an image. Deterministic checks already decided correctness; you only add a language/perceptual \
read. Judge: does the graded image match the instruction's intended global look, without \
introducing local artifacts or claims a global LUT cannot support?

Output EXACTLY ONE JSON object: {"pass": true|false, "issues": [...], "notes": "one sentence"}."""


def _run_output_judge(row, parsed_output, deterministic_results, model_clients_path,
                      graded_image_path) -> LayerResult:  # noqa: ANN001
    """HTTP path for the L8 output judge (only reachable once decode is enabled)."""
    prof = _load_profile(model_clients_path)
    get = row.get if isinstance(row, dict) else (lambda k, d=None: getattr(row, k, d))
    instruction = get("instruction", "") or ""
    user_text = (f'instruction: "{instruction}"\n'
                 "The graded image is attached. Return ONLY the JSON verdict object.")
    from .frontier_client import encode_image

    b64, media = encode_image(str(graded_image_path), 256)
    messages = [
        {"role": "system", "content": build_output_judge_system_prompt()},
        {"role": "user", "content": [
            openai_compat.image_part(f"data:{media};base64,{b64}"),
            openai_compat.text_part(user_text),
        ]},
    ]
    rd = dict(prof.get("request_defaults") or {})
    try:
        base_url, api_key = openai_compat.resolve_endpoint(prof)
        client = openai_compat.build_client(base_url, api_key)
        res = openai_compat.chat_completion(
            client, str(prof["model_id"]), messages,
            max_tokens=int(rd.get("max_tokens", 512)), temperature=rd.get("temperature"),  # None -> omit
            reasoning_effort=openai_compat.effort_from_profile(prof))
        verdict = openai_compat.parse_json_object(res.text)
    except openai_compat.OpenAICompatError as exc:
        return LayerResult(layer="L8_judge", status="not_evaluated", reason=f"judge_error:{exc}")

    passed = bool(verdict.get("pass", False))
    return LayerResult(
        layer="L8_judge",
        status="pass" if passed else "fail",
        reason=None if passed else ("judge_flagged:" + ",".join(
            str(i) for i in verdict.get("issues", []) or [])),
        details={"verdict": verdict, "authority": "non_authoritative",
                 "provenance": _provenance(prof, res.finish_reason, res.usage)},
    )


def _provenance(prof: dict, finish_reason, usage) -> dict:
    return {
        "judge_provider": prof.get("provider"),
        "judge_model_id": prof.get("model_id"),
        "judge_endpoint_env": prof.get("endpoint_env"),
        "judge_api_key_env": prof.get("api_key_env"),
        "judge_prompt_version": prof.get("prompt_version"),
        "judge_batch_id": prof.get("batch_id"),
        "credential_profile": prof.get("credential_profile"),
        "effort": openai_compat.effort_from_profile(prof),
        "finish_reason": finish_reason,
        "usage": usage,
    }
