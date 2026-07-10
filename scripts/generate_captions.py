"""Generate the caption→AttributeSpec interpreter corpus (teacher captioner; ADR 0026).

For each supported active LUT (which already carries a ``behavior_v2`` ``measured_behavior``), the
teacher writes several stylistically diverse captions (:data:`data_pipeline.captioning.CAPTION_STYLES`
— literal / metaphor / mood / concept / slang) that all map to that LUT's serialized
``attribute_spec_text`` (the grounded target). The output ``caption_rows.jsonl`` is the P5 Interpreter
training corpus (``caption -> AttributeSpec + route``) — a NEW versioned artifact; the frozen LUT/
image corpus and tokenizer are never touched.

Resumable/idempotent: teacher results cache to ``--out`` keyed by ``lut_id``; re-running skips done
ids and re-assembles. ``--dry-run`` prints the prompts with no API call.

Gateway: same ``teacher_primary`` profile + gating as the supported/unsupported teacher; requires
``TFY_BASE_URL`` + ``TFY_API_KEY`` at call time (else it prints an actionable hand-off message).

Usage:
    python -m scripts.generate_captions --dry-run --limit 3
    python -m scripts.generate_captions --limit 500          # bounded batch (resumable)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import yaml

from data_pipeline.captioning import (
    CAPTION_STYLES,
    build_caption_system_prompt,
    build_caption_user_text,
    caption_target_text,
    validate_caption,
)
from data_pipeline.errors import RequiresTeacher, TeacherGenerationError
from eval import openai_compat
from eval.refuse_taxonomy import ROUTE_GRADE
from sft.example import resolve_image

_ACTIVE_ROWS = "data/active_sft/active_rows.jsonl"
_OUT = "data/active_sft/caption_rows.jsonl"
_CACHE = "data/active_sft/caption_cache.jsonl"
_ALIASES = {"latest", "stable", "current", "default", "auto"}
_REQUIRED_PROFILE_KEYS = ("provider", "model_id", "endpoint_env", "api_key_env",
                          "prompt_version", "batch_id")


class CaptionTeacherClient:
    """Same gateway/profile + gating as the supported teacher (``teacher_primary``)."""

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
        return data.get("teacher_primary") or None

    def is_available(self) -> bool:
        prof = self._profile()
        if not prof or any(not prof.get(k) for k in _REQUIRED_PROFILE_KEYS):
            return False
        return str(prof.get("model_id", "")).lower() not in _ALIASES

    def build_messages(self, row: dict, styles: tuple[str, ...]) -> list:
        mb = row.get("measured_behavior") or {}
        title = row.get("lut_title") or row.get("title")
        system = build_caption_system_prompt(len(styles))
        user_text = build_caption_user_text(mb, title=title, styles=styles)
        parts: list = []
        img = row.get("image_path")
        if self.attach_image and img:
            resolved = resolve_image(img)
            if os.path.exists(resolved):
                from eval.frontier_client import encode_image
                b64, media = encode_image(resolved, self.min_image_edge)
                parts.append(openai_compat.image_part(f"data:{media};base64,{b64}"))
        parts.append(openai_compat.text_part(user_text))
        return [{"role": "system", "content": system},
                {"role": "user", "content": parts}]

    def generate(self, row: dict, styles: tuple[str, ...] = CAPTION_STYLES) -> dict:
        if not self.is_available():
            raise RequiresTeacher("teacher_primary missing/aliased; cannot caption.")
        prof = self._profile() or {}
        try:
            base_url, api_key = openai_compat.resolve_endpoint(prof)
        except openai_compat.OpenAICompatError as exc:
            raise RequiresTeacher(f"teacher credentials not available: {exc}") from exc
        messages = self.build_messages(row, styles)
        rd = dict(prof.get("request_defaults") or {})
        try:
            client = openai_compat.build_client(base_url, api_key, timeout=self.timeout)
            res = openai_compat.chat_completion(
                client, str(prof["model_id"]), messages,
                max_tokens=int(rd.get("max_tokens", 1024)), temperature=rd.get("temperature"),
                reasoning_effort=openai_compat.effort_from_profile(prof))
        except openai_compat.OpenAICompatError as exc:
            raise TeacherGenerationError(f"teacher API call failed: {exc}") from exc
        if res.api_refusal or not res.text:
            raise TeacherGenerationError(f"teacher returned no usable text (finish={res.finish_reason})")
        try:
            obj = openai_compat.parse_json_object(res.text)
        except openai_compat.OpenAICompatError as exc:
            raise TeacherGenerationError(str(exc)) from exc
        caps = obj.get("captions") or {}
        if not isinstance(caps, dict) or not caps:
            raise TeacherGenerationError(f"teacher JSON missing 'captions' (keys={sorted(obj)})")
        return {"captions": {k: str(v).strip() for k, v in caps.items()},
                "provenance": {"teacher_model_id": prof.get("model_id"),
                               "prompt_version": prof.get("prompt_version"),
                               "batch_id": prof.get("batch_id")}}


def _supported_rows(path: str) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r.get("is_supported") and (r.get("measured_behavior"))]


def _load_done(path: str) -> dict:
    done: dict[str, dict] = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("lut_id"):
                    done[r["lut_id"]] = r
    return done


def _caption_rows_from(cache: dict) -> list[dict]:
    """Flatten cached per-LUT caption sets into (caption -> attribute_spec_text) interpreter rows."""
    out: list[dict] = []
    for lut_id, rec in cache.items():
        if rec.get("status") != "generated":
            continue
        target = rec["attribute_spec_text"]
        for style, caption in (rec.get("captions") or {}).items():
            ok, _ = validate_caption(caption)
            if not ok:
                continue
            out.append({
                "id": f"cap_{lut_id}_{style}", "source_lut_id": lut_id,
                "caption": caption, "style": style, "route": ROUTE_GRADE,
                "attribute_spec_text": target,
            })
    return out


def run(active_rows: str, out: str, cache_path: str, config: str, limit: Optional[int],
        attach_image: bool, dry_run: bool) -> int:
    rows = _supported_rows(active_rows)
    if limit is not None:
        rows = rows[:limit]
    teacher = CaptionTeacherClient(config, attach_image=attach_image)
    if not dry_run and not teacher.is_available():
        print(f"[caption] teacher_primary not available in {config}; pass --dry-run to inspect. "
              f"(Set TFY_BASE_URL + TFY_API_KEY to run, or HAND OFF.)")
        return 2

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cache = {} if dry_run else _load_done(cache_path)
    if cache:
        print(f"[caption] resuming: {len(cache)} LUTs already captioned in {cache_path}")
    sink = None if dry_run else open(cache_path, "a", encoding="utf-8")
    tally = {"generated": 0, "error": 0, "skipped": 0, "dry_run": 0}

    for row in rows:
        lut_id = row.get("source_lut_id") or row.get("id")
        if lut_id in cache:
            tally["skipped"] += 1
            continue
        if dry_run:
            msgs = teacher.build_messages(row, CAPTION_STYLES)
            print(f"\n===== {lut_id} =====")
            print("SYSTEM:", msgs[0]["content"][:160], "...")
            print("USER:", [p for p in msgs[1]["content"] if p.get("type") == "text"][0]["text"][:400])
            tally["dry_run"] += 1
            continue
        try:
            gen = teacher.generate(row)
            rec = {"lut_id": lut_id, "status": "generated", "captions": gen["captions"],
                   "attribute_spec_text": caption_target_text(row["measured_behavior"]),
                   "provenance": gen["provenance"]}
            tally["generated"] += 1
        except Exception as exc:  # noqa: BLE001
            rec = {"lut_id": lut_id, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            tally["error"] += 1
        cache[lut_id] = rec
        sink.write(json.dumps(rec, sort_keys=True) + "\n")
        sink.flush()
        print(f"  {lut_id}: {rec['status']} {list((rec.get('captions') or {}).keys())}")
    if sink:
        sink.close()

    print(f"[caption] {tally}")
    if dry_run:
        return 0
    caption_rows = _caption_rows_from(cache)
    with open(out, "w", encoding="utf-8") as fh:
        for r in caption_rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")
    Path("data/active_sft/caption_gen_manifest.json").write_text(json.dumps(
        {"lut_count": len(cache), "caption_rows": len(caption_rows), "styles": list(CAPTION_STYLES),
         "tally": tally, "route": ROUTE_GRADE}, indent=2), encoding="utf-8")
    print(f"[caption] wrote {len(caption_rows)} caption rows -> {out}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate caption->AttributeSpec interpreter corpus.")
    ap.add_argument("--active-rows", default=_ACTIVE_ROWS)
    ap.add_argument("--out", default=_OUT)
    ap.add_argument("--cache", default=_CACHE)
    ap.add_argument("--config", default="configs/model_clients.yaml")
    ap.add_argument("--limit", type=int, default=None, help="first N supported LUTs (resumable)")
    ap.add_argument("--no-image", action="store_true", help="text-only teacher (skip source image)")
    ap.add_argument("--dry-run", action="store_true", help="print prompts, no API call")
    args = ap.parse_args(argv)
    return run(args.active_rows, args.out, args.cache, args.config, args.limit,
               attach_image=not args.no_image, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
