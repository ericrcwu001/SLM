"""Generate the clarify + out_of_gamut ROUTE SUPPLEMENT for the Stage-1 interpreter (ADR 0023).

The interpreter must route ``{grade, clarify, refuse{out_of_scope, out_of_gamut}}``. The active
corpus already carries grade rows (captioned separately) and ``out_of_scope`` refusals, but has
**zero clarify and zero out_of_gamut** training rows. This script generates them *additively*:

  - clarify (``underspecified_intent``) → ``route=clarify``
  - out_of_gamut (infrared_false_color / pure_primary_cast / hue_rotation) →
    ``route=refuse | refuse=out_of_gamut``

These are pure natural-language → route decisions (no LUT grounding, no image), so we reuse the
teacher prompts + validator from :mod:`data_pipeline.unsupported_gen` text-only.

**Why a separate script (not ``scripts.generate_unsupported``):** that script's ``build_plan``
assigns categories *positionally* (``buckets[i % len(buckets)]``) and its ``_assemble`` rewrites
``active_rows.jsonl`` in place. Re-running it to add clarify would (a) never produce clarify (no
bucket) and (b) re-shuffle the versioned 272-row out_of_scope corpus (id↔category drift). Instead
we write to a **separate** cache/output with **fresh id prefixes** (``unsup_clarify_*`` /
``unsup_gamut_*``) and each row gets its own ``split_unit_id`` (unique per synthetic prompt — these
are independent single requests with no shared identity). ``active_rows.jsonl`` is never touched;
``scripts.build_interpreter_corpus`` reads this file to add the two missing routes.

Resume discipline mirrors the Phase-0 captioner fix: only ``status=="generated"`` counts as done
(error/partial rows retry), and a cred-less run hands off (return 2) instead of writing error rows.

Usage:
    python -m scripts.generate_route_supplement --dry-run --n-clarify 4 --n-gamut 6
    python -m scripts.generate_route_supplement --n-clarify 150 --n-gamut 150
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from data_pipeline.unsupported_gen import (
    UnsupportedTeacherClient,
    build_messages,
    validate_unsupported_prompt,
)
from eval import openai_compat
from eval.refuse_taxonomy import (
    CLARIFY_CATEGORIES,
    OUT_OF_GAMUT_CATEGORIES,
    refuse_kind_for_category,
    route_for_category,
)

_OUT = "data/active_sft/route_supplement_rows.jsonl"
_CACHE = "data/active_sft/route_supplement_cache.jsonl"
_MANIFEST = "data/active_sft/route_supplement_manifest.json"


def build_supplement_plan(n_clarify: int, n_gamut: int) -> list[dict]:
    """Deterministic plan over clarify + out_of_gamut categories; fresh ids, unique split units."""
    plan: list[dict] = []
    clarify_cats = list(CLARIFY_CATEGORIES)
    for i in range(n_clarify):
        cat = clarify_cats[i % len(clarify_cats)]
        rid = f"unsup_clarify_{i + 1:06d}"
        plan.append({"id": rid, "category": cat, "mixed": False,
                     "route": route_for_category(cat), "refuse_kind": refuse_kind_for_category(cat),
                     "split_unit_id": f"unsup:{rid}"})
    gamut_cats = list(OUT_OF_GAMUT_CATEGORIES)
    for i in range(n_gamut):
        cat = gamut_cats[i % len(gamut_cats)]
        rid = f"unsup_gamut_{i + 1:06d}"
        plan.append({"id": rid, "category": cat, "mixed": False,
                     "route": route_for_category(cat), "refuse_kind": refuse_kind_for_category(cat),
                     "split_unit_id": f"unsup:{rid}"})
    return plan


def _supp_row(item: dict, prompt: str) -> dict:
    """An interpreter-corpus-friendly supplement row (same keys build_interpreter_corpus reads
    off the active out_of_scope rows: instruction_natural + route + refuse_kind + split_unit_id)."""
    return {"id": item["id"], "is_supported": False, "source_family": "unsupported_teacher",
            "instruction": prompt, "instruction_natural": prompt,
            "route": item["route"], "refuse_kind": item["refuse_kind"],
            "unsupported_category": item["category"], "split_unit_id": item["split_unit_id"]}


def _load_done(path: str) -> dict:
    """Only ``status=="generated"`` rows count as done (error/partial rows retry on resume)."""
    done: dict[str, dict] = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("id") and r.get("status") == "generated":
                    done[r["id"]] = r
    return done


def _endpoint_ready(teacher: UnsupportedTeacherClient) -> tuple[bool, str]:
    if not teacher.is_available():
        return False, "teacher_primary profile missing/aliased in config"
    prof = teacher._profile() or {}
    try:
        openai_compat.resolve_endpoint(prof)
    except openai_compat.OpenAICompatError as exc:
        return False, str(exc)
    return True, ""


def _write_atomic(path: str, rows: list[dict]) -> None:
    """Backup + .tmp + atomic replace (repo convention for a fresh JSONL artifact)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        shutil.copy2(p, p.with_suffix(p.suffix + ".bak_pre_build"))
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, p)


def _assemble(cache: dict, plan: list[dict]) -> tuple[list[dict], dict]:
    by_id = {p["id"]: p for p in plan}
    rows: list[dict] = []
    counts = {"clarify": 0, "out_of_gamut": 0}
    for r in cache.values():
        if r.get("status") != "generated" or r.get("id") not in by_id:
            continue
        item = by_id[r["id"]]
        rows.append(_supp_row(item, r["prompt"]))
        counts["clarify" if item["route"] == "clarify" else "out_of_gamut"] += 1
    return rows, counts


def run(out: str, cache_path: str, config: str, n_clarify: int, n_gamut: int,
        limit: Optional[int], dry_run: bool) -> int:
    plan = build_supplement_plan(n_clarify, n_gamut)
    teacher = UnsupportedTeacherClient(config, attach_image=False)  # text-only: no image grounding
    if not dry_run:
        ready, why = _endpoint_ready(teacher)
        if not ready:
            print(f"[supplement] teacher not ready: {why}. Set TFY_BASE_URL + TFY_API_KEY to run, "
                  f"or HAND OFF (--dry-run to inspect). No rows written.")
            return 2

    cache = {} if dry_run else _load_done(cache_path)
    if cache:
        print(f"[supplement] resuming: {len(cache)} rows already in {cache_path}")
    tally = {"generated": 0, "rejected": 0, "error": 0, "skipped": 0, "dry_run": 0}

    pending = [it for it in plan if it["id"] not in cache]
    tally["skipped"] = len(plan) - len(pending)
    if limit is not None:
        pending = pending[:limit]

    if dry_run:
        for item in pending:
            msgs = build_messages(item, attach_image=False)
            print(f"\n===== {item['id']} [{item['category']} -> route={item['route']}"
                  f"{'/' + item['refuse_kind'] if item['refuse_kind'] else ''}] =====")
            print("SYSTEM:", msgs[0]["content"][:160], "...")
            print("USER:", [p for p in msgs[1]["content"] if p.get("type") == "text"][0]["text"])
            tally["dry_run"] += 1
        print(f"[supplement] {tally}")
        return 0

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as sink:
        for item in pending:
            rid = item["id"]
            try:
                gen = teacher.generate(item)
                ok, issues = validate_unsupported_prompt(gen["prompt"], item)
                rec = {"id": rid, "status": "generated" if ok else "rejected",
                       "prompt": gen["prompt"], "category": item["category"],
                       "route": item["route"], "refuse_kind": item["refuse_kind"],
                       "validation_ok": ok, "validation_issues": issues,
                       "provenance": gen.get("provenance")}
                tally["generated" if ok else "rejected"] += 1
            except Exception as exc:  # noqa: BLE001
                rec = {"id": rid, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
                tally["error"] += 1
            cache[rid] = rec
            sink.write(json.dumps(rec, sort_keys=True) + "\n")
            sink.flush()
            note = f" issues={rec.get('validation_issues')}" if rec.get("validation_issues") else ""
            print(f"  {rid}: {rec['status']}{note}  {rec.get('prompt', '')[:70]}")

    print(f"[supplement] {tally}")
    rows, counts = _assemble(cache, plan)
    _write_atomic(out, rows)
    Path(_MANIFEST).write_text(json.dumps(
        {"plan_size": len(plan), "n_clarify": n_clarify, "n_gamut": n_gamut,
         "supplement_rows": len(rows), "by_route": counts, "tally": tally,
         "cache_total": len(cache)}, indent=2), encoding="utf-8")
    print(f"[supplement] wrote {len(rows)} rows ({counts}) -> {out}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate clarify + out_of_gamut interpreter route rows.")
    ap.add_argument("--out", default=_OUT)
    ap.add_argument("--cache", default=_CACHE)
    ap.add_argument("--config", default="configs/model_clients.yaml")
    ap.add_argument("--n-clarify", type=int, default=150)
    ap.add_argument("--n-gamut", type=int, default=150)
    ap.add_argument("--limit", type=int, default=None, help="first N not-yet-done plan items")
    ap.add_argument("--dry-run", action="store_true", help="print prompts, no API call")
    args = ap.parse_args(argv)
    return run(args.out, args.cache, args.config, args.n_clarify, args.n_gamut,
               args.limit, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
