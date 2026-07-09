"""Generate teacher instructions for active/eval rows and cache them to JSONL.

Runs instruction generation end-to-end (docs/data_collection_plan.md "Instruction
Generation"): for each row it calls the pinned teacher (``teacher_primary``) to produce
``{gold_tags, concise, natural}``, runs the authoritative deterministic tag<->behavior gate,
and — when ``judge_primary`` is pinned + its env vars are set — the non-authoritative judge
language gate. Each row is written to ``--out`` as it completes, so the run is idempotent /
resumable (rows already present are skipped) and API spend happens once.

Gated + safe by default:
  * ``--dry-run`` builds and prints the exact prompts WITHOUT any network call — use this to
    verify the wiring without spending or needing credentials.
  * A live run requires ``configs/model_clients.yaml`` to pin ``teacher_primary`` AND the
    referenced env vars (TFY_BASE_URL / TFY_API_KEY) to be set, plus ``pip install
    'slm-eval[frontier]'`` for the openai SDK.

Usage:
    python -m scripts.generate_instructions --rows data/active_sft/active_rows.jsonl --dry-run
    python -m scripts.generate_instructions --rows data/active_sft/active_rows.jsonl --limit 50
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from data_pipeline.instruction_gen import TeacherClient, generate_instructions_for_rows


def _load_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_done(path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rid = json.loads(line).get("id")
            except json.JSONDecodeError:
                continue
            if rid is not None:
                done.add(rid)
    return done


def run(rows_path: str, out_path: str, config_path: str, limit: Optional[int],
        run_judge: bool, attach_image: bool, dry_run: bool) -> int:
    rows = _load_rows(rows_path)
    teacher = TeacherClient(config_path, attach_image=attach_image)

    if not dry_run and not teacher.is_available():
        print("[instr] teacher_primary not pinned/available in "
              f"{config_path}; nothing generated. Pin teacher_primary (and set its env vars) "
              "or pass --dry-run to inspect prompts offline.")
        return 2

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    done = set() if dry_run else _load_done(out_path)
    if done:
        print(f"[instr] resuming: {len(done)} rows already in {out_path}")

    judge_path = config_path if run_judge else None
    tally = {"generated": 0, "rejected": 0, "error": 0, "dry_run": 0}

    mode = "a" if (not dry_run and os.path.exists(out_path)) else "w"
    sink = None if dry_run else open(out_path, mode, encoding="utf-8")

    def on_row(res: dict) -> None:
        status = res.get("instruction_status", "?")
        tally[status] = tally.get(status, 0) + 1
        if dry_run:
            print(f"\n===== row {res.get('id')} (dry-run prompt) =====")
            print(res.get("prompt_preview", ""))
            return
        sink.write(json.dumps(res, sort_keys=True) + "\n")
        sink.flush()
        if status == "rejected_teacher" and res.get("error"):
            print(f"  ! {res.get('id')}: {res.get('error')}")
        else:
            extra = f" issues={res.get('validation_issues')}" if res.get("validation_issues") else ""
            print(f"  {res.get('id')}: {status} tags={res.get('gold_tags')}{extra}")

    print(f"[instr] {len(rows)} rows; teacher={'dry-run' if dry_run else teacher.is_available()}; "
          f"judge={'on' if run_judge else 'off'}; out={out_path if not dry_run else '(none)'}")
    try:
        manifest = generate_instructions_for_rows(
            rows, teacher, judge_model_clients_path=judge_path, run_judge=run_judge,
            dry_run=dry_run, limit=limit, attach_image=attach_image, done_ids=done, on_row=on_row)
    finally:
        if sink is not None:
            sink.close()

    if not dry_run:
        manifest_path = os.path.join(os.path.dirname(out_path) or ".",
                                     "instruction_gen_manifest.json")
        summary = {k: v for k, v in manifest.items() if k != "rows"}
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"[instr] done: {manifest['counts']} -> {out_path} (manifest: {manifest_path})")
    else:
        print(f"[instr] dry-run complete: {manifest['counts']}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate teacher instructions (cached to JSONL).")
    ap.add_argument("--rows", default="data/active_sft/active_rows.jsonl")
    ap.add_argument("--out", default="data/active_sft/instructions.jsonl")
    ap.add_argument("--config", default="configs/model_clients.yaml")
    ap.add_argument("--limit", type=int, default=None, help="first N not-yet-done rows")
    ap.add_argument("--no-judge", action="store_true", help="skip the judge language gate")
    ap.add_argument("--no-image", action="store_true", help="text-only teacher (skip source image)")
    ap.add_argument("--dry-run", action="store_true", help="build+print prompts, no API call")
    args = ap.parse_args(argv)
    return run(args.rows, args.out, args.config, args.limit,
               run_judge=not args.no_judge, attach_image=not args.no_image, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
