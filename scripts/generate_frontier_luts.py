"""Generate prompted-frontier raw-.cube LUTs and cache them to JSONL.

Calls the frontier models (claude-opus-4-8 / claude-sonnet-5) once per eval row via
eval.frontier_client and appends each raw response to ``data/eval/frontier_<name>.jsonl``
as ``{"row_id", "model", "model_id", "text", "provenance"}``. The scorer
(eval.run_frontier_eval) then replays from these files, so API spend happens once and
re-runs are free + deterministic.

Idempotent / resumable: rows already present in a model's JSONL are skipped, so re-running
after an interruption only fills the gaps. Prints a live valid-.cube tally per model — how
often a raw 17^3 .cube even parses is itself a headline baseline result.

Usage:
    python -m scripts.generate_frontier_luts --rows data/eval/smoke_rows.jsonl --limit 10
    python -m scripts.generate_frontier_luts --rows ... --models opus_4_8 --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from eval.cube_parser import parse_frontier_cube
from eval.frontier_client import FrontierClient
from eval.schemas import load_rows


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
                rid = json.loads(line).get("row_id")
            except json.JSONDecodeError:
                continue
            if rid is not None:
                done.add(rid)
    return done


def run(rows_path: str, out_dir: str, limit: Optional[int], model_names: Optional[list[str]],
        config_path: str) -> None:
    client = FrontierClient(config_path)
    rows = load_rows(rows_path)
    if limit is not None:
        rows = rows[:limit]

    models = client.models
    if model_names:
        models = [m for m in models if m["name"] in set(model_names)]
    if not models:
        raise SystemExit(f"no matching models (asked {model_names}, have {[m['name'] for m in client.models]})")

    os.makedirs(out_dir, exist_ok=True)
    print(f"[generate] {len(rows)} rows x {len(models)} model(s); out={out_dir}")

    for m in models:
        out_path = os.path.join(out_dir, f"frontier_{m['name']}.jsonl")
        done = _load_done(out_path)
        valid = attempts = 0
        print(f"\n=== {m['name']} ({m['model_id']}, effort={m.get('effort','medium')}) ===")
        with open(out_path, "a", encoding="utf-8") as out:
            for row in rows:
                if row.id in done:
                    continue
                img = row.image_path
                if not img or not os.path.exists(img):
                    print(f"  ! {row.id}: image not found ({img}); skipping")
                    continue
                try:
                    res = client.generate(m, img, row.instruction)
                except Exception as exc:  # noqa: BLE001 - log + continue so the row retries next run
                    print(f"  ! {row.id}: generation error: {exc}")
                    continue

                parsed = parse_frontier_cube(res.text)
                attempts += 1
                valid += int(parsed.kind == "raw_lut")
                out.write(json.dumps({
                    "row_id": row.id, "model": m["name"], "model_id": m["model_id"],
                    "text": res.text, "provenance": res.provenance,
                }) + "\n")
                out.flush()
                toks = res.provenance.get("output_tokens")
                print(f"  {row.id}: kind={parsed.kind:11s} tags={row.gold_tags} "
                      f"out_tokens={toks} valid={valid}/{attempts}"
                      + (f" errs={parsed.errors}" if parsed.errors else ""))
        print(f"  -> {m['name']}: {valid}/{attempts} valid .cube this run (cached: {out_path})")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate prompted-frontier raw-.cube LUTs (cached to JSONL).")
    ap.add_argument("--rows", default="data/eval/smoke_rows.jsonl")
    ap.add_argument("--out-dir", default="data/eval")
    ap.add_argument("--limit", type=int, default=10, help="first N rows (pilot default 10)")
    ap.add_argument("--models", default=None, help="comma-separated model names (default: all in config)")
    ap.add_argument("--config", default="configs/model_clients.yaml")
    args = ap.parse_args(argv)
    names = args.models.split(",") if args.models else None
    run(args.rows, args.out_dir, args.limit, names, args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
