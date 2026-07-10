"""Tiny append-only ledger for the sft-improve greedy loop (no engine, no deps).

Keeps run state on disk so it survives context/compaction: a JSON list of
{iter, params, metric, adapter, note}. ``append`` adds a row and prints the current best;
``show`` prints the table + best. Pure stdlib.

    python ledger.py append --ledger L.json --iter 2 --params '{"lora_r":24}' --metric 0.71 \
        --adapter models/sft_adapters/bl_smoke200 --note "raised lora_r"
    python ledger.py show   --ledger L.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _load(path: Path) -> list:
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
        except (ValueError, OSError):
            return []
    return []


def _best(rows: list) -> dict | None:
    scored = [r for r in rows if isinstance(r.get("metric"), (int, float)) and not math.isnan(r["metric"])]
    return max(scored, key=lambda r: r["metric"]) if scored else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("append")
    a.add_argument("--ledger", required=True)
    a.add_argument("--iter", type=int, required=True)
    a.add_argument("--params", required=True, help="candidate params as a JSON object string")
    a.add_argument("--metric", default="nan", help="float, or 'nan' for a failed run")
    a.add_argument("--adapter", default="")
    a.add_argument("--note", default="")

    s = sub.add_parser("show")
    s.add_argument("--ledger", required=True)

    args = ap.parse_args(argv)
    path = Path(args.ledger)
    rows = _load(path)

    if args.cmd == "append":
        try:
            metric = float(args.metric)
        except ValueError:
            metric = float("nan")
        rows.append({"iter": args.iter, "params": json.loads(args.params),
                     "metric": None if math.isnan(metric) else metric,
                     "adapter": args.adapter, "note": args.note})
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    best = _best(rows)
    print(json.dumps({
        "rows": len(rows),
        "best": best,
        "table": [{"iter": r["iter"], "metric": r["metric"],
                   "params": r["params"], "note": r.get("note", "")} for r in rows],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
