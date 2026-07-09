#!/usr/bin/env python3
"""One-shot backfill: populate ``residual_key`` in the EXISTING provenance registry.

The ``persist_residual_key`` patch only writes ``residual_key`` on a future pipeline run.
This backfills the current ``data/raw_registry/provenance.jsonl`` in place — no pipeline
re-run — by recomputing the residual filename key the pipeline uses
(``lut_id or source_pair_id or file_hash``) and recording it ONLY for rows whose residual
``.npy`` actually exists on disk. That mirrors the pipeline, which sets the key only for
canonicalized rows (rejected-at-derivation rows have no residual and stay ``None``).

Idempotent (rows that already have ``residual_key`` are left untouched) and writes a
``.bak`` before rewriting. Operates on raw JSON dicts so no existing field is altered.

USAGE
    python scripts/patches/backfill_residual_key.py --dry-run   # preview counts, write nothing
    python scripts/patches/backfill_residual_key.py             # apply (writes .bak first)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _key_of(row: dict) -> str | None:
    # exact pipeline precedence for the residual filename stem (run_pipeline.py)
    return row.get("lut_id") or row.get("source_pair_id") or row.get("file_hash")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--dry-run", action="store_true", help="preview counts; change nothing")
    ap.add_argument("--no-backup", action="store_true", help="skip writing the .bak copy")
    args = ap.parse_args(argv)

    root = Path(args.root)
    prov = root / "data" / "raw_registry" / "provenance.jsonl"
    residual_dir = root / "luts" / "canonical_residual"
    if not prov.exists():
        print(f"[abort] no provenance registry at {prov}")
        return 2

    rows = [json.loads(l) for l in prov.read_text(encoding="utf-8").splitlines() if l.strip()]
    stats = {"total": len(rows), "already": 0, "set": 0, "no_residual_file": 0, "no_key": 0}
    for row in rows:
        if row.get("residual_key"):
            stats["already"] += 1
            continue
        key = _key_of(row)
        if not key:
            stats["no_key"] += 1
            continue
        if (residual_dir / f"{key}.npy").exists():
            row["residual_key"] = key
            stats["set"] += 1
        else:
            stats["no_residual_file"] += 1

    print(f"[backfill] {stats}")
    if args.dry_run:
        print("[dry-run] no files written. Re-run without --dry-run to apply.")
        return 0
    if stats["set"] == 0:
        print("[done] nothing to backfill (already populated or no matching residuals).")
        return 0

    if not args.no_backup:
        bak = prov.with_suffix(prov.suffix + ".bak")
        shutil.copy2(prov, bak)
        print(f"[backup] {bak}")

    tmp = prov.with_suffix(prov.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, prov)  # atomic
    print(f"[done] rewrote {prov} with residual_key on {stats['set']} rows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
