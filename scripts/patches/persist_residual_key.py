#!/usr/bin/env python3
"""STAGED patch — persist the residual-file key to the provenance registry.

WHY
    Each canonical residual is saved as ``luts/canonical_residual/<key>.npy`` where
    ``key = lut_id or source_pair_id or file_hash`` (data_pipeline/run_pipeline.py). That
    key is kept only in-memory as ``_residual_key`` and dropped on write, because
    ``_row_obj`` strips underscore-prefixed keys before building the ProvenanceRow. So any
    consumer (e.g. tokenizer.data) must REPLICATE the ``lut_id or source_pair_id or
    file_hash`` precedence to map a residual file back to its split — coupling it to a
    pipeline internal. Persisting an explicit, non-underscore ``residual_key`` field makes
    the filename↔row join a single authoritative key.

WHEN
    Apply AFTER the current data-gen run (run_parallel.py / run_pipeline) finishes — this
    script edits live pipeline source; applying mid-run risks a fresh subprocess picking up
    half-applied code. It refuses to run while those processes are active (override with
    --force). A subsequent pipeline pass (even a resumed one) rewrites the whole registry
    via ``store.write_all(...)``, so it backfills ``residual_key`` for cached rows too.

WHAT (idempotent; unified diff for review)
    data_pipeline/registry.py  (ProvenanceRow dataclass):
        + residual_key: Optional[str] = None  # canonical_residual/<residual_key>.npy stem

    data_pipeline/run_pipeline.py  (both places the key is set):
          row["_residual_key"] = _rk
        + row["residual_key"] = _rk   # persisted (survives _row_obj underscore strip)
        ...
          row["_residual_key"] = key
        + row["residual_key"] = key   # persisted (survives _row_obj underscore strip)

    After this, tokenizer.data resolves residuals via the explicit ``residual_key`` first
    (it already falls back to the old precedence, so it works before and after).

USAGE
    python scripts/patches/persist_residual_key.py --dry-run   # preview, change nothing
    python scripts/patches/persist_residual_key.py             # apply (post-run)
    python scripts/patches/persist_residual_key.py --force     # apply even if pipeline looks live
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# (relative path, anchor-that-must-exist-once, replacement). Replacements are idempotent:
# each adds a line that must NOT already be present.
EDITS = [
    (
        "data_pipeline/registry.py",
        "    input_embedding_id: Optional[str] = None\n",
        "    input_embedding_id: Optional[str] = None\n"
        "    residual_key: Optional[str] = None  # canonical_residual/<residual_key>.npy filename stem\n",
        'residual_key: Optional[str] = None',
    ),
    (
        "data_pipeline/run_pipeline.py",
        '            row["_residual_key"] = _rk\n'
        "            enriched.append(row)\n"
        "            continue\n",
        '            row["_residual_key"] = _rk\n'
        '            row["residual_key"] = _rk  # persisted (survives _row_obj underscore strip)\n'
        "            enriched.append(row)\n"
        "            continue\n",
        'row["residual_key"] = _rk',
    ),
    (
        "data_pipeline/run_pipeline.py",
        '        np.save(paths.canonical_residual / f"{key}.npy", can.residual)\n'
        '        row["_residual_key"] = key\n'
        "        attrition[rep.tier] += 1\n",
        '        np.save(paths.canonical_residual / f"{key}.npy", can.residual)\n'
        '        row["_residual_key"] = key\n'
        '        row["residual_key"] = key  # persisted (survives _row_obj underscore strip)\n'
        "        attrition[rep.tier] += 1\n",
        'row["residual_key"] = key',
    ),
]


def _pipeline_running() -> list[str]:
    try:
        out = subprocess.run(["ps", "ax"], capture_output=True, text=True, check=False).stdout
    except Exception:
        return []
    hits = [ln for ln in out.splitlines()
            if ("run_parallel.py" in ln or "slm_datagen" in ln or "download_freshluts.py" in ln
                or "data_pipeline.run_pipeline" in ln)
            and "persist_residual_key" not in ln
            and "grep" not in ln and "/ps" not in ln]
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="preview; change nothing")
    ap.add_argument("--force", action="store_true", help="apply even if the pipeline looks live")
    args = ap.parse_args(argv)

    live = _pipeline_running()
    if live and not (args.force or args.dry_run):
        print("[abort] data pipeline appears to be RUNNING — apply after it finishes, or pass --force:")
        for ln in live[:4]:
            print("   ", ln.strip()[:110])
        return 3

    changed, already, failed = 0, 0, 0
    for rel, old, new, sentinel in EDITS:
        path = REPO / rel
        text = path.read_text(encoding="utf-8")
        if sentinel in text:
            print(f"[skip] {rel}: already patched ({sentinel!r} present)")
            already += 1
            continue
        n = text.count(old)
        if n != 1:
            print(f"[FAIL] {rel}: anchor found {n}x (expected 1) — file drifted; patch by hand. Anchor:\n"
                  f"       {old.splitlines()[0]!r}")
            failed += 1
            continue
        if args.dry_run:
            print(f"[dry-run] {rel}: would insert residual_key line")
            changed += 1
            continue
        path.write_text(text.replace(old, new, 1), encoding="utf-8")
        print(f"[applied] {rel}: +residual_key")
        changed += 1

    if failed:
        print(f"\n[done] {changed} to-change, {already} already, {failed} FAILED — resolve failures by hand.")
        return 1
    if args.dry_run:
        print(f"\n[dry-run] {changed} edit(s) pending, {already} already applied. Re-run without --dry-run to apply.")
        return 0
    print(f"\n[done] {changed} applied, {already} already present.")
    print("[verify] run one pipeline pass to rewrite the registry, then:")
    print("         python -c \"import json; rows=[json.loads(l) for l in open('data/raw_registry/provenance.jsonl')];"
          " print('with residual_key:', sum(1 for r in rows if r.get('residual_key')), '/', len(rows))\"")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
