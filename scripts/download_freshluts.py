#!/usr/bin/env python3
"""Standalone FreshLUTs downloader (parallel track).

Downloads the full FreshLUTs catalog into luts/raw/fresh_luts/ and writes ITS provenance
rows to a SEPARATE file (data/raw_registry/freshluts_rows.jsonl) so it never races the
pipeline's writes to provenance.jsonl. Touches a sentinel on completion. Resumable
(already-downloaded ids are skipped).
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # make the repo-root `data_pipeline` package importable


def _load_env():
    envfile = ROOT / ".env"
    if envfile.exists():
        for line in envfile.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


def main() -> int:
    _load_env()
    from data_pipeline.acquire.base import AcquireLimits
    from data_pipeline.acquire.freshluts import FreshLutsConnector
    from data_pipeline.paths import artifact_paths

    paths = artifact_paths(str(ROOT)).ensure()
    rows_path = paths.raw_registry / "freshluts_rows.jsonl"
    sentinel = paths.raw_registry / ".freshluts_done"
    sentinel.unlink(missing_ok=True)

    conn = FreshLutsConnector()  # jittered pacing + full ID-range scan by default
    print("[freshluts] starting full-catalog download", flush=True)
    report = conn.acquire(paths.luts_raw, AcquireLimits(max_items=None))
    print(f"[freshluts] status={report.status} acquired={report.acquired} "
          f"skipped={report.skipped} failed={report.failed}", flush=True)

    with open(rows_path, "w", encoding="utf-8") as fh:
        for art in report.artifacts:
            fh.write(json.dumps(art.to_registry_row().to_dict(), sort_keys=True) + "\n")
    sentinel.write_text(json.dumps({"status": report.status, "rows": len(report.artifacts)}))
    print(f"[freshluts] wrote {len(report.artifacts)} rows -> {rows_path.name}; sentinel set", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
