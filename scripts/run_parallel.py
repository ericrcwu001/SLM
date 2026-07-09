#!/usr/bin/env python3
"""Parallelized pipeline: process the non-FreshLUTs corpus now (heavy pair-fits) while
FreshLUTs downloads in a separate process, then run the final pipeline over everything.

Sequence:
  Run A: pipeline over the current registry (non-FreshLUTs) -> enriches provenance.jsonl +
         caches canonical residuals. This is the compute-heavy stage (3,800 pair-fits).
  wait: poll for the FreshLUTs downloader's sentinel.
  merge: append freshluts_rows.jsonl (raw) into provenance.jsonl.
  Run B: pipeline over the full registry -> resumability reuses the non-fresh work (skips
         re-derivation), processes the FreshLUTs LUTs, runs the global stages -> final.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # make the repo-root `data_pipeline` package importable
CFG = str(ROOT / "data_pipeline" / "configs" / "overnight_full.yaml")
MAX_WAIT_S = 4 * 3600


def _load_env():
    envfile = ROOT / ".env"
    if envfile.exists():
        for line in envfile.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")
    if "KAGGLE_API_KEY" in os.environ and "KAGGLE_KEY" not in os.environ:
        os.environ["KAGGLE_KEY"] = os.environ["KAGGLE_API_KEY"]


def _stamp(msg):
    print(f"[{_dt.datetime.now().isoformat(timespec='seconds')}] [parallel] {msg}", flush=True)


def main() -> int:
    _load_env()
    from data_pipeline.paths import artifact_paths
    from data_pipeline.run_pipeline import run_pipeline

    paths = artifact_paths(str(ROOT))
    prov = paths.raw_registry / "provenance.jsonl"
    fresh_rows = paths.raw_registry / "freshluts_rows.jsonl"
    sentinel = paths.raw_registry / ".freshluts_done"

    n0 = sum(1 for _ in open(prov)) if prov.exists() else 0
    _stamp(f"Run A start: pipeline over non-FreshLUTs corpus ({n0} raw rows)")
    a = run_pipeline(config_path=CFG, out_root=str(ROOT), acquire=False)
    _stamp(f"Run A done: derive/filter={a['stages'].get('4_5_derive_filter')}")

    _stamp("waiting for FreshLUTs downloader sentinel ...")
    waited = 0
    while not sentinel.exists() and waited < MAX_WAIT_S:
        time.sleep(30)
        waited += 30
    if not sentinel.exists():
        _stamp("FreshLUTs sentinel not seen within max wait; finalizing on non-FreshLUTs only")
        return 0

    added = 0
    if fresh_rows.exists():
        with open(prov, "a", encoding="utf-8") as out:
            for line in open(fresh_rows, encoding="utf-8"):
                if line.strip():
                    out.write(line if line.endswith("\n") else line + "\n")
                    added += 1
    _stamp(f"merged {added} FreshLUTs rows into provenance.jsonl")

    _stamp("Run B start: full pipeline (resumable — reuses non-fresh, adds FreshLUTs)")
    b = run_pipeline(config_path=CFG, out_root=str(ROOT), acquire=False)
    _stamp(f"Run B done: derive/filter={b['stages'].get('4_5_derive_filter')} "
           f"active={b['stages'].get('9_active_eval', {}).get('active_selected')}")
    _stamp("ALL DONE (parallel)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
