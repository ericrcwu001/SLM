#!/usr/bin/env python3
"""Top up PPR10K from 3,000 -> all 4,055 pairs in JarvisArt/MMArt-PPR10k (parallel track).

Downloads only the pairs not already on disk (existing dirs are reused, no re-download), and
writes ALL resulting provenance rows to a SEPARATE file (ppr10k_topup_rows.jsonl) so it never
races the pipeline's writes to provenance.jsonl. Sets a sentinel on completion. Resumable.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


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
    from data_pipeline.acquire import downloaders as dl
    from data_pipeline.acquire.base import AcquireLimits
    from data_pipeline.acquire.ppr10k_hf import PPR10KHFConnector
    from data_pipeline.paths import artifact_paths

    paths = artifact_paths(str(ROOT)).ensure()
    rows_path = paths.raw_registry / "ppr10k_topup_rows.jsonl"
    sentinel = paths.raw_registry / ".ppr10k_topup_done"
    sentinel.unlink(missing_ok=True)

    def dl_fn(fn, root):
        p = pathlib.Path(root) / fn
        if p.exists() and p.stat().st_size > 0:
            return p  # already on disk -> reuse, no network
        out = dl.hf_download_file("JarvisArt/MMArt-PPR10k", fn, root, "dataset")
        print(f"[ppr10k-topup] fetched {fn}", flush=True)
        return out

    conn = PPR10KHFConnector(download_fn=dl_fn)
    print("[ppr10k-topup] acquiring ALL 4055 pairs (reuse on-disk, fetch the rest)", flush=True)
    report = conn.acquire(paths.luts_raw, AcquireLimits(max_items=None))
    print(f"[ppr10k-topup] status={report.status} acquired={report.acquired} "
          f"skipped={report.skipped} failed={report.failed}", flush=True)

    with open(rows_path, "w", encoding="utf-8") as fh:
        for art in report.artifacts:
            fh.write(json.dumps(art.to_registry_row().to_dict(), sort_keys=True) + "\n")
    sentinel.write_text(json.dumps({"status": report.status, "rows": len(report.artifacts)}))
    print(f"[ppr10k-topup] DONE wrote {len(report.artifacts)} rows -> {rows_path.name}; sentinel set",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
