#!/usr/bin/env python3
"""Robust PPR10K top-up (3,000 -> all 4,055) via snapshot_download.

Replaces the per-file loop (which hung on a stalled hf_hub_download with no timeout).
snapshot_download parallelizes, retries, resumes, and skips files already on disk. After the
download it builds provenance rows FROM DISK into a SEPARATE file (ppr10k_topup_rows.jsonl,
no provenance race) + sentinel. Detached-friendly (nohup / PPID 1).
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPO = "JarvisArt/MMArt-PPR10k"
LICENSE = "Apache-2.0 (JarvisArt/MMArt-PPR10k, built on PPR10K)"


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
    # bound per-request timeouts so a stalled connection retries instead of hanging forever
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

    from huggingface_hub import snapshot_download

    from data_pipeline.acquire import downloaders as dl
    from data_pipeline.acquire.base import RawArtifact, utcnow_iso
    from data_pipeline.paths import artifact_paths

    paths = artifact_paths(str(ROOT)).ensure()
    root = paths.luts_raw / "ppr10k"
    rows_path = paths.raw_registry / "ppr10k_topup_rows.jsonl"
    sentinel = paths.raw_registry / ".ppr10k_topup_done"
    sentinel.unlink(missing_ok=True)

    print("[ppr10k-topup2] snapshot_download before/processed/config.xmp (parallel, resumable)",
          flush=True)
    snapshot_download(
        repo_id=REPO, repo_type="dataset", local_dir=str(root),
        allow_patterns=["global/*/before.jpg", "global/*/processed.jpg", "global/*/config.xmp"],
        max_workers=8, token=os.environ.get("HF_TOKEN"),
    )
    print("[ppr10k-topup2] snapshot complete; building rows from disk", flush=True)

    ts = utcnow_iso()
    global_dir = root / "global"
    dirs = sorted([d for d in global_dir.iterdir() if d.is_dir()])
    n = 0
    with open(rows_path, "w", encoding="utf-8") as fh:
        for d in dirs:
            before, processed, xmp = d / "before.jpg", d / "processed.jpg", d / "config.xmp"
            if not before.exists() or not processed.exists():
                continue
            art = RawArtifact(
                kind="image_pair", source_pack_id="ppr10k_expert_abc", family="ppr10k_derived",
                declared_domain="srgb", license=LICENSE, source_url=f"hf://{REPO}",
                file_hash=dl.sha256_file(processed), download_timestamp=ts,
                source_pair_id=f"global_{d.name}", group_id=d.name,
                source_image_path=str(before), target_image_path=str(processed),
                xmp_path=(str(xmp) if xmp.exists() else None), derivation_method="pair_fit",
            )
            fh.write(json.dumps(art.to_registry_row().to_dict(), sort_keys=True) + "\n")
            n += 1
    sentinel.write_text(json.dumps({"status": "ok", "rows": n}))
    print(f"[ppr10k-topup2] DONE wrote {n} rows -> {rows_path.name}; dirs={len(dirs)}; sentinel set",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
