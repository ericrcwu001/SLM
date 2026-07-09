#!/usr/bin/env python3
"""Self-healing PPR10K top-up: snapshot_download under a stall watchdog.

The HF connection intermittently wedges (a worker's read hangs past the request timeout, tqdm
freezes, 0% CPU). This runs snapshot_download as a child, watches on-disk dir progress, and on a
stall (no new complete dir for STALL_S) kills + restarts it (hf resumes .incomplete files from
disk). Loops until all 4,055 pairs are present or progress genuinely stops. Then builds provenance
rows from disk into ppr10k_topup_rows.jsonl + sentinel. Detached-friendly (nohup / PPID 1).
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REPO = "JarvisArt/MMArt-PPR10k"
LICENSE = "Apache-2.0 (JarvisArt/MMArt-PPR10k, built on PPR10K)"
TARGET = 4055
STALL_S = 75          # kill+restart if no new complete dir for this long
POLL_S = 15
MAX_ATTEMPTS = 80
MAX_NOPROGRESS = 5    # consecutive attempts with zero new dirs -> accept what we have


def _load_env():
    envfile = ROOT / ".env"
    if envfile.exists():
        for line in envfile.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


def _count_complete(global_dir: pathlib.Path) -> int:
    n = 0
    if not global_dir.exists():
        return 0
    for d in global_dir.iterdir():
        if d.is_dir() and (d / "before.jpg").exists() and (d / "processed.jpg").exists():
            n += 1
    return n


_CHILD = (
    "import os\n"
    "os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','30')\n"
    "os.environ.setdefault('HF_HUB_ETAG_TIMEOUT','20')\n"
    "from huggingface_hub import snapshot_download\n"
    "snapshot_download(repo_id=%r, repo_type='dataset', local_dir=%r,\n"
    "    allow_patterns=['global/*/before.jpg','global/*/processed.jpg','global/*/config.xmp'],\n"
    "    max_workers=8, token=os.environ.get('HF_TOKEN'))\n"
)


def _run_attempt(code: str, global_dir: pathlib.Path, env: dict):
    """Run one snapshot_download child; kill it if disk progress stalls. Returns a status str."""
    p = subprocess.Popen([sys.executable, "-c", code], env=env,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    last_count = _count_complete(global_dir)
    last_change = time.time()
    while True:
        rc = p.poll()
        if rc is not None:
            return f"exit={rc}"
        time.sleep(POLL_S)
        c = _count_complete(global_dir)
        if c > last_count:
            last_count = c
            last_change = time.time()
        if time.time() - last_change > STALL_S:
            p.terminate()
            try:
                p.wait(10)
            except Exception:  # noqa: BLE001
                p.kill()
            return "stalled"


def main() -> int:
    _load_env()
    from data_pipeline.acquire import downloaders as dl
    from data_pipeline.acquire.base import RawArtifact, utcnow_iso
    from data_pipeline.paths import artifact_paths

    paths = artifact_paths(str(ROOT)).ensure()
    root = paths.luts_raw / "ppr10k"
    global_dir = root / "global"
    rows_path = paths.raw_registry / "ppr10k_topup_rows.jsonl"
    sentinel = paths.raw_registry / ".ppr10k_topup_done"
    sentinel.unlink(missing_ok=True)

    code = _CHILD % (REPO, str(root))
    env = dict(os.environ)

    noprogress = 0
    for attempt in range(1, MAX_ATTEMPTS + 1):
        done = _count_complete(global_dir)
        if done >= TARGET:
            break
        res = _run_attempt(code, global_dir, env)
        now = _count_complete(global_dir)
        print(f"[watchdog] attempt {attempt}: {done}->{now} (+{now - done}) {res}", flush=True)
        if now >= TARGET:
            break
        noprogress = noprogress + 1 if now == done else 0
        if noprogress >= MAX_NOPROGRESS:
            print(f"[watchdog] no progress {MAX_NOPROGRESS}x; accepting {now} available", flush=True)
            break
        time.sleep(3)

    ts = utcnow_iso()
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
    print(f"[watchdog] DONE wrote {n} rows -> {rows_path.name}; complete_dirs={n}; sentinel set",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
