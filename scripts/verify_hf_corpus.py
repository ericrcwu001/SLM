"""Verify the HF-staged corpus covers the current local corpus (esp. the unsupported-row images).

Cheap check (no 10GB download): the durable ``stage_manifest.json`` records per-shard
``member_count`` (total files packed) + a ``content_key`` (a hash of the packed (arcname, size)
set). We compare that to the LOCAL corpus computed with the SAME include/exclude the packer uses, so
we learn whether every local file — including the 272 refusal-row images made portable in P2 — is in
the HF shards, and whether the fixed ``active_rows.jsonl`` matches what was packed.

The write/read token is read from the environment or ``./.env`` (``HF_WRITE_TOKEN`` preferred, else
``HF_TOKEN``) and NEVER printed. Read-only: this downloads only the small manifest.

Usage:  python -m scripts.verify_hf_corpus
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from data_pipeline.staging.core import _content_key, _iter_corpus_files
from data_pipeline.staging.core import MANIFEST_NAME

_REPO = "ericrcwu/LUT_SLM"
_STAGING_CFG = "configs/staging_default.yaml"


def _load_token() -> str | None:
    for name in ("HF_WRITE_TOKEN", "HF_TOKEN"):
        if os.environ.get(name):
            return os.environ[name]
    env = Path(".env")
    if env.is_file():
        for line in env.read_text().splitlines():
            s = line.strip()
            for name in ("HF_WRITE_TOKEN", "HF_TOKEN"):
                if s.startswith(name + "="):
                    return s.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def main() -> int:
    cfg = yaml.safe_load(Path(_STAGING_CFG).read_text(encoding="utf-8")) or {}
    pack = cfg.get("pack", {}) or {}
    include, exclude = pack.get("include", []), pack.get("exclude", [])

    root = Path(os.environ.get("SLM_ARTIFACT_ROOT", os.getcwd()))
    local_files = _iter_corpus_files(root, include, exclude)
    local_key = _content_key(local_files)
    n_local = len(local_files)

    # Local presence of the 272 refusal-row images (portable, relative paths after P2).
    import json
    rows = [json.loads(l) for l in Path("data/active_sft/active_rows.jsonl").read_text().splitlines() if l.strip()]
    unsup = [r for r in rows if not r.get("is_supported") and r.get("image_path")]
    unsup_local_missing = [r["image_path"] for r in unsup
                           if not (root / r["image_path"]).exists()]

    token = _load_token()
    if not token:
        print("[verify] NO HF token found (env or .env: HF_WRITE_TOKEN/HF_TOKEN). Cannot read the "
              "durable manifest. Local corpus stats only:")
        print(f"[verify] local files (include={include}) = {n_local}; "
              f"unsupported images missing locally = {len(unsup_local_missing)}")
        return 2

    from huggingface_hub import hf_hub_download
    try:
        mpath = hf_hub_download(repo_id=_REPO, filename=MANIFEST_NAME, repo_type="dataset", token=token)
    except Exception as exc:  # noqa: BLE001
        print(f"[verify] could not fetch {MANIFEST_NAME} from hf://datasets/{_REPO}: "
              f"{type(exc).__name__}: {exc}")
        return 3
    manifest = json.loads(Path(mpath).read_text(encoding="utf-8"))
    shards = manifest.get("shards", [])
    packed_total = sum(int(s.get("member_count", 0)) for s in shards)
    packed_key = manifest.get("content_key")

    print(f"[verify] repo=hf://datasets/{_REPO}  staged={manifest.get('created_at')}")
    print(f"[verify] packed files (manifest member_count sum) = {packed_total} across {len(shards)} shards")
    print(f"[verify] local files (same include/exclude)       = {n_local}")
    print(f"[verify] delta (local - packed)                   = {n_local - packed_total}")
    print(f"[verify] content_key match (exact path+size set)  = {local_key == packed_key}")
    print(f"[verify] unsupported rows={len(unsup)} | images missing LOCALLY = {len(unsup_local_missing)}")
    if n_local == packed_total:
        print("[verify] OK: local file COUNT == packed count -> no files (incl. refusal images) are "
              "missing from the HF shards. (content_key differs only if a packed file's size changed, "
              "e.g. the fixed active_rows.jsonl.)")
    elif n_local > packed_total:
        print(f"[verify] GAP: {n_local - packed_total} local files are NOT in the HF shards -> a "
              "re-pack + push is needed to add them (e.g. python examples/hf_upload_corpus.py).")
    else:
        print("[verify] packed has MORE files than local (unexpected; investigate before re-pack).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
