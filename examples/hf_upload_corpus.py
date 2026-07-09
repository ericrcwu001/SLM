#!/usr/bin/env python3
"""Create a PRIVATE HuggingFace dataset repo and pack+upload the corpus into it (slm_stage HfBackend).

Requires a write token first:  `hf auth login`  (or set HF_TOKEN). Then:

    python examples/hf_upload_corpus.py --repo slm-corpus            # full corpus (~10 GB, incl. luts/raw)
    python examples/hf_upload_corpus.py --repo slm-corpus --dry-run  # preview shard plan, no auth/writes

The corpus dirs come from configs/staging_default.yaml `pack.include` (luts/raw + canonical_residual
+ data manifests + tokenizer/final). The repo is created private; nothing is made public.
"""
from __future__ import annotations

import argparse
import json

from data_pipeline.staging.core import run_pack
from data_pipeline.staging.run_staging import _load_config


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="slm-corpus", help="repo name or full user/repo")
    ap.add_argument("--root", default=".", help="artifact root to pack (default cwd)")
    ap.add_argument("--config", default="configs/staging_default.yaml")
    ap.add_argument("--dry-run", action="store_true", help="preview shard plan; no auth, no writes")
    args = ap.parse_args(argv)
    cfg = _load_config(args.config)

    if args.dry_run:
        summary = run_pack(args.root, "hf://datasets/preview/preview", cfg, dry_run=True)
        print(json.dumps(summary, indent=2))
        return 0

    from huggingface_hub import create_repo, get_token, whoami
    if not get_token():
        print("[hf] no token — run `hf auth login` (write scope) first, then re-run.")
        return 2
    user = whoami()["name"]
    repo_id = args.repo if "/" in args.repo else f"{user}/{args.repo}"
    url = create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
    print(f"[hf] private dataset repo ready: {url}")

    durable = f"hf://datasets/{repo_id}"
    summary = run_pack(args.root, durable, cfg, dry_run=False)
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("status") in ("ok", "skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
