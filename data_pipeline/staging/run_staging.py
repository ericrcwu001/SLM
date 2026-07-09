"""``slm_stage`` CLI — pack / stage / push corpus between a durable root and a local root.

Thin argparse front end (the repo's first subcommand-based CLI) over the library functions in
:mod:`data_pipeline.staging.core`; matches the acquire/datagen convention of
"parse args -> library fn -> print JSON -> return int". Exit codes follow ``freeze_split``:
0 = ok/skipped, 2 = precondition (missing durable root, or GCS credentials/opt-in needed),
1 = staging failure (verify/transfer/manifest).

    python -m data_pipeline.staging.run_staging pack  --root . --durable-root gs://bkt/prompt_to_lut
    python -m data_pipeline.staging.run_staging stage --durable-root gs://bkt/prompt_to_lut --local-root /content/slm
    python -m data_pipeline.staging.run_staging push  --durable-root gs://bkt/prompt_to_lut --local-root /content/slm
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ..errors import RequiresManualOptIn, StagingError
from .core import run_pack, run_push, run_stage

_DEFAULT_STAGING_PATH = Path("configs/staging_default.yaml")
_DEFAULT_LOCAL_ROOT = "/content/slm"


def _load_config(path: str | None) -> dict:
    p = Path(path) if path else _DEFAULT_STAGING_PATH
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _emit(payload: dict) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="slm_stage", description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("pack", "stage", "push"):
        sp = sub.add_parser(name, help=f"{name} corpus shards")
        sp.add_argument("--config", default=None, help="staging YAML (default configs/staging_default.yaml)")
        sp.add_argument("--durable-root", default=None,
                        help="durable root: local path, Drive mount, gs://bucket/prefix, or "
                             "hf://datasets/user/repo (overrides config durable_root)")
        if name == "pack":
            sp.add_argument("--root", default=None, help="artifact root to pack (default $SLM_ARTIFACT_ROOT or cwd)")
            sp.add_argument("--dry-run", action="store_true", help="print the shard plan; write nothing")
        if name in ("stage", "push"):
            sp.add_argument("--local-root", default=None, help=f"local root (default config or {_DEFAULT_LOCAL_ROOT})")
        if name == "push":
            sp.add_argument("--rate-limit", type=float, default=None, help="min seconds between uploads")

    args = ap.parse_args(argv)
    config = _load_config(args.config)
    durable = args.durable_root or config.get("durable_root")
    local_root = getattr(args, "local_root", None) or config.get("local_root") or _DEFAULT_LOCAL_ROOT

    try:
        if args.cmd == "pack":
            if not durable and not args.dry_run:
                _emit({"command": "pack", "status": "failed",
                       "error": "no durable_root: pass --durable-root or set it in the staging config"})
                return 2
            summary = run_pack(args.root, durable or "(dry-run)", config, dry_run=args.dry_run)
        elif args.cmd == "stage":
            if not durable:
                _emit({"command": "stage", "status": "failed", "error": "no durable_root: pass --durable-root"})
                return 2
            summary = run_stage(durable, local_root, config)
        else:  # push
            if not durable:
                _emit({"command": "push", "status": "failed", "error": "no durable_root: pass --durable-root"})
                return 2
            summary = run_push(local_root, durable, config, rate_limit_s=args.rate_limit)
    except RequiresManualOptIn as exc:
        _emit({"command": args.cmd, "status": "blocked", "error": str(exc)})
        return 2
    except StagingError as exc:
        _emit({"command": args.cmd, "status": "failed", "error": str(exc)})
        return 1

    _emit(summary)
    return 0 if summary.get("status") in ("ok", "skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
