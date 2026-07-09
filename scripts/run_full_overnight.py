#!/usr/bin/env python3
"""Overnight full data-gen run: load .env creds -> acquire -> pipeline (resumable).

Loads FreshLUTs (SLM_FRESHLUTS_*) and any Kaggle env from the repo .env into the child
environment (Kaggle also reads ~/.kaggle/kaggle.json). Runs acquisition first, then the
pipeline over the acquired registry (--no-acquire) so a pipeline retry never re-downloads.
"""

from __future__ import annotations

import datetime as _dt
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CFG = str(ROOT / "data_pipeline" / "configs" / "overnight_full.yaml")


def _load_env() -> dict:
    env = dict(os.environ)
    envfile = ROOT / ".env"
    if envfile.exists():
        for line in envfile.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    # kaggle 1.6.x reads KAGGLE_USERNAME/KAGGLE_KEY; map from the .env's KAGGLE_API_KEY
    if "KAGGLE_API_KEY" in env and "KAGGLE_KEY" not in env:
        env["KAGGLE_KEY"] = env["KAGGLE_API_KEY"]
    env["SLM_ARTIFACT_ROOT"] = str(ROOT)
    return env


def _stamp(msg: str) -> None:
    print(f"[{_dt.datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


def _run(env: dict, module: str, *args: str) -> int:
    _stamp(f"START {module} {' '.join(args)}")
    r = subprocess.run([sys.executable, "-m", module, *args], cwd=str(ROOT), env=env)
    _stamp(f"EXIT {r.returncode} {module}")
    return r.returncode


def main() -> int:
    env = _load_env()
    have_fresh = bool(env.get("SLM_FRESHLUTS_EMAIL") and env.get("SLM_FRESHLUTS_PASSWORD"))
    have_kaggle = bool(env.get("KAGGLE_KEY") and env.get("KAGGLE_USERNAME")) or (
        pathlib.Path.home() / ".kaggle" / "kaggle.json").exists()
    _stamp(f"creds: freshluts={have_fresh} kaggle={have_kaggle}")

    rc_a = _run(env, "data_pipeline.acquire.run_acquire", "--config", CFG, "--out", str(ROOT))
    rc_p = _run(env, "data_pipeline.run_pipeline", "--config", CFG, "--out", str(ROOT), "--no-acquire")
    _stamp(f"ALL DONE acquire_rc={rc_a} pipeline_rc={rc_p}")
    return rc_p


if __name__ == "__main__":
    raise SystemExit(main())
