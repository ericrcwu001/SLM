"""Bilevel objective bridge: one bilevel candidate -> one guarded (train + score) eval on Colab.

This is the self-contained unit the bilevel loop's metric-injection topology runs ON the Colab VM:
Codex writes a candidate config, Computer Use runs the notebook cell that calls this bridge, the
resulting ``METRIC=<accuracy>`` lands in the (local) ``.ipynb`` output, Codex reads it and feeds it
back to the engine via ``run_iteration.py --pre-shaped --metric``.

Contract (mode ``colab``):
  input  : ``--config`` a JSON file — a FLAT dict of param overrides, or ``{"params": {...}}``
           (the bilevel param_space knobs: lora_r, lora_alpha, learning_rate_lora, max_pixels, ...).
  guards : (1) pre-validate the merged config against ``SFTConfig.__post_init__`` (reject
           epochs/batch-triple/dtype violations with a clear message, never a raw traceback);
           (2) run training in its OWN process group under a GPU flock, with a timeout that kills the
           whole CUDA tree; (3) parse the trainer's ``sft_summary`` and FAIL if ``rows_trained==0``
           (the silent-success trap); (4) score decoder-free held-out token accuracy.
  output : a ``{"bridge_summary": {...}}`` JSON line, then exactly ONE final ``METRIC=<accuracy>``
           line (direction=MAX). Exit non-zero (no METRIC) on any hard failure, so the engine records
           a clean discard instead of a fabricated improvement.

Pure orchestration (no torch import here) — it shells out to :mod:`sft.train` and
:mod:`sft.score_tokens`, so it is import-safe without the ``sft`` extra.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

from sft.config import SFTConfig

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SFT_SUMMARY_RE = re.compile(r'\{"sft_summary":.*\}$')
_METRIC_RE = re.compile(r"(?i)\bMETRIC\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")


def _candidate_params(config_path: str) -> dict:
    raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
    params = raw.get("params", raw) if isinstance(raw, dict) else {}
    if not isinstance(params, dict):
        raise ValueError(f"candidate config must be a dict or {{'params': {{...}}}}, got {type(raw)}")
    return params


def _merged_config(base_config: str, params: dict) -> dict:
    fields = {f.name for f in __import__("dataclasses").fields(SFTConfig)}
    base = {}
    if base_config and Path(base_config).is_file():
        base = yaml.safe_load(Path(base_config).read_text(encoding="utf-8")) or {}
    merged = {k: v for k, v in base.items() if k in fields}
    unknown = [k for k in params if k not in fields]
    if unknown:
        raise ValueError(f"candidate has non-SFTConfig knobs {unknown}; keep them out of param_space")
    merged.update({k: v for k, v in params.items() if k in fields})
    return merged


def _validate(merged: dict) -> None:
    """Trip SFTConfig.__post_init__ now, with a clear message (never a raw traceback mid-train)."""
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in merged.items()}
    try:
        SFTConfig(**kw)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"invalid candidate config (SFTConfig rejected it): {exc}") from exc


def _run(cmd: list[str], *, cwd: Path, env: dict, timeout: int) -> subprocess.CompletedProcess:
    """Run in its own process group so a timeout kills the whole CUDA tree (no orphaned GPU procs)."""
    proc = subprocess.Popen(cmd, cwd=str(cwd), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(cmd, proc.returncode, out, "")
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        proc.communicate()
        raise


def run_colab(config_path: str, *, repo_root: str, artifact_root: str, resized_model: str,
              base_config: str, out_dir: str, run_id: str, smoke_size: int, max_steps: int | None,
              score_limit: int, timeout: int) -> dict:
    params = _candidate_params(config_path)
    merged = _merged_config(base_config, params)
    _validate(merged)

    repo = Path(repo_root)
    env = dict(os.environ)
    env["SLM_ARTIFACT_ROOT"] = artifact_root  # images + frozen tokenizer resolve to the staged corpus

    # Write the merged SFTConfig YAML both train and score consume (identical processor levers).
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir="/tmp") as fh:
        yaml.safe_dump(merged, fh)
        merged_cfg = fh.name

    smoke_tag = str(smoke_size) if smoke_size else "full"
    adapter_dir = str(Path(out_dir) / f"{run_id}_smoke{smoke_tag}")

    # --- serialize the single A100 across concurrent bridge invocations (best-effort flock) ---
    lock_fh = open("/tmp/slm_gpu.lock", "w")
    try:
        import fcntl
        fcntl.flock(lock_fh, fcntl.LOCK_EX)
    except Exception:  # noqa: BLE001 — flock unavailable; metric-injection already serializes
        pass

    try:
        train_cmd = [sys.executable, "-m", "sft.train", "--config", merged_cfg,
                     "--resized-model", resized_model, "--smoke-size", str(smoke_size),
                     "--out", out_dir, "--run-id", run_id]
        if max_steps:
            train_cmd += ["--max-steps", str(max_steps)]
        train = _run(train_cmd, cwd=repo, env=env, timeout=timeout)
        sys.stdout.write(train.stdout)

        summary = {}
        for line in train.stdout.splitlines():
            if _SFT_SUMMARY_RE.search(line.strip()):
                summary = json.loads(line.strip()).get("sft_summary", {})
        if train.returncode != 0:
            raise RuntimeError(f"training exited {train.returncode} (see log above)")
        if not summary or not summary.get("rows_trained"):
            raise RuntimeError(f"training did nothing (rows_trained={summary.get('rows_trained')}) — "
                               f"silent-success trap; check SLM_ARTIFACT_ROOT={artifact_root}")

        score_cmd = [sys.executable, "-m", "sft.score_tokens", "--config", merged_cfg,
                     "--resized-model", resized_model, "--adapter", adapter_dir,
                     "--limit", str(score_limit)]
        score = _run(score_cmd, cwd=repo, env=env, timeout=timeout)
        sys.stdout.write(score.stdout)
        if score.returncode != 0:
            raise RuntimeError(f"scoring exited {score.returncode} (see log above)")
        metrics = _METRIC_RE.findall(score.stdout)
        if not metrics:
            raise RuntimeError("scorer emitted no METRIC= line")
        metric = float(metrics[-1])
    finally:
        try:
            lock_fh.close()
        except Exception:  # noqa: BLE001
            pass

    return {"metric": metric, "adapter": adapter_dir, "train_summary": summary,
            "smoke_size": smoke_size, "params": params}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=["colab"], default="colab",
                    help="colab = engine+objective co-located on the VM (the chosen topology)")
    ap.add_argument("--config", required=True, help="candidate JSON: flat overrides or {'params':{}}")
    ap.add_argument("--repo-root", default=str(_REPO_ROOT))
    ap.add_argument("--artifact-root", default=os.environ.get("SLM_ARTIFACT_ROOT", "/content/slm"))
    ap.add_argument("--resized-model", default="models/base_resized")
    ap.add_argument("--base-config", default="configs/sft_default.yaml")
    ap.add_argument("--out", default="models/sft_adapters")
    ap.add_argument("--run-id", default="bl")
    ap.add_argument("--smoke-size", type=int, default=200, help="0 = full dataset")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--score-limit", type=int, default=48)
    ap.add_argument("--timeout", type=int, default=3600, help="per train/score subprocess, seconds")
    args = ap.parse_args(argv)

    try:
        rep = run_colab(args.config, repo_root=args.repo_root, artifact_root=args.artifact_root,
                        resized_model=args.resized_model, base_config=args.base_config,
                        out_dir=args.out, run_id=args.run_id, smoke_size=args.smoke_size,
                        max_steps=args.max_steps, score_limit=args.score_limit, timeout=args.timeout)
    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"bridge_summary": {"error": str(exc)}}))
        print(f"[bridge][ABORT] {exc}")
        return 1
    print(json.dumps({"bridge_summary": rep}))
    print(f"METRIC={rep['metric']:.6f}")   # the single sentinel the engine/Codex reads (direction=max)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
