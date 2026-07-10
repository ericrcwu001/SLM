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
  upload : if ``--push-hf-repo`` (or env ``PUSH_HF_REPO``) is set, the trained adapter is uploaded to
           that HF MODEL repo under a per-config subfolder using ``HF_WRITE_TOKEN`` (read-only
           ``HF_TOKEN`` 403s). Best-effort: a push failure warns but keeps the (valid) metric.

Pure orchestration (no torch import here) — it shells out to :mod:`sft.train` and
:mod:`sft.score_tokens`, so it is import-safe without the ``sft`` extra.
"""

from __future__ import annotations

import argparse
import hashlib
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


def _push_to_hf(adapter_dir: str, repo_id: str, path_in_repo: str, token: str | None) -> dict:
    """Upload a trained adapter dir to an HF MODEL repo (created if missing). Best-effort: a push
    failure is reported but does NOT discard the (valid) metric."""
    if not repo_id:
        return {"pushed": False, "skipped": "no PUSH_HF_REPO / --push-hf-repo set"}
    if not token:
        return {"pushed": False, "error": "no HF WRITE token (set HF_WRITE_TOKEN; read-only HF_TOKEN 403s)"}
    try:
        from huggingface_hub import HfApi, create_repo
    except Exception as exc:  # noqa: BLE001
        return {"pushed": False, "error": f"huggingface_hub unavailable: {exc}"}
    try:
        create_repo(repo_id, repo_type="model", private=True, exist_ok=True, token=token)
        HfApi().upload_folder(folder_path=adapter_dir, repo_id=repo_id, repo_type="model",
                              path_in_repo=path_in_repo, token=token,
                              commit_message=f"sft adapter {path_in_repo}")
        return {"pushed": True, "repo": repo_id, "path": path_in_repo}
    except Exception as exc:  # noqa: BLE001
        return {"pushed": False, "error": str(exc)}


def run_colab(config_path: str, *, repo_root: str, artifact_root: str, resized_model: str,
              base_config: str, out_dir: str, run_id: str, smoke_size: int, max_steps: int | None,
              score_limit: int, timeout: int, push_hf_repo: str = "", hf_token: str | None = None) -> dict:
    params = _candidate_params(config_path)
    merged = _merged_config(base_config, params)
    _validate(merged)
    # Distinct adapter + HF subfolder per distinct config (so runs never overwrite each other).
    eff_run_id = f"{run_id}_{hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:8]}"

    repo = Path(repo_root)
    env = dict(os.environ)
    env["SLM_ARTIFACT_ROOT"] = artifact_root  # images + frozen tokenizer resolve to the staged corpus

    # Write the merged SFTConfig YAML both train and score consume (identical processor levers).
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, dir="/tmp") as fh:
        yaml.safe_dump(merged, fh)
        merged_cfg = fh.name

    smoke_tag = str(smoke_size) if smoke_size else "full"
    adapter_dir = str(Path(out_dir) / f"{eff_run_id}_smoke{smoke_tag}")

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
                     "--out", out_dir, "--run-id", eff_run_id]
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
        push = _push_to_hf(adapter_dir, push_hf_repo, f"{eff_run_id}_smoke{smoke_tag}", hf_token)
        if push.get("error"):
            sys.stdout.write(f"[bridge][warn] HF upload failed: {push['error']}\n")
    finally:
        try:
            lock_fh.close()
        except Exception:  # noqa: BLE001
            pass

    return {"metric": metric, "run_id": eff_run_id, "adapter": adapter_dir, "hf_push": push,
            "train_summary": summary, "smoke_size": smoke_size, "params": params}


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
    ap.add_argument("--push-hf-repo", default=os.environ.get("PUSH_HF_REPO", ""),
                    help="HF model repo to upload the trained adapter to (default env PUSH_HF_REPO; "
                         "empty = no upload). Needs a WRITE token in HF_WRITE_TOKEN.")
    args = ap.parse_args(argv)
    hf_token = os.environ.get("HF_WRITE_TOKEN") or os.environ.get("HF_TOKEN")

    try:
        rep = run_colab(args.config, repo_root=args.repo_root, artifact_root=args.artifact_root,
                        resized_model=args.resized_model, base_config=args.base_config,
                        out_dir=args.out, run_id=args.run_id, smoke_size=args.smoke_size,
                        max_steps=args.max_steps, score_limit=args.score_limit, timeout=args.timeout,
                        push_hf_repo=args.push_hf_repo, hf_token=hf_token)
    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"bridge_summary": {"error": str(exc)}}))
        print(f"[bridge][ABORT] {exc}")
        return 1
    print(json.dumps({"bridge_summary": rep}))
    print(f"METRIC={rep['metric']:.6f}")   # the single sentinel the engine/Codex reads (direction=max)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
