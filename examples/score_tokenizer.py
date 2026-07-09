#!/usr/bin/env python3
"""Fast PROXY objective for the bilevel tokenizer-hyperparameter search.

Reads a candidate ``{config}`` JSON (training-knob overrides), trains a SHORT MLX run
(``max_steps=PROXY_STEPS``, fixed seed/batch) on the FROZEN-split train records, then
scores the tokenizer-dev holdout with the AUTHORITATIVE float64 torch gate
(``tokenizer.freeze.run_gate`` / ``tokenizer.metrics``) and prints ``{"metric": <mean ΔE00>}``
on stdout (the quantity the Stage-1 gate minimises).

This is a *stand-in* for the full 20k-step gate: it locates a good hyperparameter REGION,
not the final tokenizer. It never touches geometry/token-grammar knobs (those define the
frozen tokenizer identity) — only the training knobs in ``ALLOWED_KNOBS`` are honoured and
any LOCKED key in the incoming config is a hard error.

GPU serialisation: one Metal GPU cannot train two candidates at once (batch 128 memory
cliff, and even two batch-16 trainings thrash). We take an exclusive ``filelock`` on
``GPU_LOCK`` around the GPU training + gate so concurrent candidate evals SERIALISE.

Usage (the bilevel OBJECTIVE command):
    cd /Users/ericwu/Developer/SLM && PYTHONUNBUFFERED=1 \
        python examples/score_tokenizer.py --config <candidate.json>

Stdout is a single JSON line ``{"metric": ...}`` (plus diagnostics keys); ALL training /
progress logging goes to stderr so the metric line stays clean for the caller.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import replace

# --- proxy constants (define the PROXY; NOT search knobs) ---------------------------
PROXY_STEPS = 1200          # short stand-in for the 20k full run
PROXY_SEED = 0              # fixed so GPU nondeterminism is the only run-to-run noise
PROXY_BATCH = 16            # Metal sweet spot (128 => memory cliff)
PROXY_EVAL_EVERY = 300      # evals at 300/600/900 -> best.pt written on improvement
DEV_FRAC = 0.10             # tokenizer-dev holdout fraction (matches freeze default)
GPU_LOCK = os.environ.get("SLM_GPU_LOCK", "/tmp/slm_gpu.lock")

# training-only knobs the search may set (mirrors the bilevel PARAM_SPACE)
ALLOWED_KNOBS = {
    "w_recon", "w_deltaE", "w_tail", "w_smooth", "w_clip", "w_neutral",
    "commit_beta", "lr", "tail_frac", "augment", "scale_jitter",
}
# geometry / grammar keys that would change the frozen tokenizer identity — never settable
LOCKED_KNOBS = {
    "grid", "latent_grid", "token_count", "codebook_size", "code_dim",
    "enc_channels", "dec_channels", "norm_groups",
}
# proxy-defining keys the candidate may not override (they parameterise the stand-in)
FIXED_KNOBS = {"max_steps", "batch_size", "seed", "eval_every", "ckpt_every"}


def _log(*a):
    print(*a, file=sys.stderr, flush=True)


def _load_overrides(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise SystemExit(f"[score] config must be a JSON object, got {type(raw).__name__}")
    # a bilevel candidate is sometimes wrapped as {"config": {...}} or {"params": {...}}
    for wrap in ("config", "params", "knobs"):
        if wrap in raw and isinstance(raw[wrap], dict):
            raw = raw[wrap]
            break

    bad_locked = sorted(k for k in raw if k in LOCKED_KNOBS)
    if bad_locked:
        raise SystemExit(f"[score] refusing to set LOCKED identity knobs: {bad_locked}")

    over: dict = {}
    ignored: list[str] = []
    for k, v in raw.items():
        if k in FIXED_KNOBS:
            ignored.append(k)          # proxy-defined; not a search knob
            continue
        if k not in ALLOWED_KNOBS:
            ignored.append(k)
            continue
        if k == "augment":
            over[k] = bool(v) if not isinstance(v, str) else v.lower() in ("1", "true", "yes")
        else:
            over[k] = float(v)
    if ignored:
        _log(f"[score] ignoring non-knob keys: {sorted(ignored)}")
    return over


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Proxy objective: short MLX train + float64 gate -> mean ΔE00.")
    ap.add_argument("--config", required=True, help="candidate knob-override JSON (path)")
    ap.add_argument("--root", default=".")
    ap.add_argument("--steps", type=int, default=PROXY_STEPS, help="override proxy step count (sanity/tuning)")
    ap.add_argument("--out", default=None, help="checkpoint dir (default: a temp dir, removed on exit)")
    args = ap.parse_args(argv)

    over = _load_overrides(args.config)
    _log(f"[score] knob overrides: {json.dumps(over, sort_keys=True)}")

    # heavy imports after arg parsing so --help stays instant
    import mlx.core as mx
    from filelock import FileLock

    from tokenizer import data as D
    from tokenizer import freeze as F
    from tokenizer.config import DEFAULT_CONFIG

    if not mx.metal.is_available():
        _log("[score][warn] Metal GPU not available; MLX will run on CPU (slow).")

    # frozen-split train records + deterministic tokenizer-dev holdout
    records, cov = D.build_records_from_registry(root=args.root)
    if not records:
        raise SystemExit("[score] no train records resolved from the frozen split")
    train_recs, dev_recs = D.dev_holdout(records, frac=DEV_FRAC)
    _log(f"[score] records={len(records)} train={len(train_recs)} dev={len(dev_recs)} coverage={cov}")
    if not dev_recs:
        raise SystemExit("[score] empty dev holdout")

    cfg = replace(
        DEFAULT_CONFIG,
        max_steps=int(args.steps),
        batch_size=PROXY_BATCH,
        seed=PROXY_SEED,
        eval_every=PROXY_EVAL_EVERY,
        ckpt_every=int(args.steps),   # only the final extra ckpt; best.pt is what we score
        **over,
    )

    out_dir = args.out
    tmp = None
    if out_dir is None:
        tmp = tempfile.mkdtemp(prefix="tok_proxy_")
        out_dir = tmp

    lock = FileLock(GPU_LOCK)
    _log(f"[score] acquiring GPU lock {GPU_LOCK} ...")
    with lock:  # serialise GPU work across concurrent candidate evals
        _log(f"[score] lock held; training {cfg.max_steps} steps -> {out_dir}")
        from tokenizer.mlx import train_mlx as T

        T.train(cfg, train_recs, out_dir=out_dir, dev_records=dev_recs, seed=PROXY_SEED, log_fn=_log)

        # score the checkpoint the real Stage-4 gate would freeze (best.pt), authoritative
        # float64 torch gate; fall back to the always-written final ckpt.
        best = os.path.join(out_dir, "best.pt")
        final = os.path.join(out_dir, f"ckpt_{cfg.max_steps}.pt")
        ckpt = best if os.path.exists(best) else final
        if not os.path.exists(ckpt):
            raise SystemExit(f"[score] no checkpoint produced in {out_dir}")
        _log(f"[score] gating {os.path.basename(ckpt)} (float64 torch) on {len(dev_recs)} dev LUTs ...")
        model, _ck, gcfg = F.load_model_from_checkpoint(ckpt, device="cpu")
        report = F.run_gate(model, gcfg, dev_recs)

    o = report["overall"]
    cb = report["codebook"]
    metric = float(o["mean_deltae"])
    result = {
        "metric": metric,                       # <-- bilevel objective (min mean ΔE00)
        "mean_deltae": metric,
        "p95_deltae": float(o["p95_deltae"]),
        "p99_deltae": float(o["p99_deltae"]),
        "max_deltae": float(o["max_deltae"]),
        "mean_psnr": float(o["mean_psnr"]),
        "p5_psnr": float(o["p5_psnr"]),
        "active_frac": float(cb["active_frac"]),
        "perplexity": float(cb["perplexity"]),
        "gate_pass": bool(report["gate"]["pass"] and report["roundtrip"]["pass"]),
        "ckpt": os.path.basename(ckpt),
        "steps": int(cfg.max_steps),
    }
    _log(f"[score] DONE meanΔE={metric:.4f} p95={result['p95_deltae']:.3f} "
         f"p99={result['p99_deltae']:.3f} PSNR={result['mean_psnr']:.2f} "
         f"active={result['active_frac']:.1%} ppl={result['perplexity']:.1f} "
         f"gate_pass={result['gate_pass']}")

    if tmp is not None:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)

    print(json.dumps(result))   # single clean stdout line
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
