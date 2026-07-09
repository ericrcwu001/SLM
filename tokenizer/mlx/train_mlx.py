"""MLX (Apple-GPU) VQ LUT tokenizer training loop + CLI.

Trains the MLX mirror of the tokenizer, then writes a **torch** checkpoint each time (via
tokenizer.mlx.convert) so the existing torch freeze/gate/eval path consumes it unchanged.
Dev evaluation reuses the numpy gate (tokenizer.metrics) directly on the MLX model
(encode/decode contract is identical to torch).

Runs nothing on import; train only via ``python -m tokenizer.mlx.train_mlx ...``.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import replace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from .. import data as D
from .. import metrics as M
from ..config import DEFAULT_CONFIG, TokenizerConfig
from . import convert
from . import losses_mlx as L
from .data_mlx import MlxBatcher
from .model_mlx import VQVAEmlx


def _lr_at(step: int, cfg: TokenizerConfig) -> float:
    """Linear warmup then (v2) cosine decay to lr_min — mirrors tokenizer.train._lr_at."""
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if not cfg.lr_decay:
        return cfg.lr
    total = max(1, cfg.max_steps - cfg.warmup_steps)
    prog = min(1.0, max(0.0, (step - cfg.warmup_steps) / total))
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1.0 + math.cos(math.pi * prog))


def evaluate_dev(model: VQVAEmlx, dev_records, cfg: TokenizerConfig) -> dict | None:
    """Reconstruction gate on the tokenizer-dev holdout (numpy, via the MLX encode/decode)."""
    if not dev_records:
        return None
    targets = D.load_residual_arrays(dev_records)
    recons, codes = M.reconstruct(model, targets)          # model.encode/decode (MLX)
    fams = [r.source_family for r in dev_records]
    agg = M.aggregate_reconstruction(targets, recons, fams)
    cb = M.codebook_stats(codes, cfg.codebook_size)
    gate = M.evaluate_gate(agg, cb)
    return {"overall": agg["overall"], "per_family": agg["per_family"], "codebook": cb, "gate": gate}


def train(cfg: TokenizerConfig, train_records, out_dir: str, dev_records=None,
          seed: int = 0, log_fn=print) -> str:
    mx.random.seed(seed)
    np.random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    model = VQVAEmlx(cfg)
    mx.eval(model.parameters())
    opt = optim.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    identity = L.identity_chlast(cfg.grid)

    def loss_fn(model, x):
        return L.total_loss(model(x), x, cfg, identity)

    lag = nn.value_and_grad(model, loss_fn)
    batcher = MlxBatcher(train_records, cfg.batch_size, seed=seed,
                         augment=cfg.augment, scale_jitter=cfg.scale_jitter)

    best = None
    t0 = time.time()
    for step in range(cfg.max_steps):
        xb = batcher.batch()
        opt.learning_rate = _lr_at(step, cfg)
        loss, grads = lag(model, xb)
        grads, _gnorm = optim.clip_grad_norm(grads, cfg.grad_clip)
        opt.update(model, grads)
        model.vq.ema_update(model.encoder(xb))             # EMA outside the grad transform
        mx.eval(model.parameters(), opt.state, model.vq._codebook,
                model.vq._cluster_size, model.vq._embed_avg, loss)

        if step % 50 == 0:
            comp = L.components(model(xb), xb, cfg, identity)
            log_fn(f"step {step}/{cfg.max_steps} "
                   + " ".join(f"{k}={comp[k]:.4f}" for k in
                              ("loss", "recon", "deltaE", "tail", "smooth", "clip", "neutral", "commit", "perplexity"))
                   + f" lr={_lr_at(step, cfg):.2e}")

        if cfg.eval_every and step > 0 and step % cfg.eval_every == 0:
            rep = evaluate_dev(model, dev_records, cfg)
            if rep:
                o = rep["overall"]
                log_fn(f"[dev@{step}] meanΔE={o['mean_deltae']:.3f} p95={o['p95_deltae']:.3f} "
                       f"PSNR={o['mean_psnr']:.2f} pass={rep['gate']['pass']} alerts={rep['gate']['alerts']}")
                if best is None or o["mean_deltae"] < best:
                    best = o["mean_deltae"]
                    convert.save_torch_checkpoint(model, cfg, train_records, os.path.join(out_dir, "best.pt"))

        if cfg.ckpt_every and step > 0 and step % cfg.ckpt_every == 0:
            convert.save_torch_checkpoint(model, cfg, train_records, os.path.join(out_dir, f"ckpt_{step}.pt"))

    final = os.path.join(out_dir, f"ckpt_{cfg.max_steps}.pt")
    convert.save_torch_checkpoint(model, cfg, train_records, final)
    try:
        model.save_weights(os.path.join(out_dir, "mlx_final.safetensors"))
    except Exception:  # noqa: BLE001
        pass
    log_fn(f"[done] {cfg.max_steps} steps in {time.time()-t0:.1f}s -> {final} (torch checkpoint)")
    return final


# --- CLI ----------------------------------------------------------------------------
def _load_records(args):
    if args.manifest and os.path.exists(args.manifest):
        recs = D.load_train_manifest(args.manifest)
        print(f"[data] {len(recs)} train records from manifest {args.manifest}")
        return recs
    recs, cov = D.build_records_from_registry(root=args.root)
    print(f"[data] {len(recs)} train records from registry; coverage={cov}")
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the VQ LUT tokenizer on the Apple GPU (MLX). Runs only when invoked.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--out", default="tokenizer/checkpoints_mlx")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--dev-frac", type=float, default=0.10)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--augment", action="store_true", help="enable neutral-preserving scale-jitter augmentation")
    ap.add_argument("--scale-jitter", type=float, default=None, help="augmentation magnitude (e.g. 0.05)")
    ap.add_argument("--smoke", type=int, default=0, help="use only N records + few steps (dev sanity)")
    args = ap.parse_args(argv)

    if not mx.metal.is_available():
        print("[warn] Metal GPU not available; MLX will run on CPU.")

    cfg = DEFAULT_CONFIG
    over = {}
    for k, v in (("max_steps", args.max_steps), ("batch_size", args.batch_size), ("lr", args.lr),
                 ("eval_every", args.eval_every), ("ckpt_every", args.ckpt_every), ("seed", args.seed),
                 ("scale_jitter", args.scale_jitter)):
        if v is not None:
            over[k] = v
    if args.augment:
        over["augment"] = True
        if args.scale_jitter is None:
            over["scale_jitter"] = 0.05
    if over:
        cfg = replace(cfg, **over)

    records = _load_records(args)
    if args.smoke:
        records = records[: args.smoke]
        cfg = replace(cfg, max_steps=min(cfg.max_steps, 40), ckpt_every=20, eval_every=20)
    if not records:
        print("[abort] no train records; provide --manifest or run the data pipeline first")
        return 2

    train_recs, dev_recs = D.dev_holdout(records, frac=args.dev_frac)
    print(f"[data] train={len(train_recs)} dev-holdout={len(dev_recs)} "
          f"families={sorted({r.source_family for r in records})} | device={mx.default_device()}")
    out = os.path.join(args.root, args.out) if not os.path.isabs(args.out) else args.out
    train(cfg, train_recs, out_dir=out, dev_records=dev_recs, seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
