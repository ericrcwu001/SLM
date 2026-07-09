"""VQ LUT tokenizer training loop + CLI (master-plan Stage 7; training_plan_colab.md Stage 1).

Resumable AdamW training with EMA VQ. Produces candidate checkpoints under
``tokenizer/checkpoints/``; the reconstruction/tail/per-family gate and the freeze to
``tokenizer/final/`` are in :mod:`tokenizer.freeze`.

IMPORTANT: importing this module runs nothing. Training starts ONLY via an explicit
``python -m tokenizer.train ...`` invocation (the ``__main__`` guard at the bottom).
There is no import-time or auto-run behavior.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import time
from dataclasses import replace

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import data as data_mod
from . import metrics as metrics_mod
from .config import DEFAULT_CONFIG, TokenizerConfig
from .losses import total_loss
from .manifest import hash_state_dict
from .model import VQVAE


# --- RNG (de)serialization ----------------------------------------------------------
# numpy's RNG state contains a raw ndarray, which torch.load(weights_only=True) (the
# PyTorch >=2.6 default) refuses to unpickle. Store it as a tensor + scalars so the
# checkpoint stays loadable without weights_only=False.
def _np_rng_state_safe() -> dict:
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {"name": str(name), "keys": torch.from_numpy(np.asarray(keys, dtype=np.int64)),
            "pos": int(pos), "has_gauss": int(has_gauss), "cached": float(cached)}


def _restore_np_rng(d: dict) -> None:
    keys = d["keys"].cpu().numpy().astype(np.uint32)
    np.random.set_state((d["name"], keys, d["pos"], d["has_gauss"], d["cached"]))


def lut_corpus_hash(records) -> str:
    h = hashlib.sha256()
    for k in sorted(r.residual_key for r in records):
        h.update(k.encode("utf-8"))
    return h.hexdigest()


def _lr_at(step: int, cfg: TokenizerConfig) -> float:
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if not cfg.lr_decay:
        return cfg.lr
    # cosine decay from lr -> lr_min over the post-warmup steps (polishes the fine tail
    # the p99/max/PSNR gates are strictest on).
    total = max(1, cfg.max_steps - cfg.warmup_steps)
    prog = min(1.0, max(0.0, (step - cfg.warmup_steps) / total))
    return cfg.lr_min + 0.5 * (cfg.lr - cfg.lr_min) * (1.0 + math.cos(math.pi * prog))


def save_checkpoint(path: str, model: VQVAE, opt, step: int, cfg: TokenizerConfig,
                    corpus_hash: str, extra: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "step": step,
            "arch_version": cfg.arch_version,
            "config": cfg.to_dict(),
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "lut_corpus_hash": corpus_hash,
            "tokenizer_weights_hash": hash_state_dict(model.state_dict()),
            "torch_rng": torch.get_rng_state(),
            "numpy_rng": _np_rng_state_safe(),
            "cuda_rng": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
            **(extra or {}),
        },
        path,
    )


def _prune_checkpoints(out_dir: str, keep_last: int) -> None:
    cks = sorted(glob.glob(os.path.join(out_dir, "ckpt_*.pt")),
                 key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))
    for p in cks[:-keep_last] if keep_last > 0 else []:
        os.remove(p)


@torch.no_grad()
def evaluate_dev(model: VQVAE, dev_records, cfg: TokenizerConfig) -> dict | None:
    """Reconstruction gate on the tokenizer-dev holdout (train-split, not eval-reserved)."""
    if not dev_records:
        return None
    model.eval()
    targets = data_mod.load_residual_arrays(dev_records)
    recons, codes = metrics_mod.reconstruct(model, targets)
    families = [r.source_family for r in dev_records]
    agg = metrics_mod.aggregate_reconstruction(targets, recons, families)
    cb = metrics_mod.codebook_stats(codes, cfg.codebook_size)
    gate = metrics_mod.evaluate_gate(agg, cb)
    model.train()
    return {"overall": agg["overall"], "per_family": agg["per_family"],
            "codebook": cb, "gate": gate}


def train(cfg: TokenizerConfig, records, out_dir: str, device: str = "cpu",
          resume: str | None = None, dev_records=None, log_fn=print) -> str:
    dev = torch.device(device)

    # Seed ONLY on a fresh start; on resume we restore the saved RNG below so the
    # continued run follows the same trajectory (the sampler and dead-code revival both
    # draw from the global RNG — reseeding here would fork it). Seed before model init so
    # a fresh run's weights are reproducible.
    resuming = bool(resume and os.path.exists(resume))
    if not resuming:
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)

    model = VQVAE(cfg).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    start_step = 0
    corpus_hash = lut_corpus_hash(records)

    if resuming:
        ck = torch.load(resume, map_location=dev)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["optimizer_state"])
        start_step = int(ck["step"])
        if ck.get("torch_rng") is not None:
            torch.set_rng_state(ck["torch_rng"].cpu())
        if ck.get("numpy_rng") is not None:
            _restore_np_rng(ck["numpy_rng"])
        if ck.get("cuda_rng") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ck["cuda_rng"]])
        log_fn(f"[resume] from {resume} @ step {start_step} (torch/numpy/cuda RNG restored)")

    ds = data_mod.ResidualDataset(records, augment=cfg.augment, scale_jitter=cfg.scale_jitter)
    sampler = data_mod.family_balanced_sampler(records, num_samples=cfg.batch_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler, drop_last=False)

    def batches():
        while True:
            for xb, _fam in loader:
                yield xb

    model.train()
    best = None
    t0 = time.time()
    it = batches()
    os.makedirs(out_dir, exist_ok=True)
    for step in range(start_step, cfg.max_steps):
        xb = next(it).to(dev)
        for pg in opt.param_groups:
            pg["lr"] = _lr_at(step, cfg)
        opt.zero_grad()
        out = model(xb)
        loss, comp = total_loss(out, xb, cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if step % 50 == 0:
            log_fn(f"step {step}/{cfg.max_steps} lr={_lr_at(step, cfg):.2e} " + " ".join(
                f"{k}={comp[k]:.4f}" for k in
                ("loss", "recon", "deltaE", "tail", "smooth", "clip", "neutral", "commit", "perplexity")))

        if cfg.eval_every and step > 0 and step % cfg.eval_every == 0:
            rep = evaluate_dev(model, dev_records, cfg)
            if rep:
                o = rep["overall"]
                fam_ns = " ".join(f"{f}:n={s['n']}{'' if s.get('enforced', True) else '*'}"
                                  for f, s in sorted(rep["per_family"].items()))
                log_fn(f"[dev@{step}] meanΔE={o['mean_deltae']:.3f} p95={o['p95_deltae']:.3f} "
                       f"p99={o['p99_deltae']:.3f} PSNR={o['mean_psnr']:.2f} pass={rep['gate']['pass']} "
                       f"alerts={rep['gate']['alerts']} | fam[{fam_ns}]  (*=per-family gate not enforced)")
                if best is None or o["mean_deltae"] < best:
                    best = o["mean_deltae"]
                    save_checkpoint(os.path.join(out_dir, "best.pt"), model, opt, step, cfg, corpus_hash,
                                    extra={"dev_report": rep})

        if cfg.ckpt_every and step > 0 and step % cfg.ckpt_every == 0:
            save_checkpoint(os.path.join(out_dir, f"ckpt_{step}.pt"), model, opt, step, cfg, corpus_hash)
            _prune_checkpoints(out_dir, cfg.keep_last)

    final = os.path.join(out_dir, f"ckpt_{cfg.max_steps}.pt")
    save_checkpoint(final, model, opt, cfg.max_steps, cfg, corpus_hash,
                    extra={"dev_report": evaluate_dev(model, dev_records, cfg)})
    log_fn(f"[done] {cfg.max_steps - start_step} steps in {time.time()-t0:.1f}s -> {final}")
    return final


# --- CLI ----------------------------------------------------------------------------
def _load_records(args):
    if args.manifest and os.path.exists(args.manifest):
        recs = data_mod.load_train_manifest(args.manifest)
        print(f"[data] {len(recs)} train records from manifest {args.manifest}")
        return recs
    recs, cov = data_mod.build_records_from_registry(root=args.root)
    print(f"[data] reconstructed {len(recs)} train records from registry; coverage={cov}")
    if cov.get("unresolved_no_row"):
        print(f"[data][WARN] {cov['unresolved_no_row']} residuals unresolvable from the persisted "
              f"registry (pipeline does not persist _residual_key). Full-scale training needs a "
              f"pipeline-emitted train manifest; see tokenizer/data.py 'KNOWN GAP'.")
    return recs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train the VQ LUT tokenizer (Stage 7). Runs only when invoked.")
    ap.add_argument("--root", default=".", help="repo root containing luts/ and data/")
    ap.add_argument("--manifest", default=None, help="train manifest jsonl (preferred over registry reconstruction)")
    ap.add_argument("--out", default="tokenizer/checkpoints", help="checkpoint output dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--augment", action="store_true",
                    help="enable neutral-preserving residual scale-jitter (train-only remedy for tail/codebook gate failures)")
    ap.add_argument("--scale-jitter", type=float, default=None, help="augmentation magnitude (default 0.05 when --augment set)")
    ap.add_argument("--no-lr-decay", action="store_true", help="disable cosine LR decay (constant LR after warmup)")
    ap.add_argument("--dev-frac", type=float, default=0.10,
                    help="tokenizer-dev holdout fraction, stratified per family (larger => per-family gate enforceable)")
    ap.add_argument("--smoke", type=int, default=0, help="if >0, use only N records + max_steps=200 (dev sanity)")
    args = ap.parse_args(argv)

    cfg = DEFAULT_CONFIG
    over = {}
    if args.max_steps is not None:
        over["max_steps"] = args.max_steps
    if args.batch_size is not None:
        over["batch_size"] = args.batch_size
    if args.lr is not None:
        over["lr"] = args.lr
    if args.no_lr_decay:
        over["lr_decay"] = False
    if args.augment:
        over["augment"] = True
        over["scale_jitter"] = args.scale_jitter if args.scale_jitter is not None else 0.05
    elif args.scale_jitter is not None:
        over["scale_jitter"] = args.scale_jitter
    if over:
        cfg = replace(cfg, **over)

    records = _load_records(args)
    if args.smoke:
        records = records[: args.smoke]
        cfg = replace(cfg, max_steps=min(cfg.max_steps, 200), ckpt_every=100, eval_every=100)
    if not records:
        print("[abort] no train records resolved; provide --manifest or run the data pipeline first")
        return 2

    train_recs, dev_recs = data_mod.dev_holdout(records, frac=args.dev_frac)
    print(f"[data] train={len(train_recs)} dev-holdout={len(dev_recs)} families={sorted({r.source_family for r in records})}")
    train(cfg, train_recs, out_dir=os.path.join(args.root, args.out) if not os.path.isabs(args.out) else args.out,
          device=args.device, resume=args.resume, dev_records=dev_recs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
