"""Non-learned PCA + scalar-quant LUT tokenizer baseline, scored on the tokenizer gate.

Purpose (audit action #7): establish whether the learned VQ-VAE earns its complexity. LUT
residuals live on a low-dimensional smooth manifold, so a deterministic PCA basis with
per-component scalar quantization is collapse-free and byte-reproducible by construction.
If this clears the SAME reconstruction gate the VQ-VAE targets, the learned machinery is
merely sufficient, not necessary; if it does not, the VQ-VAE is justified.

Bit-budget parity: the VQ tokenizer emits 64 tokens x 8 bits = 512 bits/LUT. PCA with
`dim` components at `bits` bits each = dim*bits bits/LUT — set dim=64, bits=8 for a
like-for-like 512-bit comparison.

Runs nothing on import. Invoke explicitly:
    python3 -m scripts.tokenizer_pca_baseline --root . [--dim 64 --bits 8 --dev-frac 0.10]
"""

from __future__ import annotations

import argparse

import numpy as np

from data_pipeline.leakage import fit_lut_pca
from tokenizer import data as data_mod
from tokenizer import metrics as metrics_mod


def _scalar_quantize(proj: np.ndarray, bits: int, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Per-component uniform mid-rise quantization to `bits`, then dequantize."""
    levels = (1 << bits) - 1
    span = np.maximum(hi - lo, 1e-12)
    q = np.clip(np.round((proj - lo) / span * levels), 0, levels)
    return lo + q / levels * span


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PCA + scalar-quant LUT tokenizer baseline on the gate.")
    ap.add_argument("--root", default=".")
    ap.add_argument("--manifest", default=None, help="train manifest jsonl (else registry reconstruction)")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--bits", type=int, default=8)
    ap.add_argument("--dev-frac", type=float, default=0.10)
    ap.add_argument("--max-train", type=int, default=3000, help="cap train arrays used to fit PCA (memory)")
    args = ap.parse_args(argv)

    if args.manifest:
        records = data_mod.load_train_manifest(args.manifest)
    else:
        records, cov = data_mod.build_records_from_registry(root=args.root)
        print(f"[data] registry coverage={cov}")
    if not records:
        print("[abort] no train records resolved; provide --manifest or run the pipeline")
        return 2

    train, dev = data_mod.dev_holdout(records, frac=args.dev_frac)
    if not dev:
        print("[abort] empty dev holdout")
        return 2
    print(f"[data] train={len(train)} dev={len(dev)} families={sorted({r.source_family for r in records})}")

    train_arrays = data_mod.load_residual_arrays(train[: args.max_train])
    pca = fit_lut_pca([a.reshape(-1) for a in train_arrays], dim=args.dim)
    proj_train = np.stack([pca.project(a.reshape(-1)) for a in train_arrays])
    lo, hi = proj_train.min(0), proj_train.max(0)

    dev_arrays = data_mod.load_residual_arrays(dev)
    fams = [r.source_family for r in dev]
    shape = dev_arrays[0].shape
    recons = []
    for a in dev_arrays:
        pq = _scalar_quantize(pca.project(a.reshape(-1)), args.bits, lo, hi)
        recons.append((pca.mean + pq @ pca.components).reshape(shape))

    agg = metrics_mod.aggregate_reconstruction(dev_arrays, recons, fams)
    # PCA has no codebook; feed evaluate_gate a trivially-healthy usage stub (it only reads
    # active_frac and perplexity for its non-blocking alerts).
    cb = {"active_frac": 1.0, "perplexity": float(1 << args.bits),
          "active_codes": args.dim, "dead_code_count": 0, "top_code_share": 0.0}
    gate = metrics_mod.evaluate_gate(agg, cb)
    o = agg["overall"]
    print(f"[pca-{args.dim} q{args.bits}b | {args.dim * args.bits} bits vs vq 512 bits] "
          f"meanΔE={o['mean_deltae']:.3f} p95={o['p95_deltae']:.3f} p99={o['p99_deltae']:.3f} "
          f"max={o['max_deltae']:.3f} PSNR={o['mean_psnr']:.2f} pass={gate['pass']}")
    print(f"[gate.checks] {gate['checks']}")
    for f, s in sorted(agg["per_family"].items()):
        flag = "" if s.get("enforced", True) else "  (n<min, not enforced)"
        print(f"   family {f}: n={s['n']} meanΔE={s['mean_deltae']:.3f} p95={s['p95_deltae']:.3f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
