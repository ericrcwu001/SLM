#!/usr/bin/env python3
"""Per-LUT tail diagnostic for the tokenizer Stage-1 gate.

Answers "which LUTs fail, by how much, and does the error sit where real images live?"
for a single checkpoint on the frozen dev holdout. Prints:
  * the overall worst-N LUTs by mean-node ΔE (these drive p5_psnr and p99/max),
  * every enforced family's mean/p95 with pass/fail vs the per-family gate,
  * for the hardest family (scraped_web), each LUT over the p95 threshold, with its
    error split into gamut-BOUNDARY nodes (any of r,g,b at 0 or N-1 — saturated
    extremes real photos rarely hit) vs INTERIOR nodes (mid-gamut, where real pixels
    cluster). A boundary-concentrated error overstates real-image impact.

Read-only; uses the authoritative float64 torch path (freeze.load_model_from_checkpoint)
and the NumPy CIEDE2000 gate primitives. Does not train or write anything.

Usage: python examples/diagnose_tail.py --ckpt tokenizer/checkpoints_mlx/ckpt_40000.pt
"""
from __future__ import annotations

import argparse
import numpy as np

from tokenizer import data as D
from tokenizer import freeze as F
from tokenizer import metrics as M


def _boundary_mask(n: int) -> np.ndarray:
    """[n,n,n] bool: True where any axis index is 0 or n-1 (cube surface / saturated extremes)."""
    idx = np.arange(n)
    edge = (idx == 0) | (idx == n - 1)
    ex, ey, ez = np.meshgrid(edge, edge, edge, indexing="ij")
    return ex | ey | ez


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="tokenizer/checkpoints_mlx/best.pt")
    ap.add_argument("--root", default=".")
    ap.add_argument("--dev-frac", type=float, default=0.10)
    ap.add_argument("--family", default="scraped_web", help="family to break down node-wise")
    ap.add_argument("--top", type=int, default=12, help="how many worst overall LUTs to list")
    args = ap.parse_args(argv)

    recs, _ = D.build_records_from_registry(root=args.root)
    _tr, dev = D.dev_holdout(recs, frac=args.dev_frac)
    fams = [r.source_family for r in dev]
    targets = D.load_residual_arrays(dev)

    model, _c, cfg = F.load_model_from_checkpoint(args.ckpt, device="cpu")
    recons, _codes = M.reconstruct(model, targets)

    n = cfg.grid
    bmask = _boundary_mask(n).reshape(-1)
    imask = ~bmask

    per = []  # (idx, family, mean_dE, psnr, bnd_mean, int_mean)
    for i, (t, r) in enumerate(zip(targets, recons)):
        nodes = M.lut_deltae_nodes(t, r)          # [n**3] float64 CIEDE2000 per node
        per.append((i, fams[i], float(nodes.mean()), M.lut_psnr(t, r),
                    float(nodes[bmask].mean()), float(nodes[imask].mean())))

    dE = np.array([p[2] for p in per])
    psnr = np.array([p[3] for p in per])
    print(f"[diag] ckpt={args.ckpt}  dev={len(dev)} LUTs")
    print(f"[diag] overall: mean ΔE={dE.mean():.3f}  p95={np.percentile(dE,95):.3f}  "
          f"p99={np.percentile(dE,99):.3f}  max={dE.max():.2f}  "
          f"mean PSNR={psnr.mean():.2f}  p5 PSNR={np.percentile(psnr,5):.2f}  (gate p5≥30)")

    print(f"\n[diag] worst {args.top} LUTs overall (drive p5_psnr / p99 / max):")
    print(f"  {'#':>3s} {'family':16s} {'meanΔE':>7s} {'PSNR':>6s} {'bndΔE':>7s} {'intΔE':>7s}")
    for p in sorted(per, key=lambda x: -x[2])[: args.top]:
        print(f"  {p[0]:>3d} {p[1]:16s} {p[2]:7.3f} {p[3]:6.2f} {p[4]:7.3f} {p[5]:7.3f}")

    print(f"\n[diag] enforced-family gate (mean≤{M.GATE.per_family_mean_deltae}, p95≤{M.GATE.per_family_p95_deltae}):")
    fam_arr = np.array(fams)
    for f in sorted(set(fams)):
        m = fam_arr == f
        nrows = int(m.sum())
        fam_mean = float(dE[m].mean())
        fam_p95 = float(np.percentile(dE[m], 95))
        enf = nrows >= M.GATE.min_family_rows
        ok = (fam_mean <= M.GATE.per_family_mean_deltae) and (fam_p95 <= M.GATE.per_family_p95_deltae)
        tag = ("PASS" if ok else "FAIL") if enf else "not-enforced(n<30)"
        print(f"  {f:16s} n={nrows:3d} mean={fam_mean:.3f} p95={fam_p95:.3f}  {tag}")

    # node-wise breakdown of the hard family's over-threshold LUTs
    thr = M.GATE.per_family_p95_deltae
    fam_luts = [p for p in per if p[1] == args.family]
    over = sorted([p for p in fam_luts if p[2] > thr], key=lambda x: -x[2])
    print(f"\n[diag] '{args.family}' LUTs with mean-node ΔE > {thr} "
          f"({len(over)} of {len(fam_luts)}); bnd=saturated-extreme nodes, int=mid-gamut:")
    print(f"  {'#':>3s} {'meanΔE':>7s} {'PSNR':>6s} {'bndΔE':>7s} {'intΔE':>7s} {'int/mean':>8s}")
    for p in over:
        ratio = p[5] / p[2] if p[2] else float("nan")
        print(f"  {p[0]:>3d} {p[2]:7.3f} {p[3]:6.2f} {p[4]:7.3f} {p[5]:7.3f} {ratio:8.2f}")
    if over:
        int_share = np.mean([p[5] / p[2] for p in over if p[2]])
        print(f"\n[diag] across those LUTs, interior(mid-gamut) ΔE is on avg {int_share:.0%} of the "
              f"whole-LUT mean ΔE — the rest is concentrated on saturated gamut extremes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
