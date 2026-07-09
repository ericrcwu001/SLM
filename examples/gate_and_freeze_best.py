#!/usr/bin/env python3
"""Gate several tokenizer checkpoints on the dev holdout and freeze the best PASSING one.

The MLX trainer writes ``best.pt`` selected by dev **mean** ΔE, but the Stage-1 gate is a
conjunction whose binding constraint here is the **tail** (p99 / max / p5-PSNR / hard family).
A checkpoint with a slightly higher mean can have a materially tighter tail and pass when the
mean-optimal one does not. So we gate ``best.pt`` + the late (low-LR, most-polished)
checkpoints with the authoritative float64 torch gate and freeze the passing checkpoint with
the most tail headroom. Prints a table; writes ``tokenizer/final/`` only on a real gate pass.

Usage: python examples/gate_and_freeze_best.py --ckpt-dir tokenizer/checkpoints_mlx --out tokenizer/final
"""
from __future__ import annotations

import argparse
import glob
import os

from tokenizer import data as D
from tokenizer import freeze as F
from tokenizer import metrics as M


def _slack(o, cb, thr=M.GATE):
    """Min normalized slack across the hard overall checks (>0 => passes that check)."""
    checks = {
        "mean": (thr.mean_deltae - o["mean_deltae"]) / thr.mean_deltae,
        "p95": (thr.p95_deltae - o["p95_deltae"]) / thr.p95_deltae,
        "p99": (thr.p99_deltae - o["p99_deltae"]) / thr.p99_deltae,
        "max": (thr.max_deltae - o["max_deltae"]) / thr.max_deltae,
        "psnr": (o["mean_psnr"] - thr.mean_psnr) / thr.mean_psnr,
        "p5psnr": (o["p5_psnr"] - thr.p5_psnr) / thr.p5_psnr,
    }
    return min(checks.values()), checks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="tokenizer/checkpoints_mlx")
    ap.add_argument("--out", default="tokenizer/final")
    ap.add_argument("--root", default=".")
    ap.add_argument("--dev-frac", type=float, default=0.10)
    ap.add_argument("--last-n", type=int, default=6, help="also gate the N newest ckpt_*.pt")
    args = ap.parse_args(argv)

    recs, _ = D.build_records_from_registry(root=args.root)
    _tr, dev = D.dev_holdout(recs, frac=args.dev_frac)
    print(f"[gate] dev holdout: {len(dev)} LUTs")

    # candidate set: best.pt + the newest N step checkpoints (highest step first)
    cands = []
    best = os.path.join(args.ckpt_dir, "best.pt")
    if os.path.exists(best):
        cands.append(best)
    steps = sorted(
        glob.glob(os.path.join(args.ckpt_dir, "ckpt_*.pt")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]),
        reverse=True,
    )
    cands += steps[: args.last_n]

    results = []
    print(f"[gate] evaluating {len(cands)} checkpoints (float64 authoritative gate)")
    for ck in cands:
        try:
            model, _c, cfg = F.load_model_from_checkpoint(ck, device="cpu")
            rep = F.run_gate(model, cfg, dev)
        except Exception as exc:  # noqa: BLE001
            print(f"  {os.path.basename(ck):18s} ERROR {exc}")
            continue
        o, cb = rep["overall"], rep["codebook"]
        gate = rep["gate"]
        passed = all(gate["checks"].values()) and rep["roundtrip"]["pass"]
        slack, _ = _slack(o, cb)
        sw = rep["per_family"].get("scraped_web", {})
        results.append((ck, passed, slack, o, cb, rep, sw))
        print(f"  {os.path.basename(ck):18s} pass={passed!s:5s} slack={slack:+.3f} "
              f"mean={o['mean_deltae']:.3f} p95={o['p95_deltae']:.3f} p99={o['p99_deltae']:.3f} "
              f"max={o['max_deltae']:.2f} PSNR={o['mean_psnr']:.2f} p5PSNR={o['p5_psnr']:.2f} "
              f"sw_mean={sw.get('mean_deltae', float('nan')):.3f} sw_p95={sw.get('p95_deltae', float('nan')):.3f}")

    passers = [r for r in results if r[1]]
    if passers:
        # among passers, prefer the most overall slack (most robust margin)
        winner = max(passers, key=lambda r: r[2])
        ck = winner[0]
        print(f"[gate] PASS — freezing {os.path.basename(ck)} (slack {winner[2]:+.3f})")
        ok, _rep = F.freeze(ck, os.path.join(args.root, args.out) if not os.path.isabs(args.out) else args.out,
                            dev, device="cpu")
        print(f"[gate] freeze wrote tokenizer/final: {ok}")
        return 0 if ok else 1

    # no passer — report the closest (max slack) for diagnosis
    if results:
        closest = max(results, key=lambda r: r[2])
        print(f"[gate] NO CHECKPOINT PASSED. Closest: {os.path.basename(closest[0])} slack={closest[2]:+.3f}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
