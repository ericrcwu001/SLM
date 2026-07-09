#!/usr/bin/env python3
"""Freeze the tokenizer under a gate-owner REVIEWED EXCEPTION for the p5_psnr check.

Context (2026-07-09): after bilevel tuning + 4 full runs, no checkpoint passes the Stage-1
gate outright. Run-4 `best.pt` passes every check EXCEPT p5_psnr (28.37 < 30). The per-LUT
diagnostic shows the entire worst-5% tail is `diagnostic_only`-tier (non-headline-eligible per
data_collection_plan.md / schemas.py); on the gold (headline-eligible) population best.pt passes
the full gate incl. p5_psnr=31.24. The gate owner (user) has signed off on waiving p5_psnr on
that documented basis. Thresholds are NOT changed — the true measured value is recorded in the
manifest (raw_gate_checks + overall) alongside the waiver + rationale, so the exception is auditable.
"""
from __future__ import annotations

import json
import numpy as np
from collections import Counter

from tokenizer import data as D, freeze as F, metrics as M

CKPT = "tokenizer/checkpoints_mlx/best.pt"
OUT = "tokenizer/final"
NAMED = {305: "web_forum_blackmagicdesign_com_faux_infrared",
         316: "web_forum_blackmagicdesign_com_green_mono",
         328: "web_forum_blackmagicdesign_com_solarized_color",
         344: "web_gmic_eu_avalanche"}


def main() -> int:
    recs, _ = D.build_records_from_registry(root=".")
    _tr, dev = D.dev_holdout(recs, frac=0.10)
    tiers = np.array([r.source_family for r in dev])
    tier = np.array([r.representability_tier for r in dev])

    model, _c, _cfg = F.load_model_from_checkpoint(CKPT, device="cpu")
    tgt = D.load_residual_arrays(dev)
    recons, _ = M.reconstruct(model, tgt)
    psnr = np.array([M.lut_psnr(t, r) for t, r in zip(tgt, recons)])

    p5_all = float(np.percentile(psnr, 5))
    p5_gold = float(np.percentile(psnr[tier == "gold"], 5))
    worst = np.argsort(psnr)[: int(round(0.05 * len(dev)))]
    worst_tiers = dict(Counter(tier[worst].tolist()))

    note = (
        "GATE-OWNER REVIEWED EXCEPTION (signed off by user, 2026-07-09). "
        f"Waived check: p5_psnr (measured {p5_all:.2f} dB on the full {len(dev)}-LUT dev holdout; "
        "threshold >=30.0; threshold UNCHANGED). "
        "Basis: (1) the worst-5% PSNR tail is 100% representability_tier=diagnostic_only "
        f"({worst_tiers}) — zero gold/headline-eligible LUTs are in it; "
        f"(2) on the gold (headline-eligible) subset best.pt PASSES the full gate incl. p5_psnr={p5_gold:.2f}>=30; "
        "(3) the worst LUTs are extreme diagnostic-only special-effect LUTs that violate the tokenizer's "
        "smoothness/monotonicity/full-3D-rank assumptions: "
        "#305 faux_infrared (extreme-magnitude remap), #316 green_mono (monochrome/rank-collapse), "
        "#328 solarized_color (non-monotonic tone reversal), #344 avalanche (roughest LUT in family). "
        "diagnostic_only is non-headline per data_collection_plan.md 'Derived LUT Representability Gate' "
        "and eval/schemas.py (headline rows require tier=gold). All other Stage-1 checks pass outright "
        "(mean/p95/p99/max deltaE, mean PSNR, per-family incl. scraped_web, roundtrip; codebook 100%/ppl~164)."
    )

    print("[freeze-exc] substantiating evidence:")
    print(f"  full-holdout p5_psnr = {p5_all:.2f}  (FAIL vs 30)")
    print(f"  gold-only    p5_psnr = {p5_gold:.2f}  (PASS vs 30)")
    print(f"  worst-5% tier breakdown = {worst_tiers}")
    print(f"  named worst LUTs = {NAMED}")
    print()

    ok, _rep = F.freeze(CKPT, OUT, dev, waive_checks=("p5_psnr",), exception_note=note, device="cpu")
    print(f"\n[freeze-exc] freeze ok = {ok}")
    if ok:
        with open(f"{OUT}/manifest.json", encoding="utf-8") as fh:
            m = json.load(fh)
        print(f"[freeze-exc] tokenizer_version    = {m['tokenizer_version']}")
        print(f"[freeze-exc] vq_codebook_sha256    = {m['vq_codebook_sha256']}")
        print(f"[freeze-exc] vq_decoder_sha256     = {m['vq_decoder_sha256']}")
        print(f"[freeze-exc] reviewed_exception    = {json.dumps(m['gate_report']['reviewed_exception'])[:120]}…")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
