"""Freeze a passing tokenizer checkpoint to ``tokenizer/final/`` (master-plan Stage 8).

Re-checks the reconstruction/tail/per-family/roundtrip gate on the tokenizer-dev holdout
and, only if it passes (or ``--allow-reviewed-exception`` for the max-ΔE clause), writes
the frozen decoder + codebook + manifest. It does NOT enable the runtime decoder stubs
(eval/lut_decoder.py, data_pipeline/tokenize_targets.py) — that wiring is a separate,
explicit step, printed as a reminder here.

Runs nothing on import; use ``python -m tokenizer.freeze --ckpt ...``.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch

from . import data as data_mod
from . import metrics as metrics_mod
from .config import TokenizerConfig
from .manifest import build_frozen_manifest
from .model import VQVAE


def load_model_from_checkpoint(ckpt_path: str, device: str = "cpu") -> tuple[VQVAE, dict, TokenizerConfig]:
    ck = torch.load(ckpt_path, map_location=device)
    cfg = TokenizerConfig(**ck["config"])
    model = VQVAE(cfg).to(device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    return model, ck, cfg


def run_gate(model: VQVAE, cfg: TokenizerConfig, dev_records) -> dict:
    targets = data_mod.load_residual_arrays(dev_records)
    recons, codes = metrics_mod.reconstruct(model, targets)
    families = [r.source_family for r in dev_records]
    agg = metrics_mod.aggregate_reconstruction(targets, recons, families)
    cb = metrics_mod.codebook_stats(codes, cfg.codebook_size)
    gate = metrics_mod.evaluate_gate(agg, cb)
    roundtrip = metrics_mod.roundtrip_contracts(model)
    return {"overall": agg["overall"], "per_family": agg["per_family"],
            "codebook": cb, "gate": gate, "roundtrip": roundtrip}


def freeze(ckpt_path: str, out_dir: str, dev_records, allow_exception: bool = False,
           device: str = "cpu", log_fn=print) -> tuple[bool, dict]:
    model, ck, cfg = load_model_from_checkpoint(ckpt_path, device)
    report = run_gate(model, cfg, dev_records)

    fam = report.get("per_family", {})
    log_fn("[freeze][families] " + " ".join(
        f"{f}:n={s['n']}{'' if s.get('enforced', True) else '*'}" for f, s in sorted(fam.items()))
        + "  (*=too few dev rows; per-family gate not enforced)")

    checks = dict(report["gate"]["checks"])
    # the max-ΔE clause allows a reviewed exception (model_architecture.md / Stage 1)
    if allow_exception and "max_deltae" in checks:
        checks["max_deltae"] = True
    gate_pass = all(checks.values()) and report["roundtrip"]["pass"]

    if not gate_pass:
        failed = [k for k, v in checks.items() if not v]
        if not report["roundtrip"]["pass"]:
            failed.append("roundtrip:" + ",".join(k for k, v in report["roundtrip"]["checks"].items() if not v))
        log_fn(f"[freeze][ABORT] gate not passed: {failed}; alerts={report['gate']['alerts']}")
        return False, report

    os.makedirs(out_dir, exist_ok=True)
    manifest = build_frozen_manifest(
        model, cfg,
        lut_corpus_hash=ck.get("lut_corpus_hash", "unknown"),
        tokenizer_weights_hash=ck.get("tokenizer_weights_hash", "unknown"),
        gate_report={"overall": report["overall"], "per_family": report["per_family"],
                     "codebook": report["codebook"], "gate_checks": checks,
                     "reviewed_exception": bool(allow_exception)},
    )
    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))
    torch.save(model.decoder.state_dict(), os.path.join(out_dir, "decoder.pt"))
    torch.save(model.encoder.state_dict(), os.path.join(out_dir, "encoder.pt"))
    np.save(os.path.join(out_dir, "codebook.npy"), model.vq.codebook.detach().cpu().numpy())
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    log_fn(f"[freeze][OK] {out_dir}/manifest.json  tokenizer_version={manifest['tokenizer_version']}")
    log_fn(f"           vq_codebook_sha256={manifest['vq_codebook_sha256'][:16]}… "
           f"vq_decoder_sha256={manifest['vq_decoder_sha256'][:16]}…")
    log_fn("[next] wiring (separate explicit step — NOT done here): set ENABLED=True and implement the "
           "encoder in data_pipeline/tokenize_targets.py and the decoder in eval/lut_decoder.py against "
           "this frozen manifest, then re-run per-target SFT-admission checks (mean<=3.0, p95<=6.0).")
    return True, report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Freeze a tokenizer checkpoint to tokenizer/final/ (Stage 8).")
    ap.add_argument("--ckpt", required=True, help="checkpoint to freeze (e.g. tokenizer/checkpoints/best.pt)")
    ap.add_argument("--root", default=".")
    ap.add_argument("--manifest", default=None, help="train manifest (to rebuild the tokenizer-dev holdout)")
    ap.add_argument("--out", default="tokenizer/final")
    ap.add_argument("--dev-frac", type=float, default=0.10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--allow-reviewed-exception", action="store_true",
                    help="waive only the max-ΔE clause (reviewed exception per Stage 1)")
    args = ap.parse_args(argv)

    if args.manifest and os.path.exists(args.manifest):
        records = data_mod.load_train_manifest(args.manifest)
    else:
        records, _cov = data_mod.build_records_from_registry(root=args.root)
    if not records:
        print("[abort] no records to build the dev holdout; provide --manifest")
        return 2
    _train, dev = data_mod.dev_holdout(records, frac=args.dev_frac)
    if not dev:
        print("[abort] tokenizer-dev holdout is empty (too few records at this scale)")
        return 2

    out = os.path.join(args.root, args.out) if not os.path.isabs(args.out) else args.out
    ok, _report = freeze(args.ckpt, out, dev, allow_exception=args.allow_reviewed_exception, device=args.device)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
