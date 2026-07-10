"""Decoder-free held-out token-accuracy metric for an SFT adapter (the bilevel objective).

Loads the resized base + a trained LoRA adapter, and for each HELD-OUT supported row
(:mod:`sft.holdout`, **unit-aware** since ADR 0024) runs ONE teacher-forced forward pass, argmaxes
the logits over the assistant span, and measures how often the model predicts the correct
``<lut_NNN>`` code token. This is:

  * FAITHFUL-ish — the 64 code tokens deterministically decode to the target residual LUT under the
    frozen VQ encoder (predicting the exact codes ≈ predicting the target LUT), so it is the best
    quality proxy obtainable WITHOUT enabling the frozen decoder (eval/lut_decoder.py stays disabled);
  * DETERMINISTIC — argmax, no sampling, so it is not polluted by generation noise (the bilevel
    variance gate stays meaningful);
  * CHEAP — one forward pass per held-out row (no autoregressive generation loop).

Per ADR 0024 (eval-honesty contract) this scorer:
  * scores the FULL held-out slice by default (``--limit 0``); ``--limit N`` is only a cost lever;
  * reports macro per-family token accuracy with **group-bootstrap CIs** clustered on
    ``split_unit_id`` (the leakage unit), alongside the overall micro accuracy;
  * enforces the **exact-64** invariant — a scored supported row must retain all 64 code positions
    (partial-truncation rows are rejected at :func:`sft.example.build_supervised_example` and, as
    defence-in-depth, skipped here), closing AUDIT F8.

Prints exactly one ``METRIC=<accuracy>`` sentinel line (direction=MAX; the overall micro token
accuracy on the unit-aware holdout) that the bilevel objective parser takes verbatim, plus a
``{"score_summary": {...}}`` JSON line. Exits non-zero (no METRIC) if no held-out row could be
scored — so a mis-rooted corpus can never yield a bogus metric.

Heavy deps (torch/transformers/peft) are imported lazily. Runs on the Colab GPU stack.

Usage (on Colab, after training an adapter):
    python -m sft.score_tokens --resized-model models/base_resized \
        --adapter models/sft_adapters/bl_smoke200        # full holdout (honest default)
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from data_pipeline.errors import SFTError
from eval.vocab import code_token
from sft.config import SFTConfig
from sft.example import build_supervised_example, load_rows, supported_rows

_DEFAULT_CFG_PATH = Path("configs/sft_default.yaml")
_CODES_PER_ROW = 64          # exact-64 invariant (ADR 0024 / AUDIT F8)
_CI_BOOTSTRAP_B = 2000       # cluster-bootstrap resamples for per-slice CIs (deterministic, cheap)


def _load_config(path: str | None) -> SFTConfig:
    p = Path(path) if path else _DEFAULT_CFG_PATH
    overrides = yaml.safe_load(p.read_text(encoding="utf-8")) or {} if p.exists() else {}
    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in overrides.items() if k in fields}
    return SFTConfig(**kw)


def _group_bootstrap_ratio(units, corrects, totals, *, B: int = _CI_BOOTSTRAP_B, seed: int = 0):
    """Cluster (unit-level) bootstrap CI of ``sum(correct)/sum(total)``.

    The scoring unit is ``split_unit_id`` — near-duplicate rows share a unit, so resampling UNITS
    (not rows) gives an honest CI that accounts for their correlation. Returns
    ``(point, ci_low, ci_high)``; the CI is ``None`` when there are <2 units (not estimable).
    """
    if not totals or sum(totals) == 0:
        return (None, None, None)
    agg: dict[str, list[float]] = {}
    for u, c, t in zip(units, corrects, totals):
        a = agg.setdefault(u, [0.0, 0.0])
        a[0] += c
        a[1] += t
    uc = np.array([v[0] for v in agg.values()], dtype=float)
    ut = np.array([v[1] for v in agg.values()], dtype=float)
    point = float(uc.sum() / ut.sum())
    n = uc.size
    if n < 2 or B <= 0:
        return (point, None, None)
    rng = np.random.default_rng(seed)
    boots = np.empty(B, dtype=float)
    for i in range(B):
        idx = rng.integers(0, n, size=n)
        tt = ut[idx].sum()
        boots[i] = uc[idx].sum() / tt if tt > 0 else np.nan
    boots = boots[~np.isnan(boots)]
    if boots.size == 0:
        return (point, None, None)
    return (point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))


def summarize_scores(records: list[dict], *, B: int = _CI_BOOTSTRAP_B, seed: int = 0) -> dict:
    """Aggregate per-row scoring records into the honest score summary (ADR 0024).

    Each record: ``{"unit": split_unit_id, "family": source_family, "correct": int, "total": int,
    "exact": bool}``. Reports overall MICRO accuracy (the METRIC) + a unit-clustered CI, MACRO
    per-family accuracy, and per-family breakdowns with their own CIs. Pure (numpy only) so it is
    unit-testable without the GPU stack.
    """
    scored_rows = len(records)
    total_correct = sum(r["correct"] for r in records)
    total_pos = sum(r["total"] for r in records)
    exact = sum(1 for r in records if r["exact"])
    overall = total_correct / total_pos if total_pos else 0.0

    _, olo, ohi = _group_bootstrap_ratio(
        [r["unit"] for r in records], [r["correct"] for r in records],
        [r["total"] for r in records], B=B, seed=seed)

    by_fam: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_fam[r.get("family") or "unknown"].append(r)
    per_family: dict[str, dict] = {}
    fam_accs: list[float] = []
    for fam in sorted(by_fam):
        recs = by_fam[fam]
        ft = sum(r["total"] for r in recs)
        fc = sum(r["correct"] for r in recs)
        acc = fc / ft if ft else 0.0
        _, flo, fhi = _group_bootstrap_ratio(
            [r["unit"] for r in recs], [r["correct"] for r in recs],
            [r["total"] for r in recs], B=B, seed=seed)
        per_family[fam] = {
            "accuracy": acc, "ci_low": flo, "ci_high": fhi,
            "rows": len(recs), "units": len({r["unit"] for r in recs}),
            "code_positions": ft,
            "exact_match_rate": sum(1 for r in recs if r["exact"]) / len(recs),
        }
        fam_accs.append(acc)

    return {
        "metric": overall,
        "token_accuracy": overall,
        "overall_ci_low": olo, "overall_ci_high": ohi,
        "macro_family_accuracy": sum(fam_accs) / len(fam_accs) if fam_accs else 0.0,
        "exact_match_rate": exact / scored_rows if scored_rows else 0.0,
        "code_positions": total_pos, "correct": total_correct,
        "scored_rows": scored_rows, "scored_units": len({r["unit"] for r in records}),
        "per_family": per_family,
    }


def score(cfg: SFTConfig, resized_model: str, adapter: str, limit: int,
          *, input_field: str = "instruction", prep_row=None) -> dict:
    """Score an adapter's held-out token accuracy. ``input_field`` selects the generator input
    (``instruction`` one-stage, or ``attribute_spec_text`` two-stage); ``prep_row`` is an optional
    callable applied to each row before example construction (used by the oracle gate to stamp
    ``attribute_spec_text`` from ground-truth ``measured_behavior``)."""
    try:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig
        from peft import PeftModel
    except Exception as exc:  # noqa: BLE001
        raise SFTError(f"QLoRA stack unavailable (install the `sft` extra): {exc}") from exc
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _ModelCls
    except Exception:  # noqa: BLE001
        from transformers import AutoModelForVision2Seq as _ModelCls  # type: ignore

    from sft.example import resolve_compute_dtype
    compute_dtype = resolve_compute_dtype(cfg)   # bf16 on A100; auto fp16 on T4/Volta (no hw bf16)
    processor = AutoProcessor.from_pretrained(resized_model, trust_remote_code=True,
                                              min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels)
    tok = processor.tokenizer

    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit, bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant, bnb_4bit_compute_dtype=compute_dtype)
    base = _ModelCls.from_pretrained(resized_model, quantization_config=bnb, torch_dtype=compute_dtype,
                                     device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()

    # The 256 code-token ids in the resized vocab; accuracy is measured ONLY over these positions
    # (ignore <lut_bos>/<lut_eos>/chat-template tokens the model gets trivially).
    code_ids = torch.tensor([tok.convert_tokens_to_ids(code_token(k)) for k in range(256)],
                            device=model.device)

    rows = supported_rows(load_rows(cfg.active_rows_path), holdout=True)
    if limit:                     # 0 = score the FULL held-out slice (honest default, ADR 0024)
        rows = rows[:limit]
    if not rows:
        raise SFTError("no held-out supported rows to score (empty holdout slice)")

    records: list[dict] = []
    skipped = partial = 0
    for row in rows:
        if prep_row is not None:
            prep_row(row)
        try:
            batch = build_supervised_example(processor, row, cfg, device=model.device,
                                             input_field=input_field)
        except Exception as exc:  # noqa: BLE001 — skip a bad/missing-image/truncated row
            skipped += 1
            print(f"[score][skip] {row.get('id')}: {type(exc).__name__}: {exc}")
            continue
        labels = batch.pop("labels")
        with torch.no_grad():
            logits = model(**batch).logits
        pred = logits[:, :-1, :].argmax(dim=-1)          # predicts token t+1
        gold = batch["input_ids"][:, 1:]
        mask = labels[:, 1:] != -100                      # assistant span only
        is_code = torch.isin(gold, code_ids)
        sel = mask & is_code
        n_sel = int(sel.sum())
        if n_sel == 0:
            skipped += 1
            continue
        if n_sel != _CODES_PER_ROW:                       # exact-64 defence-in-depth (ADR 0024 / F8)
            partial += 1
            print(f"[score][partial] {row.get('id')}: {n_sel} code positions != {_CODES_PER_ROW} "
                  f"(truncated target — not scored)")
            continue
        n_hit = int(((pred == gold) & sel).sum())
        records.append({
            "unit": row.get("split_unit_id") or row.get("id", ""),
            "family": row.get("source_family"),
            "correct": n_hit, "total": n_sel, "exact": n_hit == n_sel,
        })

    if not records:
        raise SFTError("scored 0 code positions (all held-out rows skipped — check SLM_ARTIFACT_ROOT / images)")

    rep = summarize_scores(records)
    rep["skipped"] = skipped
    rep["partial"] = partial
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(_DEFAULT_CFG_PATH))
    ap.add_argument("--resized-model", default="models/base_resized")
    ap.add_argument("--adapter", required=True, help="trained adapter dir (models/sft_adapters/<run>)")
    ap.add_argument("--limit", type=int, default=0,
                    help="max held-out rows to score; 0 = full slice (honest default, ADR 0024)")
    args = ap.parse_args(argv)
    cfg = _load_config(args.config)
    try:
        rep = score(cfg, args.resized_model, args.adapter, args.limit)
    except SFTError as exc:
        print(json.dumps({"score_summary": {"error": str(exc)}}))
        print(f"[score][ABORT] {exc}")
        return 1
    print(json.dumps({"score_summary": rep}))
    print(f"METRIC={rep['metric']:.6f}")   # bilevel sentinel (direction=max), wins over all parsing
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
