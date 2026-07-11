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


def _codebook_tensor(device):
    """The frozen ``[256,64]`` codebook as a float tensor (for embedding-distance credit), or None."""
    import numpy as _np
    import torch

    from tokenizer.frozen import frozen_final_dir
    try:
        cb = _np.load(frozen_final_dir() / "codebook.npy")
        return torch.tensor(cb, dtype=torch.float32, device=device)
    except Exception:  # noqa: BLE001 — diagnostics only; never break the sentinel
        try:
            from tokenizer.frozen import load_frozen_vqvae
            m, _ = load_frozen_vqvae()
            return m.vq.codebook.detach().to(device=device, dtype=torch.float32)
        except Exception:  # noqa: BLE001
            return None


def _run_behavioral(model, processor, rows, *, input_field, bucketize, sampling, device):
    """One free-running behavioral pass over ``rows`` under a given decode (``sampling`` None=greedy).

    Generates each row's codes (matching the trained conditioning), decodes + re-measures behavior,
    scores agreement vs the CANONICAL requested spec (bucketize only affects the model INPUT), and
    aggregates. Returns the :func:`eval.behavioral_fidelity.summarize_fidelity` dict."""
    from data_pipeline.attribute_spec import ground_truth_attribute_spec_text
    from eval.behavioral_fidelity import score_generation, summarize_fidelity
    from sft.generate import generate_codes_for_row

    brecs: list[dict] = []
    for row in rows:
        spec_text = ground_truth_attribute_spec_text(row)   # what the LUT SHOULD do (canonical)
        try:
            codes = generate_codes_for_row(model, processor, row, input_field=input_field,
                                           bucketize=bucketize, sampling=sampling, device=device)
        except Exception as exc:  # noqa: BLE001
            print(f"[score][bhv-skip] {row.get('id')}: gen {type(exc).__name__}: {exc}")
            continue
        if codes is None:            # refusal on a supported row == total miss
            brecs.append({"route": "grade", "behavioral_fidelity": 0.0, "residual_norm": 0.0,
                          "collapsed": True, "degenerate_identity": True, "refused": True})
            continue
        try:
            brecs.append(score_generation(codes, spec_text, target_codes=row.get("target_tokens")))
        except Exception as exc:  # noqa: BLE001
            print(f"[score][bhv-skip] {row.get('id')}: score {type(exc).__name__}: {exc}")
    bsum = summarize_fidelity(brecs)
    bsum["scored"] = len(brecs)
    bsum["refused"] = sum(1 for r in brecs if r.get("refused"))
    return bsum


def score(cfg: SFTConfig, resized_model: str, adapter: str, limit: int,
          *, input_field: str = "instruction", prep_row=None,
          behavioral: bool = True, behavioral_limit: int = 48,
          behavioral_sampling: str = "greedy", behavioral_temperature: float = 0.7,
          behavioral_top_p: float = 0.9) -> dict:
    """Score an adapter's held-out token accuracy. ``input_field`` selects the generator input
    (``instruction`` one-stage, or ``attribute_spec_text`` two-stage); ``prep_row`` is an optional
    callable applied to each row before example construction (used by the oracle gate to stamp
    ``attribute_spec_text`` from ground-truth ``measured_behavior``).

    The teacher-forced ``METRIC=`` sentinel (``rep["metric"]``) is UNCHANGED — it stays the locked
    bilevel contract. Additionally, unless ``behavioral=False``, a FREE-RUNNING pass generates each
    row's codes (``sft.generate``), decodes + re-measures behavior, and reports behavioral fidelity
    + collapse stats (``eval.behavioral_fidelity``) — the only metric that catches the exposure-bias
    collapse teacher forcing is blind to. ``behavioral_limit`` caps the (slow, autoregressive) free
    pass (0 = full holdout). Cheap teacher-forced diagnostics (top-5 code accuracy, embedding-distance
    partial credit, per-position accuracy) are computed from the same logits as *secondary lenses*."""
    from sft.loader import load_eval_model  # shared loader (raises SFTError if the stack is missing)
    model, processor = load_eval_model(cfg, resized_model, adapter)
    import torch
    tok = processor.tokenizer

    # The 256 code-token ids in the resized vocab; accuracy is measured ONLY over these positions
    # (ignore <lut_bos>/<lut_eos>/chat-template tokens the model gets trivially).
    code_ids = torch.tensor([tok.convert_tokens_to_ids(code_token(k)) for k in range(256)],
                            device=model.device)

    rows = supported_rows(load_rows(cfg.active_rows_path), holdout=True)
    if limit:                     # 0 = score the FULL held-out slice (honest default, ADR 0024)
        rows = rows[:limit]
    if not rows:
        raise SFTError("no held-out supported rows to score (empty holdout slice)")

    # Teacher-forced diagnostics (secondary lenses; never break the sentinel): the [256,64]
    # codebook enables embedding-distance partial credit, and dmax scales it to [0,1].
    cb = _codebook_tensor(model.device)
    dmax = float(torch.cdist(cb, cb).max()) if cb is not None else None
    if not dmax:
        dmax = None
    pos_correct = torch.zeros(_CODES_PER_ROW, device=model.device)
    pos_total = 0
    tf_top5_hits = 0
    tf_emb_credit = 0.0
    tf_positions = 0

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

        # --- teacher-forced diagnostics (from the SAME forward pass; failures are non-fatal) ---
        try:
            sel_pos = sel[0].nonzero(as_tuple=True)[0]           # the 64 code positions (in order)
            pred_codes = pred[0][sel_pos]
            gold_codes = gold[0][sel_pos]
            pos_correct += (pred_codes == gold_codes).float()    # per-position 0/1 across rows
            pos_total += 1
            lg = logits[0, :-1, :][sel_pos][:, code_ids]         # [64, 256] logits over codes only
            gold_idx = (code_ids.unsqueeze(0) == gold_codes.unsqueeze(1)).int().argmax(1)  # [64]
            top5 = lg.topk(5, dim=-1).indices
            tf_top5_hits += int((top5 == gold_idx.unsqueeze(1)).any(1).sum())
            if cb is not None and dmax:
                d = (cb[lg.argmax(-1)] - cb[gold_idx]).norm(dim=-1)   # code-embedding distance
                tf_emb_credit += float((1.0 - (d / dmax).clamp(0.0, 1.0)).sum())
            tf_positions += _CODES_PER_ROW
        except Exception as exc:  # noqa: BLE001
            print(f"[score][diag-skip] {row.get('id')}: {type(exc).__name__}: {exc}")

    if not records:
        raise SFTError("scored 0 code positions (all held-out rows skipped — check SLM_ARTIFACT_ROOT / images)")

    rep = summarize_scores(records)
    rep["skipped"] = skipped
    rep["partial"] = partial

    # --- teacher-forced diagnostic aggregates (secondary lenses) ---
    if pos_total:
        ppa = (pos_correct / pos_total).tolist()
        rep["per_position_accuracy"] = ppa
        rep["per_position_accuracy_min"] = float(min(ppa))
        rep["per_position_accuracy_max"] = float(max(ppa))
    if tf_positions:
        rep["top5_code_accuracy"] = tf_top5_hits / tf_positions
        if cb is not None and dmax:
            rep["embedding_partial_credit"] = tf_emb_credit / tf_positions

    # --- free-running BEHAVIORAL fidelity (the ruler that catches the collapse) ---
    # ``behavioral_sampling``: "greedy" (default), "sample" (t=behavioral_temperature), or "both"
    # (Phase-0 showed greedy over-commits to a dominant code while t=0.7 recovers target strength —
    # "both" measures agreement under each so the gap is quantified in one pass).
    if behavioral:
        try:
            brows = rows if not behavioral_limit else rows[:behavioral_limit]
            bucketize = getattr(cfg, "spec_bucketize", False)
            samp = {"temperature": behavioral_temperature, "top_p": behavioral_top_p}
            modes = {"greedy": [("behavioral", None)],
                     "sample": [("behavioral_sampled", samp)],
                     "both": [("behavioral", None), ("behavioral_sampled", samp)]}[behavioral_sampling]
            for key, sampling in modes:
                bsum = _run_behavioral(model, processor, brows, input_field=input_field,
                                       bucketize=bucketize, sampling=sampling, device=model.device)
                rep[key] = bsum
                tag = "greedy" if key == "behavioral" else f"t={behavioral_temperature}"
                print(f"[score][behavioral:{tag}] fidelity={bsum.get('behavioral_fidelity_mean')} "
                      f"collapse_rate={bsum.get('collapse_rate')} "
                      f"resid_median={bsum.get('residual_norm_median')} "
                      f"scored={bsum['scored']} refused={bsum['refused']}")
        except Exception as exc:  # noqa: BLE001 — behavioral pass must never break the sentinel
            rep["behavioral"] = {"error": f"{type(exc).__name__}: {exc}"}
            print(f"[score][behavioral][ERR] {exc}")

    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(_DEFAULT_CFG_PATH))
    ap.add_argument("--resized-model", default="models/base_resized")
    ap.add_argument("--adapter", required=True, help="trained adapter dir (models/sft_adapters/<run>)")
    ap.add_argument("--limit", type=int, default=0,
                    help="max held-out rows to score; 0 = full slice (honest default, ADR 0024)")
    ap.add_argument("--no-behavioral", dest="behavioral", action="store_false",
                    help="skip the free-running behavioral-fidelity pass (teacher-forced METRIC only)")
    ap.add_argument("--behavioral-limit", type=int, default=48,
                    help="cap rows in the (slow) free-running behavioral pass; 0 = full holdout")
    ap.add_argument("--behavioral-sampling", choices=["greedy", "sample", "both"], default="greedy",
                    help="decode for the behavioral pass; 'both' compares greedy vs sampling (Phase 0)")
    ap.add_argument("--behavioral-temperature", type=float, default=0.7)
    ap.add_argument("--behavioral-top-p", type=float, default=0.9)
    args = ap.parse_args(argv)
    cfg = _load_config(args.config)
    try:
        # Score with the SAME conditioning input the adapter was trained on (cfg.input_field):
        # "instruction" (one-stage) or "attribute_spec_text" (two-stage, P6). Two-stage rows derive
        # the ground-truth spec on the fly (sft.example.input_text_for), so no corpus rewrite.
        rep = score(cfg, args.resized_model, args.adapter, args.limit, input_field=cfg.input_field,
                    behavioral=args.behavioral, behavioral_limit=args.behavioral_limit,
                    behavioral_sampling=args.behavioral_sampling,
                    behavioral_temperature=args.behavioral_temperature,
                    behavioral_top_p=args.behavioral_top_p)
    except SFTError as exc:
        print(json.dumps({"score_summary": {"error": str(exc)}}))
        print(f"[score][ABORT] {exc}")
        return 1
    print(json.dumps({"score_summary": rep}))
    print(f"METRIC={rep['metric']:.6f}")   # bilevel sentinel (direction=max), wins over all parsing
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
