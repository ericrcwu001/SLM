"""Measure oracle@N / best-of-N behavioral fidelity for an adapter — the COVERAGE GATE.

The free-running collapse (docs/collapse_fix) could be a *coverage* problem (the model never samples a
good trajectory) or merely a *selection* problem (a good trajectory exists in N samples but greedy
misses it). This module decides which: for each held-out row it draws N samples, scores each by
behavioral fidelity, and reports

  * ``oracle@k`` = mean over rows of the MAX fidelity among the first k samples (the ceiling a perfect
    reranker could reach), for k in {1,4,8,16,32,64};
  * ``best_of_N`` = mean over rows of the fidelity of the sample the canonical reranker actually picks
    (``eval.behavioral_fidelity.rerank_key``) — the realistic deploy number.

Gate (see docs/collapse_fix/README.md): oracle@N >= ~0.6 → coverage good (do best-of-N + distillation);
0.3-0.6 → capped; <= ~0.3 → escalate to RL. Report the full oracle@k curve so the shape informs the call.

Prints one ``{"oracle_summary": {...}}`` JSON line + a human table. Heavy deps loaded lazily via
:func:`sft.loader.load_eval_model`; the aggregation helpers are pure numpy (unit-testable, no GPU).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data_pipeline.attribute_spec import ground_truth_attribute_spec_text
from eval.behavioral_fidelity import rerank_key, score_generation
from sft.example import load_rows, supported_rows
from sft.score_tokens import _load_config

DEFAULT_KS = (1, 4, 8, 16, 32, 64)


def score_row_samples(codes_list, spec_text, target_codes) -> list[dict]:
    """Score each of the N sampled code lists vs the (canonical) spec.

    A ``None`` (refusal) or non-64 sample becomes a ``{"behavioral_fidelity": 0.0, "collapsed": True}``
    record — counted as a miss, matching the shipped ``sft.score_tokens._run_behavioral`` accounting.
    """
    recs: list[dict] = []
    for codes in codes_list:
        if codes is None or len(codes) != 64:
            recs.append({"behavioral_fidelity": 0.0, "collapsed": True, "refused": codes is None})
        else:
            recs.append(score_generation(codes, spec_text, target_codes=target_codes))
    return recs


def oracle_and_best(recs_by_row: list[list[dict]], ks=DEFAULT_KS) -> dict:
    """Aggregate per-row sample records into the oracle@k curve + best-of-N (pure numpy)."""
    rows = [recs for recs in recs_by_row if recs]
    out: dict = {"rows": len(recs_by_row), "scored_rows": len(rows)}
    for k in ks:
        vals = [max((r.get("behavioral_fidelity") or 0.0) for r in recs[:k]) for recs in rows]
        out[f"oracle@{k}"] = float(np.mean(vals)) if vals else None
    best = [(max(recs, key=rerank_key).get("behavioral_fidelity") or 0.0) for recs in rows]
    out["best_of_N"] = float(np.mean(best)) if best else None
    # collapse rate of the reranker's picks — a healthy pick should rarely be collapsed
    picks_collapsed = [1.0 if max(recs, key=rerank_key).get("collapsed") else 0.0 for recs in rows]
    out["best_pick_collapse_rate"] = float(np.mean(picks_collapsed)) if picks_collapsed else None
    return out


def run(model, processor, cfg, *, n: int = 32, temperature: float = 0.7, top_p: float = 0.9,
        limit: int = 32, chunk: int = 16, input_field: str | None = None) -> dict:
    """Draw n samples per held-out row and aggregate. ``input_field`` defaults to ``cfg.input_field``."""
    from sft.generate import generate_codes_for_row_batch

    input_field = input_field or cfg.input_field
    rows = supported_rows(load_rows(cfg.active_rows_path), holdout=True)
    if limit:
        rows = rows[:limit]
    recs_by_row: list[list[dict]] = []
    for row in rows:
        spec = ground_truth_attribute_spec_text(row)               # canonical (SCORING)
        try:
            codes_list = generate_codes_for_row_batch(
                model, processor, row, input_field=input_field, n=n, chunk=chunk,
                sampling={"temperature": temperature, "top_p": top_p}, device=model.device)  # CONDITIONING
        except Exception as exc:  # noqa: BLE001 — skip a bad row, keep going
            print(f"[oracle][skip] {row.get('id')}: {type(exc).__name__}: {exc}")
            continue
        recs_by_row.append(score_row_samples(codes_list, spec, row.get("target_tokens")))
    summ = oracle_and_best(recs_by_row)
    summ.update({"n": n, "temperature": temperature, "top_p": top_p})
    return summ


def _print_table(tag: str, s: dict) -> None:
    curve = "  ".join(f"@{k}={s.get(f'oracle@{k}'):.3f}" for k in DEFAULT_KS if s.get(f"oracle@{k}") is not None)
    print(f"[oracle:{tag}] oracle {curve}")
    print(f"[oracle:{tag}] best_of_N={s.get('best_of_N')}  best_pick_collapse_rate="
          f"{s.get('best_pick_collapse_rate')}  scored_rows={s.get('scored_rows')}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/candidate_two_stage.json")
    ap.add_argument("--resized-model", default="models/base_resized")
    ap.add_argument("--adapter", required=True, help="trained adapter dir (models/sft_adapters/<run>)")
    ap.add_argument("--limit", type=int, default=32, help="held-out rows to sample; 0 = full holdout")
    ap.add_argument("--n", type=int, default=32, help="samples per row (oracle@k reported up to min(n,64))")
    ap.add_argument("--chunk", type=int, default=16, help="num_return_sequences per .generate call (memory cap)")
    ap.add_argument("--temperatures", default="0.7,1.0", help="comma-separated sampling temperatures")
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args(argv)

    cfg = _load_config(args.config)
    from sft.loader import load_eval_model
    model, processor = load_eval_model(cfg, args.resized_model, args.adapter)

    temps = [float(t) for t in str(args.temperatures).split(",") if t.strip()]
    summary: dict = {"adapter": args.adapter, "input_field": cfg.input_field, "limit": args.limit,
                     "n": args.n, "by_temperature": {}}
    for t in temps:
        s = run(model, processor, cfg, n=args.n, temperature=t, top_p=args.top_p,
                limit=args.limit, chunk=args.chunk)
        summary["by_temperature"][f"{t}"] = s
        _print_table(f"t={t}", s)

    print(json.dumps({"oracle_summary": summary}))
    # Gate hint (thresholds are provisional; read the full curve — see docs/collapse_fix/README.md).
    best = max((s.get("best_of_N") or 0.0) for s in summary["by_temperature"].values()) if temps else 0.0
    band = "GOOD (do 02+03)" if best >= 0.6 else ("CAPPED (ship 02; 03 capped)" if best > 0.3 else "RL (escalate)")
    print(f"[oracle][gate] best best_of_N across temps={best:.3f} -> {band}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
