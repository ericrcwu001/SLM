"""Best-of-N generation reranked by behavioral fidelity (inference; docs/collapse_fix/02).

Sample N candidate 64-code sequences, decode + score each against the REQUESTED spec, and return the
reranker-best VALID one. The reranker (:func:`eval.behavioral_fidelity.rerank_key`) uses only the
requested spec + generated codes — **no target LUT** — so this is the deployable path. It ~doubled
free-running fidelity in the P6 gate (0.16 greedy → ~0.30), bounded by the model's sampling coverage
(``oracle@N``; see :mod:`eval.oracle_at_n`).

The scoring/reranking is thin orchestration over existing pieces; the heavy generation runs on Colab.
"""

from __future__ import annotations

import argparse
import json

from data_pipeline.attribute_spec import ground_truth_attribute_spec_text
from eval.behavioral_fidelity import rerank_key, score_generation, summarize_fidelity
from sft.example import load_rows, supported_rows
from sft.score_tokens import _load_config


def best_of_n_codes(model, processor, *, image, cond_text, spec_text=None, n: int = 16,
                    sampling: dict | None = None, chunk: int = 16, device=None):
    """Return ``(best_codes, best_record)``. Generate ``n`` samples CONDITIONED on ``cond_text``, score
    each against ``spec_text`` (the REQUESTED spec; defaults to ``cond_text`` — identical at deploy),
    and return the reranker-best valid candidate. Returns ``(None, {...})`` if every sample refused or
    was malformed. No target LUT needed (fidelity is agreement with the request)."""
    from sft.generate import generate_codes_batch

    spec_text = spec_text or cond_text
    sampling = sampling or {"temperature": 0.7, "top_p": 0.9}
    cand = generate_codes_batch(model, processor, image=image, text=cond_text, n=n,
                                sampling=sampling, chunk=chunk, device=device)
    scored = [(codes, score_generation(codes, spec_text))
              for codes in cand if codes is not None and len(codes) == 64]
    if not scored:
        return None, {"behavioral_fidelity": None, "refused_all": True}
    return max(scored, key=lambda t: rerank_key(t[1]))


def best_of_n_for_row(model, processor, row: dict, *, n: int = 16, sampling: dict | None = None,
                      input_field: str = "attribute_spec_text", chunk: int = 16, device=None):
    """Row convenience: CONDITION via ``input_text_for`` (matches training), SCORE via the canonical spec."""
    from sft.example import input_text_for, resolve_image

    return best_of_n_codes(
        model, processor, image=resolve_image(row["image_path"]),
        cond_text=input_text_for(row, input_field),                 # conditioning (training parity)
        spec_text=ground_truth_attribute_spec_text(row),            # scoring (canonical)
        n=n, sampling=sampling, chunk=chunk, device=device)


def evaluate(model, processor, cfg, *, n: int = 16, temperature: float = 1.0, top_p: float = 0.9,
             limit: int = 32, chunk: int = 16, input_field: str | None = None) -> dict:
    """Best-of-N over the held-out slice; returns a ``summarize_fidelity`` dict of the reranker picks.

    An all-refused row is folded in as a ``behavioral_fidelity=0.0`` record (matching the shipped
    ``sft.score_tokens._run_behavioral`` accounting), so this is directly comparable to the greedy
    baseline (0.159) and to ``eval.oracle_at_n``'s ``best_of_N`` at the same ``n`` and slice."""
    input_field = input_field or cfg.input_field
    rows = supported_rows(load_rows(cfg.active_rows_path), holdout=True)
    if limit:
        rows = rows[:limit]
    recs: list[dict] = []
    for row in rows:
        try:
            codes, rec = best_of_n_for_row(model, processor, row, n=n, input_field=input_field,
                                           sampling={"temperature": temperature, "top_p": top_p},
                                           chunk=chunk, device=model.device)
        except Exception as exc:  # noqa: BLE001
            print(f"[bestofn][skip] {row.get('id')}: {type(exc).__name__}: {exc}")
            continue
        if codes is None:            # all N refused/malformed -> total miss (same accounting as greedy)
            recs.append({"behavioral_fidelity": 0.0, "collapsed": True, "refused": True})
        else:
            recs.append(rec)
    summ = summarize_fidelity(recs)
    summ.update({"scored": len(recs), "n": n, "temperature": temperature,
                 "refused": sum(1 for r in recs if r.get("refused"))})
    return summ


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/candidate_two_stage.json")
    ap.add_argument("--resized-model", default="models/base_resized")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--limit", type=int, default=32, help="held-out rows; 0 = full holdout")
    ap.add_argument("--n", type=int, default=16, help="samples per row to rerank")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args(argv)

    cfg = _load_config(args.config)
    from sft.loader import load_eval_model
    model, processor = load_eval_model(cfg, args.resized_model, args.adapter)
    summ = evaluate(model, processor, cfg, n=args.n, temperature=args.temperature, top_p=args.top_p,
                    limit=args.limit, chunk=args.chunk)
    print(json.dumps({"best_of_n_summary": summ}))
    print(f"[bestofn] n={args.n} t={args.temperature} fidelity={summ.get('behavioral_fidelity_mean')} "
          f"collapse_rate={summ.get('collapse_rate')} scored={summ.get('scored')} "
          f"refused={summ.get('refused')}  (greedy baseline 0.159)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
