"""Score the interpreter on the leakage-safe holdout: route accuracy + per-axis attribute F1.

Emits the single ``METRIC=`` sentinel (a unit-clustered joint score, mirroring ``sft.score_tokens``)
plus a ``score_summary`` whose columns map 1:1 to the ``eval_interpreter`` registry entries
(route_accuracy / per-route recall / refuse_kind_accuracy / interpreter_over_refusal_rate /
attribute_f1). The headline is aggregated per ``split_unit_id`` (equal weight per LUT), since the
number of surviving captions per LUT varies.

    python -m interpreter.score --config configs/interpreter_default.yaml --adapter models/interpreter/interp_r1_smokefull
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Optional

from eval.refuse_taxonomy import ROUTE_CLARIFY, ROUTE_GRADE, ROUTE_REFUSE
from eval.stats import wilson_ci
from interpreter.comparator import compare_specs, joint_score
from interpreter.config import InterpreterConfig, load_config
from interpreter.corpus import load_interpreter_rows, split_train_holdout
from sft.score_tokens import _group_bootstrap_ratio


def _is_procedural(source_lut_id: Optional[str]) -> bool:
    return bool(source_lut_id) and str(source_lut_id).startswith("proc")


def _f1_macro(records: list[dict], predicate) -> Optional[dict]:
    grade = [r for r in records if r["route_gold"] == ROUTE_GRADE and predicate(r)]
    if not grade:
        return None
    # route miss on a grade gold contributes 0 to attribute F1 (it never produced usable axes).
    vals = [(r["attribute_f1"] if (r["route_correct"] and r["attribute_f1"] is not None) else 0.0)
            for r in grade]
    return {"mean": sum(vals) / len(vals), "n": len(grade)}


def summarize_interpreter(records: list[dict]) -> dict:
    """Pure aggregation of per-row comparison records into the score_summary + headline METRIC."""
    n = len(records)
    if n == 0:
        return {"n": 0, "metric": None}

    # Headline: equal-weight-per-unit joint, with a unit-clustered bootstrap CI.
    unit_joints: dict[str, list[float]] = defaultdict(list)
    for r in records:
        unit_joints[r["split_unit_id"]].append(r["joint"])
    units = list(unit_joints)
    unit_means = [sum(v) / len(v) for v in unit_joints.values()]
    metric, ci_lo, ci_hi = _group_bootstrap_ratio(units, unit_means, [1.0] * len(units))

    # route accuracy (3-way), per-row Wilson CI.
    route_k = sum(1 for r in records if r["route_correct"])
    route = wilson_ci(route_k, n)

    per_route_recall = {}
    for rt in (ROUTE_GRADE, ROUTE_CLARIFY, ROUTE_REFUSE):
        gold_rt = [r for r in records if r["route_gold"] == rt]
        if gold_rt:
            k = sum(1 for r in gold_rt if r["route_correct"])
            w = wilson_ci(k, len(gold_rt))
            per_route_recall[rt] = {"recall": w.point, "low": w.low, "n": len(gold_rt)}

    # refuse_kind accuracy over gold-refuse rows (route + kind both correct).
    gold_refuse = [r for r in records if r["route_gold"] == ROUTE_REFUSE]
    refuse_kind = None
    if gold_refuse:
        k = sum(1 for r in gold_refuse if r["route_correct"] and r["refuse_kind_correct"])
        refuse_kind = {"accuracy": k / len(gold_refuse), "n": len(gold_refuse)}

    # over-refusal: gold grade routed to non-grade. Wilson UPPER bound (we bound the false-refuse rate).
    gold_grade = [r for r in records if r["route_gold"] == ROUTE_GRADE]
    over_ref = None
    if gold_grade:
        k = sum(1 for r in gold_grade if r["route_pred"] != ROUTE_GRADE)
        w = wilson_ci(k, len(gold_grade))
        over_ref = {"rate": w.point, "high": w.high, "n": len(gold_grade)}

    return {
        "n": n, "n_units": len(units), "metric": metric,
        "metric_ci": [ci_lo, ci_hi],
        "route_accuracy": {"point": route.point, "low": route.low, "high": route.high},
        "per_route_recall": per_route_recall,
        "refuse_kind_accuracy": refuse_kind,
        "interpreter_over_refusal_rate": over_ref,
        "attribute_f1": {
            "overall": _f1_macro(records, lambda r: True),
            "real_lut": _f1_macro(records, lambda r: not _is_procedural(r.get("source_lut_id"))),
            "procedural": _f1_macro(records, lambda r: _is_procedural(r.get("source_lut_id"))),
        },
        "attribute_f1_by_style": {
            style: _f1_macro(records, lambda r, s=style: r.get("style") == s)
            for style in sorted({r.get("style") for r in records if r.get("style")})
        },
        "parse_ok_rate": sum(1 for r in records if r["parse_ok"]) / n,
    }


def _record(row: dict, pred_text: str) -> dict:
    cmp = compare_specs(pred_text, row["attribute_spec_text"])
    return {**cmp, "joint": joint_score(cmp), "split_unit_id": row["split_unit_id"],
            "source_lut_id": row.get("source_lut_id"), "style": row.get("style")}


def _load_model(cfg: InterpreterConfig, adapter: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(adapter)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if cfg.tuning_mode == "lora":
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(cfg.base_model_id)
        model = PeftModel.from_pretrained(base, adapter)
    else:
        model = AutoModelForCausalLM.from_pretrained(adapter)
    return model.to(device).eval(), tokenizer, device


def score(cfg: InterpreterConfig, adapter: str, *, limit: int = 0) -> int:
    import torch
    from interpreter.example import build_prompt_ids

    rows = load_interpreter_rows(cfg.corpus_path)
    _train, holdout = split_train_holdout(rows, cfg.holdout_frac)
    if limit:
        holdout = holdout[:limit]
    if not holdout:
        print("[interp][ABORT] 0 holdout rows")
        return 1

    model, tokenizer, device = _load_model(cfg, adapter)
    records = []
    for row in holdout:
        prompt_ids = build_prompt_ids(tokenizer, row["text"])
        with torch.no_grad():
            out = model.generate(torch.tensor([prompt_ids]).to(device),
                                  max_new_tokens=cfg.max_new_tokens, do_sample=False,
                                  eos_token_id=tokenizer.eos_token_id,
                                  pad_token_id=tokenizer.pad_token_id)
        pred_text = tokenizer.decode(out[0][len(prompt_ids):], skip_special_tokens=True)
        records.append(_record(row, pred_text))

    summary = summarize_interpreter(records)
    print(f"[interp] score_summary")
    print(json.dumps({"score_summary": summary}, sort_keys=True))
    print(f"METRIC={summary['metric']}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Score the interpreter on the leakage-safe holdout.")
    ap.add_argument("--config", default=None)
    ap.add_argument("--adapter", required=True, help="trained model/adapter dir (from interpreter.train)")
    ap.add_argument("--limit", type=int, default=0, help="first N holdout rows (0 = all)")
    args = ap.parse_args(argv)
    return score(load_config(args.config), args.adapter, limit=args.limit)


if __name__ == "__main__":
    raise SystemExit(main())
