"""Score cached prompted-frontier raw-.cube outputs — the frontier LUT-quality baseline.

Replays ``data/eval/frontier_<name>.jsonl`` (produced by scripts.generate_frontier_luts),
parses each raw ``.cube``, and scores:

  * **L0 boundary** (reuses eval.unsupported_metrics): does the model correctly refuse
    unsupported prompts vs. attempt supported ones? — unsupported recall/precision,
    over-refusal, boundary accuracy/F1.
  * **valid-.cube rate**: how often a raw 17^3 .cube even parses (a first-class result —
    frontier models are not expected to reliably emit 4913 exact rows).
  * **L4 direction + L6 safety** (eval.frontier_scoring): on supported rows that produced
    a valid LUT, does it move the image the way the gold tags ask, safely?
  * **Headline LUT-quality pass rate**: valid .cube AND direction-correct AND safe, over
    supported rows that carry a directional gold tag.

L5 target fidelity stays ``not_evaluated`` — the frozen eval rows carry no target LUTs.

Usage:
    python -m eval.run_frontier_eval --rows data/eval/smoke_rows.jsonl --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

from . import report
from .cube_parser import parse_frontier_cube
from .frontier_scoring import NOT_EVALUATED, score_lut
from .schemas import load_rows
from .stats import wilson_ci
from .unsupported_metrics import DecisionRecord, compute_unsupported_metrics

_KIND_MAP = {"raw_lut": "lut_tokens", "unsupported": "unsupported", "invalid": "invalid"}
_BEHAVIOR_KEYS = ["temperature_delta_b", "tint_delta_a", "mean_l_delta",
                  "contrast_l_spread_delta", "chroma_delta", "shadow_l_delta",
                  "foldover_rate", "clip_rate", "smoothness", "residual_norm"]

OVERALL_COLUMNS = [
    "model", "model_id", "N", "split",
    "raw_cube_valid_rate", "raw_cube_valid_ci_low", "raw_cube_valid_ci_high",
    "unsupported_recall", "unsupported_precision", "over_refusal_rate",
    "boundary_accuracy", "boundary_f1",
    "direction_pass_rate", "direction_N",
    "safety_pass_rate", "safety_N",
    "lut_quality_pass_rate", "lut_quality_ci_low", "lut_quality_ci_high", "lut_quality_N",
    "target_fidelity_status",
]


def _cube_path(out_dir: str, name: str) -> str:
    return os.path.join(out_dir, f"frontier_{name}.jsonl")


def _load_outputs(path: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rid = d.get("row_id")
            if rid is not None:
                out[rid] = d
    return out


def _wilson(k: int, n: int):
    w = wilson_ci(k, n)
    return (round(w.point, 4) if w.point is not None else None,
            round(w.low, 4) if w.low is not None else None,
            round(w.high, 4) if w.high is not None else None)


def _score_model(name: str, rows, outputs: dict[str, dict]) -> tuple[dict, list[dict]]:
    decisions: list[DecisionRecord] = []
    per_row: list[dict] = []
    model_id = ""

    n_valid = 0
    dir_pass = dir_n = 0
    safe_pass = safe_n = 0
    q_pass = q_n = 0

    for row in rows:
        rec = outputs.get(row.id)
        text = rec.get("text") if rec else None
        if rec:
            model_id = rec.get("model_id", model_id)
        parsed = parse_frontier_cube(text)
        n_valid += int(parsed.kind == "raw_lut")

        decisions.append(DecisionRecord(
            id=row.id, is_supported=row.is_supported, kind=_KIND_MAP[parsed.kind],
            syntax_pass=parsed.syntax_pass, mixed_prompt=bool(row.mixed_prompt),
            boundary_pair_id=row.boundary_pair_id,
        ))

        prow = {
            "row_id": row.id, "model": name, "is_supported": row.is_supported,
            "gold_tags": row.gold_tags, "instruction": row.instruction,
            "kind": parsed.kind, "errors": parsed.errors,
            "output_tokens": (rec or {}).get("provenance", {}).get("output_tokens"),
            "api_refusal": (rec or {}).get("provenance", {}).get("api_refusal"),
            "direction_status": None, "safety_status": None, "safety_reasons": None,
            "lut_quality_pass": None, "behavior": None,
        }

        if parsed.kind == "raw_lut":
            score = score_lut(parsed.lut_abs, row.gold_tags)
            prow["safety_status"] = score.safety.status
            prow["safety_reasons"] = score.safety.reasons
            prow["behavior"] = {k: round(float(score.behavior.get(k, 0.0)), 4) for k in _BEHAVIOR_KEYS}
            safe_n += 1
            safe_pass += int(score.safety.status == "pass")
            # direction + headline quality only on supported rows with a directional tag
            if row.is_supported and score.direction.status != NOT_EVALUATED:
                prow["direction_status"] = score.direction.status
                prow["direction_detail"] = score.direction.per_tag
                dir_n += 1
                dir_pass += int(score.direction.status == "pass")
                q_n += 1
                q_pass += int(bool(score.lut_quality_pass))
                prow["lut_quality_pass"] = bool(score.lut_quality_pass)
            elif row.is_supported:
                prow["direction_status"] = NOT_EVALUATED
        per_row.append(prow)

    n = len(rows)
    res = compute_unsupported_metrics(decisions)
    m, sc = res["metrics"], res["scalars"]
    valid_pt, valid_lo, valid_hi = _wilson(n_valid, n)
    q_pt, q_lo, q_hi = _wilson(q_pass, q_n)

    overall = {
        "model": name, "model_id": model_id, "N": n, "split": "smoke",
        "raw_cube_valid_rate": valid_pt, "raw_cube_valid_ci_low": valid_lo, "raw_cube_valid_ci_high": valid_hi,
        "unsupported_recall": _r(sc["unsupported_recall"]), "unsupported_precision": _r(sc["unsupported_precision"]),
        "over_refusal_rate": _r(m["over_refusal_rate"].rate), "boundary_accuracy": _r(m["boundary_accuracy"].rate),
        "boundary_f1": _r(sc["boundary_f1"]),
        "direction_pass_rate": _rate(dir_pass, dir_n), "direction_N": dir_n,
        "safety_pass_rate": _rate(safe_pass, safe_n), "safety_N": safe_n,
        "lut_quality_pass_rate": q_pt, "lut_quality_ci_low": q_lo, "lut_quality_ci_high": q_hi,
        "lut_quality_N": q_n,
        "target_fidelity_status": "not_evaluated:no_target_luts_in_frozen_eval_set",
    }
    return overall, per_row


def _r(x: Optional[float]) -> Optional[float]:
    return round(x, 4) if x is not None else None


def _rate(k: int, n: int) -> Optional[float]:
    return round(k / n, 4) if n else None


def run(rows_path: str, out_root: str, out_dir: str, limit: Optional[int],
        model_names: Optional[list[str]], run_id: Optional[str]) -> str:
    run_id = run_id or f"frontier_{int(time.time())}"
    rows = load_rows(rows_path)
    if limit is not None:
        rows = rows[:limit]

    # discover model files (or use the requested names)
    if model_names:
        names = model_names
    else:
        names = []
        if os.path.isdir(out_dir):
            for fn in sorted(os.listdir(out_dir)):
                if fn.startswith("frontier_") and fn.endswith(".jsonl"):
                    names.append(fn[len("frontier_"):-len(".jsonl")])
    if not names:
        raise SystemExit(f"no frontier_*.jsonl found in {out_dir} — run scripts.generate_frontier_luts first")

    run_dir = report.ensure_run_dir(out_root, run_id)
    overall_rows: list[dict] = []
    all_per_row: list[dict] = []
    for name in names:
        outputs = _load_outputs(_cube_path(out_dir, name))
        if not outputs:
            print(f"[frontier-eval] no cached outputs for '{name}' ({_cube_path(out_dir, name)}); skipping")
            continue
        overall, per_row = _score_model(name, rows, outputs)
        overall_rows.append(overall)
        all_per_row.extend(per_row)

    report.write_csv(os.path.join(run_dir, "frontier_overall.csv"), overall_rows, OVERALL_COLUMNS)
    report.write_jsonl(os.path.join(run_dir, "frontier_per_row.jsonl"), all_per_row)
    report.write_config(os.path.join(run_dir, "config.yaml"), {
        "run_id": run_id, "rows_path": rows_path, "N_rows": len(rows),
        "models": names, "baseline": "prompted_frontier__raw_cube",
        "grid_size": 17, "output_mode": "raw_cube",
        "note": ("Frontier LUT-quality baseline. L0 boundary + valid-.cube rate + L4 "
                 "direction + L6 safety scored; L5 target fidelity not_evaluated "
                 "(no target LUTs in the frozen eval set). Safety/direction thresholds "
                 "are provisional pilot values, not the frozen calibration_manifest."),
    })
    _print_summary(run_dir, overall_rows)
    return run_dir


def _print_summary(run_dir: str, overall_rows: list[dict]) -> None:
    print(f"[frontier-eval] wrote {run_dir}")
    for o in overall_rows:
        print(f"  [{o['model']}] valid_cube={o['raw_cube_valid_rate']} "
              f"boundary_acc={o['boundary_accuracy']} unsup_recall={o['unsupported_recall']} "
              f"direction={o['direction_pass_rate']}(N={o['direction_N']}) "
              f"safety={o['safety_pass_rate']}(N={o['safety_N']}) "
              f"LUT_quality={o['lut_quality_pass_rate']}(N={o['lut_quality_N']})")


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Score cached prompted-frontier raw-.cube outputs.")
    ap.add_argument("--rows", default="data/eval/smoke_rows.jsonl")
    ap.add_argument("--out", default="eval_runs", help="report root")
    ap.add_argument("--cube-dir", default="data/eval", help="dir with frontier_<name>.jsonl")
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--models", default=None, help="comma-separated model names (default: discover files)")
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)
    names = args.models.split(",") if args.models else None
    run(args.rows, args.out, args.cube_dir, args.limit, names, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
