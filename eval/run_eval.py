"""Eval harness orchestration (Stage 1 spine).

Runs the frozen rows through decoder-free baselines across seeds and both decode modes
(free-generation, runtime-constrained), scores L0 boundary + L1 syntax + the full
unsupported/boundary metric suite, attaches ``not_evaluated: decoder_disabled`` results
for L2-L8, computes CIs / paired deltas / gate status, and writes the
``eval_runs/{run_id}/`` report set.

Usage:
    python -m eval.run_eval --config eval/configs/eval_default.yaml \
        --rows data/eval/smoke_rows.jsonl \
        --mock-outputs data/eval/mock_outputs.jsonl --out eval_runs
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

import numpy as np
import yaml

from . import baseline_adapters as ba
from . import deterministic_checks, judge_client, lut_decoder, report, target_fidelity
from .constrained_decoding import LutGrammarFSM
from .output_parsers import parse_output
from .schemas import (
    DECODER_DISABLED_REASON,
    EVAL_CONFIG_VERSION,
    LayerResult,
    STATUS_NOT_EVALUATED,
    build_version_manifest,
    load_rows,
)
from .stats import (
    GateResult,
    NOT_EVALUABLE,
    evaluable,
    mcnemar,
    paired_delta_bootstrap,
    seed_summary,
    wilson_ci,
    wilson_gate,
)
from .unsupported_metrics import DecisionRecord, compute_unsupported_metrics
from .vocab import DEFAULT_VOCAB

FREE = ba.FREE_GENERATION
CONSTRAINED = ba.RUNTIME_CONSTRAINED

# Metric -> (Wilson bound, threshold) from docs "Pass Criteria".
# boundary_f1 is intentionally absent: an F1 is not a binomial proportion, so it is
# gated via a bootstrap lower bound (bootstrap_f1_lower), matching the gating registry's
# CI_method: paired_bootstrap_95 (the doc's "Wilson lower bound" wording for F1 is a
# known spec inconsistency; the registry method governs).
WILSON_GATES = {
    "free_generation_valid_token_rate": ("lower", 0.85),
    "unsupported_recall": ("lower", 0.80),
    "unsupported_precision": ("lower", 0.80),
    "mixed_unsupported_recall": ("lower", 0.80),
    "near_boundary_pair_accuracy": ("lower", 0.85),
    "over_refusal_rate": ("upper", 0.10),
    "supported_prompt_to_lut_pass_rate": ("lower", 0.60),
}


# --- config ----------------------------------------------------------------------
def load_config(path: Optional[str]) -> dict:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_gating_registry(path: Optional[str]) -> dict[str, int]:
    """Return metric -> min_N (or min_paired_N) from gating_slice_registry.yaml."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        reg = yaml.safe_load(fh) or {}
    out: dict[str, int] = {}
    for entry in reg.get("entries", []):
        metric = entry.get("metric")
        min_n = entry.get("min_N", entry.get("min_paired_N"))
        if metric and min_n is not None and metric not in out:
            out[metric] = int(min_n)
    return out


# --- disabled supported layers ---------------------------------------------------
def evaluate_supported_layers(row, parsed) -> dict[str, LayerResult]:  # noqa: ANN001
    """L2-L8 for a (would-be) supported output. All disabled here.

    For a valid LUT-token output we genuinely attempt the decode to exercise the guard,
    then record the resulting ``not_evaluated`` status; L3-L8 follow.
    """
    results: dict[str, LayerResult] = {}
    if not lut_decoder.is_enabled():
        # decoder disabled: L2 is unconditionally not-evaluated (never a fabricated pass)
        results["L2_decode"] = LayerResult.disabled("L2_decode")
    elif parsed.kind == "lut_tokens":
        try:
            lut_decoder.decode_tokens_to_residual(parsed.token_ids)
            # NOTE: when the decoder is enabled, set this from the real L2 gate (finite
            # 17^3 LUT + valid .cube + canonical-domain/vq-hash match), not merely from
            # the absence of an exception.
            results["L2_decode"] = LayerResult("L2_decode", "pass")
        except lut_decoder.DecoderDisabled:
            results["L2_decode"] = LayerResult.disabled("L2_decode")
        except Exception as exc:  # noqa: BLE001 - degrade cleanly on any decode error
            results["L2_decode"] = LayerResult("L2_decode", "fail", reason=f"decode_error:{exc}")
    else:
        results["L2_decode"] = LayerResult.disabled("L2_decode")
    results["L3_tokenizer_gate"] = LayerResult.disabled("L3_tokenizer_gate")
    results.update(deterministic_checks.run_all(row))
    results["L5_target_fidelity"] = target_fidelity.target_fidelity_check(row)
    results["L8_judge"] = judge_client.score(row, parsed, results, None)
    return results


# --- single (adapter, mode, seed) run --------------------------------------------
def run_single(adapter, rows, mode: str, seed: int) -> dict:
    decisions: list[DecisionRecord] = []
    raw_list: list[dict] = []
    parsed_list: list[dict] = []
    per_row_metrics: list[dict] = []

    n_valid_syntax = 0
    for row in rows:
        out = adapter.predict(row, mode, seed)
        parsed = parse_output(out.text)
        valid_syntax = parsed.kind in ("lut_tokens", "unsupported")
        n_valid_syntax += int(valid_syntax)

        decisions.append(
            DecisionRecord(
                id=row.id,
                is_supported=row.is_supported,
                kind=parsed.kind,
                syntax_pass=parsed.syntax_pass,
                mixed_prompt=bool(row.mixed_prompt),
                boundary_pair_id=row.boundary_pair_id,
                route=getattr(row, "route", None),
                refuse_kind=getattr(row, "refuse_kind", None),
            )
        )
        layers = evaluate_supported_layers(row, parsed)
        raw_list.append({"row_id": row.id, "adapter_id": adapter.id, "seed": seed,
                         "mode": mode, "text": out.text, "provenance": out.provenance})
        parsed_list.append({"row_id": row.id, "adapter_id": adapter.id, "seed": seed,
                            "mode": mode, "kind": parsed.kind, "token_count": parsed.token_count,
                            "syntax_pass": parsed.syntax_pass, "parser_errors": parsed.parser_errors})
        per_row_metrics.append({
            "row_id": row.id, "adapter_id": adapter.id, "seed": seed, "mode": mode,
            "split": row.split, "is_supported": row.is_supported,
            "kind": parsed.kind, "syntax_pass": parsed.syntax_pass,
            "boundary_pass": (parsed.kind == "unsupported") == (not row.is_supported),
            "valid_syntax": valid_syntax,
            "L2_decode": layers["L2_decode"].status,
            "L4_direction": layers["L4_direction"].status,
            "L5_target_fidelity": layers["L5_target_fidelity"].status,
            "L6_safety": layers["L6_safety"].status,
            "L7_style": layers["L7_style"].status,
            "L8_judge": layers["L8_judge"].status,
            "supported_pass_status": STATUS_NOT_EVALUATED,
            "supported_pass_reason": DECODER_DISABLED_REASON,
        })

    n = len(rows)
    result = compute_unsupported_metrics(decisions)
    valid_rate = (n_valid_syntax / n) if n else None
    return {
        "adapter": adapter.id,
        "mode": mode,
        "seed": seed,
        "n": n,
        "valid_token_rate": valid_rate,  # denominator = all rows (pinned decision)
        "metrics": result["metrics"],
        "confusion": result["confusion"],
        "scalars": result["scalars"],
        "decisions": decisions,
        "raw_list": raw_list,
        "parsed_list": parsed_list,
        "per_row_metrics": per_row_metrics,
    }


# --- boundary F1 bootstrap -------------------------------------------------------
def bootstrap_f1_lower(decisions: list[DecisionRecord], B: int = 10_000, seed: int = 0,
                       confidence: float = 0.95) -> tuple[Optional[float], Optional[float]]:
    """Point F1 + lower CI bound of the refusal detector (positive class = unsupported)."""
    n = len(decisions)
    if n == 0:
        return None, None
    is_unsup = np.array([not d.is_supported for d in decisions])
    refused = np.array([d.refused for d in decisions])

    def f1(mask):
        tp = np.sum(is_unsup[mask] & refused[mask])
        fp = np.sum(~is_unsup[mask] & refused[mask])
        fn = np.sum(is_unsup[mask] & ~refused[mask])
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    point = f1(np.ones(n, dtype=bool))
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    for i in range(B):
        pick = rng.integers(0, n, size=n)
        boots[i] = _f1_from_index(is_unsup[pick], refused[pick])
    lo = float(np.percentile(boots, (1 - confidence) / 2 * 100))
    return float(point), lo


def _f1_from_index(is_unsup: np.ndarray, refused: np.ndarray) -> float:
    tp = np.sum(is_unsup & refused)
    fp = np.sum(~is_unsup & refused)
    fn = np.sum(is_unsup & ~refused)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


# --- gate evaluation -------------------------------------------------------------
def evaluate_gates(single: dict, min_n_map: dict[str, int], boot_seed: int) -> list[dict]:
    metrics = single["metrics"]
    rows: list[dict] = []

    def emit(metric, bound, threshold, k, n, status, observed):
        rows.append({
            "model": single["adapter"], "mode": single["mode"], "seed": single["seed"],
            "split": "smoke", "metric": metric, "ship_gate_family": "sft_ship",
            "bound": bound, "threshold": threshold, "observed": observed,
            "N": n, "min_N": min_n_map.get(metric), "status": status,
        })

    # free_generation_valid_token_rate (only meaningful in free-generation mode)
    if single["mode"] == FREE:
        k = int(round((single["valid_token_rate"] or 0) * single["n"]))
        g = wilson_gate(k, single["n"], bound="lower", threshold=0.85,
                        min_n=min_n_map.get("free_generation_valid_token_rate"),
                        name="free_generation_valid_token_rate")
        emit("free_generation_valid_token_rate", "lower", 0.85, k, single["n"], g.status, g.observed)

    for metric in ("unsupported_recall", "unsupported_precision", "over_refusal_rate",
                   "mixed_unsupported_recall", "near_boundary_pair_accuracy"):
        bm = metrics.get(metric)
        if bm is None:
            continue
        bound, thr = WILSON_GATES[metric]
        g = wilson_gate(bm.k, bm.n, bound=bound, threshold=thr,
                        min_n=min_n_map.get(metric), name=metric)
        emit(metric, bound, thr, bm.k, bm.n, g.status, g.observed)

    # boundary_f1 via bootstrap lower bound
    point, lo = bootstrap_f1_lower(single["decisions"], seed=boot_seed)
    n_pairs_pop = single["scalars"]["n_gold_unsupported"] + single["scalars"]["n_gold_supported"]
    min_n = min_n_map.get("boundary_f1")
    status = NOT_EVALUABLE if not evaluable(n_pairs_pop, min_n) else (
        "pass" if (lo is not None and lo >= 0.80) else "fail")
    emit("boundary_f1", "lower", 0.80, None, n_pairs_pop, status, lo)

    # supported pass rate: not evaluable without the decoder
    emit("supported_prompt_to_lut_pass_rate", "lower", 0.60, None, single["n"],
         f"{STATUS_NOT_EVALUATED}:{DECODER_DISABLED_REASON}", None)
    return rows


# --- orchestration ---------------------------------------------------------------
def run(
    config_path: Optional[str],
    rows_path: str,
    out_root: str,
    mock_outputs_path: Optional[str] = None,
    seeds: Optional[list[int]] = None,
    modes: Optional[list[str]] = None,
    run_id: Optional[str] = None,
) -> str:
    config = load_config(config_path)
    seeds = seeds or config.get("seeds", [1234])
    modes = modes or config.get("modes", [FREE, CONSTRAINED])
    run_id = run_id or f"run_{int(time.time())}"

    rows = load_rows(rows_path)
    vocab = DEFAULT_VOCAB

    gating_path = _resolve(config.get("gating_slice_registry"),
                           "eval/configs/gating_slice_registry.yaml")
    min_n_map = load_gating_registry(gating_path)

    adapters = list(ba.default_decoder_free_adapters())
    if mock_outputs_path and os.path.exists(mock_outputs_path):
        adapters.append(ba.MockReplayAdapter.from_jsonl(mock_outputs_path, vocab=vocab))
    primary_id = "mock_replay" if any(a.id == "mock_replay" for a in adapters) \
        else "null_always_support_fixed_tokens"

    run_dir = report.ensure_run_dir(out_root, run_id)

    overall_rows: list[dict] = []
    unsupported_rows: list[dict] = []
    gate_rows: list[dict] = []
    all_raw: list[dict] = []
    all_parsed: list[dict] = []
    all_row_metrics: list[dict] = []
    baseline_delta_rows: list[dict] = []
    seed_summary_rows: list[dict] = []

    # cache per (adapter, mode, seed) -> single result, for baseline deltas + seeds
    singles: dict[tuple[str, str, int], dict] = {}

    for adapter in adapters:
        for mode in modes:
            for seed in seeds:
                single = run_single(adapter, rows, mode, seed)
                singles[(adapter.id, mode, seed)] = single
                all_raw.extend(single["raw_list"])
                all_parsed.extend(single["parsed_list"])
                all_row_metrics.extend(single["per_row_metrics"])

                overall_rows.append(_overall_row(single, mode))
                unsupported_rows.extend(_unsupported_rows(single, rows))

    # ship gates evaluate the primary model only (free-generation for the syntax gate;
    # boundary decisions are mode-independent). Smoke N is far below min_N by design.
    primary_free = singles.get((primary_id, FREE if FREE in modes else modes[0], seeds[0]))
    if primary_free is not None:
        gate_rows = evaluate_gates(primary_free, min_n_map, boot_seed=seeds[0])

    # baseline deltas: primary vs each null/constant baseline, evaluable metrics only
    baseline_delta_rows = _baseline_deltas(singles, primary_id, seeds, modes, adapters)

    # seed summaries for the primary adapter (free-generation mode)
    seed_summary_rows = _seed_summaries(singles, primary_id, seeds)

    # --- write artifacts ---
    report.write_csv(os.path.join(run_dir, "overall_results.csv"), overall_rows, report.OVERALL_COLUMNS)
    report.write_csv(os.path.join(run_dir, "unsupported_results.csv"), unsupported_rows, report.UNSUPPORTED_COLUMNS)
    report.write_csv(os.path.join(run_dir, "gate_results.csv"), gate_rows, report.GATE_COLUMNS)
    report.write_csv(os.path.join(run_dir, "baseline_delta.csv"), baseline_delta_rows, report.BASELINE_DELTA_COLUMNS)
    report.write_csv(os.path.join(run_dir, "seed_summary.csv"), seed_summary_rows, report.SEED_SUMMARY_COLUMNS)
    _write_disabled_tables(run_dir)

    report.write_jsonl(os.path.join(run_dir, "rows.jsonl"), (r.to_dict() for r in rows))
    report.write_jsonl(os.path.join(run_dir, "raw_model_outputs.jsonl"), all_raw)
    report.write_jsonl(os.path.join(run_dir, "parsed_outputs.jsonl"), all_parsed)
    report.write_metrics_by_row(os.path.join(run_dir, "metrics_by_row.parquet"), all_row_metrics)
    _write_failure_manifest(run_dir, singles, primary_id, modes, seeds)

    manifest = build_version_manifest(
        vocab_added_special_token_ids=vocab.added_special_token_ids,
        vocab_size_after_resize=None,
    )
    report.write_config(os.path.join(run_dir, "config.yaml"), {
        "eval_config_version": EVAL_CONFIG_VERSION,
        "run_id": run_id,
        "rows_path": rows_path,
        "mock_outputs_path": mock_outputs_path,
        "seeds": seeds,
        "modes": modes,
        "adapters_run": [a.id for a in adapters],
        "primary_adapter": primary_id,
        "gating_slice_registry": gating_path,
        "calibration_manifest": _resolve(config.get("calibration_manifest"),
                                          "eval/configs/calibration_manifest.json"),
        "renderer_baseline": _resolve(config.get("renderer_baseline"),
                                      "configs/renderer_baseline.yaml"),
        "model_clients": config.get("model_clients", "configs/model_clients.yaml"),
        "decoder_enabled": lut_decoder.is_enabled(),
        "note": "Stage 1 spine: decoder disabled; L2-L8 not_evaluated; smoke is non-gating.",
        "version_manifest": manifest,
    })

    # console summary
    _print_summary(run_dir, singles, primary_id, modes, seeds)
    return run_dir


# --- row builders ----------------------------------------------------------------
def _rate(bm) -> Optional[float]:  # noqa: ANN001
    return bm.rate if bm is not None else None


def _overall_row(single: dict, mode: str) -> dict:
    m = single["metrics"]
    return {
        "model": single["adapter"], "checkpoint_id": "", "seed": single["seed"],
        "mode": mode, "split": "smoke", "N": single["n"],
        "supported_pass_n": "", "supported_pass_rate": "",
        "supported_pass_ci_low": "", "supported_pass_ci_high": "",
        "supported_pass_status": f"{STATUS_NOT_EVALUATED}:{DECODER_DISABLED_REASON}",
        "free_generation_valid_token_rate": single["valid_token_rate"] if mode == FREE else "",
        "constrained_syntax_valid_rate": single["valid_token_rate"] if mode == CONSTRAINED else "",
        "decode_valid_rate": DECODER_DISABLED_REASON,
        "target_fidelity_pass": DECODER_DISABLED_REASON,
        "safety_fail": DECODER_DISABLED_REASON,
        "judge_means": "",
        "boundary_accuracy": _rate(m.get("boundary_accuracy")),
        "over_refusal_rate": _rate(m.get("over_refusal_rate")),
        "unsupported_recall": single["scalars"]["unsupported_recall"],
        "unsupported_precision": single["scalars"]["unsupported_precision"],
        "boundary_f1": single["scalars"]["boundary_f1"],
        "mixed_unsupported_recall": _rate(m.get("mixed_unsupported_recall")),
        "near_boundary_pair_accuracy": _rate(m.get("near_boundary_pair_accuracy")),
    }


def _unsupported_rows(single: dict, rows) -> list[dict]:  # noqa: ANN001
    m = single["metrics"]
    base = {
        "model": single["adapter"], "mode": single["mode"], "seed": single["seed"],
        "category": "ALL",
        "N": single["scalars"]["n_gold_unsupported"],
        "recall": single["scalars"]["unsupported_recall"],
        "precision": single["scalars"]["unsupported_precision"],
        "false_support": _rate(m.get("false_support_rate")),
        "over_refusal": _rate(m.get("over_refusal_rate")),
        "coverage": _rate(m.get("supported_coverage")),
        "boundary_f1": single["scalars"]["boundary_f1"],
        "mixed_recall": _rate(m.get("mixed_unsupported_recall")),
    }
    out = [base]
    # per-category refusal recall (diagnostic; typically N<100)
    by_cat: dict[str, list[bool]] = {}
    id_to_row = {r.id: r for r in rows}
    for uid, passed in m["unsupported_recall"].as_pairs():
        cat = (id_to_row[uid].unsupported_category or "uncategorized")
        by_cat.setdefault(cat, []).append(passed)
    for cat, vals in sorted(by_cat.items()):
        out.append({
            "model": single["adapter"], "mode": single["mode"], "seed": single["seed"],
            "category": cat, "N": len(vals),
            "recall": (sum(vals) / len(vals)) if vals else None,
            "precision": "", "false_support": "", "over_refusal": "",
            "coverage": "", "boundary_f1": "", "mixed_recall": "",
        })
    return out


def _baseline_deltas(singles, primary_id, seeds, modes, adapters) -> list[dict]:  # noqa: ANN001
    rows: list[dict] = []
    seed = seeds[0]
    mode = ba.FREE_GENERATION if ba.FREE_GENERATION in modes else modes[0]
    prim = singles.get((primary_id, mode, seed))
    if prim is None:
        return rows
    baselines = [a.id for a in adapters if a.id != primary_id and not getattr(a, "diagnostic", False)]
    for base_id in baselines:
        base = singles.get((base_id, mode, seed))
        if base is None:
            continue
        # evaluable paired metric: boundary_accuracy over all rows
        pa = {uid: p for uid, p in prim["metrics"]["boundary_accuracy"].as_pairs()}
        pb = {uid: p for uid, p in base["metrics"]["boundary_accuracy"].as_pairs()}
        shared = [uid for uid in pa if uid in pb]
        a = [1.0 if pa[uid] else 0.0 for uid in shared]
        b = [1.0 if pb[uid] else 0.0 for uid in shared]
        pd = paired_delta_bootstrap(a, b, seed=seed)
        mc = mcnemar(a, b)
        rows.append({
            "model_pair": f"{primary_id}__vs__{base_id}", "seed_policy": f"seed={seed}",
            "metric": "boundary_accuracy", "N_paired": pd.n,
            "delta_pp": None if pd.delta is None else round(pd.delta * 100, 3),
            "paired_boot_ci_low_pp": None if pd.ci_low is None else round(pd.ci_low * 100, 3),
            "paired_boot_ci_high_pp": None if pd.ci_high is None else round(pd.ci_high * 100, 3),
            "paired_test_p": round(mc.p_value, 5),
            "gate_threshold": "n/a (smoke, non-gating)", "gate_result": "diagnostic",
        })
        # supported pass-rate delta: not evaluable without decoder
        rows.append({
            "model_pair": f"{primary_id}__vs__{base_id}", "seed_policy": f"seed={seed}",
            "metric": "supported_prompt_to_lut_pass_rate", "N_paired": prim["n"],
            "delta_pp": "", "paired_boot_ci_low_pp": "", "paired_boot_ci_high_pp": "",
            "paired_test_p": "", "gate_threshold": "+30pp/+20pp/+5pp",
            "gate_result": f"{STATUS_NOT_EVALUATED}:{DECODER_DISABLED_REASON}",
        })
    return rows


def _seed_summaries(singles, primary_id, seeds) -> list[dict]:  # noqa: ANN001
    rows: list[dict] = []
    mode = ba.FREE_GENERATION
    for metric in ("free_generation_valid_token_rate", "boundary_accuracy",
                   "unsupported_recall", "boundary_f1"):
        vals = []
        for seed in seeds:
            s = singles.get((primary_id, mode, seed))
            if s is None:
                continue
            if metric == "free_generation_valid_token_rate":
                vals.append(s["valid_token_rate"])
            elif metric in s["scalars"]:
                vals.append(s["scalars"][metric])
            elif metric in s["metrics"]:
                vals.append(s["metrics"][metric].rate)
        ss = seed_summary(metric, [v for v in vals if v is not None])
        rows.append({
            "model_stage": primary_id, "seed_count": ss.seed_count, "metric": metric,
            "mean": ss.mean, "std": ss.std, "min": ss.min, "median": ss.median,
            "max": ss.max, "seed_mean_ci_low": ss.seed_mean_ci_low,
            "seed_mean_ci_high": ss.seed_mean_ci_high,
        })
    return rows


def _write_failure_manifest(run_dir, singles, primary_id, modes, seeds) -> None:  # noqa: ANN001
    mode = modes[0]
    seed = seeds[0]
    s = singles.get((primary_id, mode, seed))
    recs: list[dict] = []
    if s is not None:
        for d in s["decisions"]:
            correct = d.refused == (not d.is_supported)
            if not correct or d.kind == "invalid":
                layer = "L1_syntax" if d.kind == "invalid" else "L0_boundary"
                recs.append({
                    "row_id": d.id, "adapter": primary_id, "mode": mode, "seed": seed,
                    "failure_layer": layer, "is_supported": d.is_supported,
                    "kind": d.kind, "likely_cause": "boundary/syntax (spine-evaluable layers only)",
                })
    report.write_jsonl(os.path.join(run_dir, "failure_manifest.jsonl"), recs)


def _write_disabled_tables(run_dir: str) -> None:
    note = [{"status": f"{STATUS_NOT_EVALUATED}:{DECODER_DISABLED_REASON}",
             "detail": "requires frozen VQ tokenizer (training Stages 7-8)"}]
    for name in ("target_fidelity_results.csv", "safety_results.csv",
                 "style_results.csv", "attribute_results.csv"):
        report.write_csv(os.path.join(run_dir, name), note, ["status", "detail"])


def _resolve(value: Optional[str], default: str) -> str:
    return value if value else default


def _print_summary(run_dir, singles, primary_id, modes, seeds) -> None:  # noqa: ANN001
    print(f"[run_eval] wrote {run_dir}")
    for mode in modes:
        s = singles.get((primary_id, mode, seeds[0]))
        if s is None:
            continue
        rate = s["valid_token_rate"]
        label = "constrained_syntax_valid_rate" if mode == CONSTRAINED else "free_gen_valid_token_rate"
        print(f"  [{primary_id}/{mode}] {label}={rate}  "
              f"boundary_acc={s['metrics']['boundary_accuracy'].rate}  "
              f"unsup_recall={s['scalars']['unsupported_recall']}")


# --- CLI -------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run the prompt-to-LUT eval harness (Stage 1 spine).")
    ap.add_argument("--config", default="eval/configs/eval_default.yaml")
    ap.add_argument("--rows", required=True)
    ap.add_argument("--mock-outputs", default=None)
    ap.add_argument("--out", default="eval_runs")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--seeds", default=None, help="comma-separated seeds (overrides config)")
    ap.add_argument("--modes", default=None, help="comma-separated modes (overrides config)")
    args = ap.parse_args(argv)

    seeds = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    modes = args.modes.split(",") if args.modes else None
    run(args.config, args.rows, args.out, mock_outputs_path=args.mock_outputs,
        seeds=seeds, modes=modes, run_id=args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
