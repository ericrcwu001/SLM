"""Report writers for ``eval_runs/{run_id}/`` (docs/eval_harness_implementation.md
"Reports"). This build populates every table it can from the L0/L1 + boundary +
stats stack; decode-dependent tables (target fidelity, safety, style) are written with
a ``not_evaluated: decoder_disabled`` status row rather than a fabricated pass.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Iterable, Optional

# Column contracts (documentation + stable ordering) from the "Reports" tables.
OVERALL_COLUMNS = [
    "model", "checkpoint_id", "seed", "mode", "split", "N",
    "supported_pass_n", "supported_pass_rate", "supported_pass_ci_low",
    "supported_pass_ci_high", "supported_pass_status",
    "free_generation_valid_token_rate", "constrained_syntax_valid_rate",
    "decode_valid_rate", "target_fidelity_pass", "safety_fail", "judge_means",
    "boundary_accuracy", "over_refusal_rate", "unsupported_recall",
    "unsupported_precision", "boundary_f1", "mixed_unsupported_recall",
    "near_boundary_pair_accuracy",
]
UNSUPPORTED_COLUMNS = [
    "model", "mode", "seed", "category", "N", "recall", "precision",
    "false_support", "over_refusal", "coverage", "boundary_f1", "mixed_recall",
]
BASELINE_DELTA_COLUMNS = [
    "model_pair", "seed_policy", "metric", "N_paired", "delta_pp",
    "paired_boot_ci_low_pp", "paired_boot_ci_high_pp", "paired_test_p",
    "gate_threshold", "gate_result",
]
SEED_SUMMARY_COLUMNS = [
    "model_stage", "seed_count", "metric", "mean", "std", "min", "median", "max",
    "seed_mean_ci_low", "seed_mean_ci_high",
]
GATE_COLUMNS = [
    "model", "mode", "seed", "split", "metric", "ship_gate_family", "bound",
    "threshold", "observed", "N", "min_N", "status",
]


def ensure_run_dir(out_root: str, run_id: str) -> str:
    path = os.path.join(out_root, run_id)
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "qualitative"), exist_ok=True)
    return path


def write_jsonl(path: str, records: Iterable[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True, default=_json_default) + "\n")


def write_csv(path: str, rows: list[dict], columns: Optional[list[str]] = None) -> None:
    if columns is None:
        columns = sorted({k for r in rows for k in r.keys()}) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow({c: _csv_cell(r.get(c)) for c in columns})


def write_metrics_by_row(path_parquet: str, rows: list[dict]) -> str:
    """Write per-row metrics. Prefers ``.parquet`` (pyarrow); falls back to ``.csv``."""
    try:
        import pandas as pd  # noqa: WPS433

        df = pd.DataFrame(rows)
        df.to_parquet(path_parquet, index=False)
        return path_parquet
    except Exception:
        csv_path = os.path.splitext(path_parquet)[0] + ".csv"
        write_csv(csv_path, rows)
        return csv_path


def write_config(path: str, config: dict) -> None:
    try:
        import yaml  # noqa: WPS433

        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(config, fh, sort_keys=True, default_flow_style=False)
    except Exception:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(config, fh, indent=2, sort_keys=True, default=_json_default)


def _json_default(o: Any):
    try:
        import numpy as np

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def _csv_cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (dict, list)):
        return json.dumps(v, sort_keys=True, default=_json_default)
    return v
