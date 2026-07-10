"""Oracle upper-bound gate (ADR 0021 §8; ``docs/attribute_spec.md`` §8) — HARD go/no-go for P5/P6.

Tests the semantic-IR **seam**: is a LUT's structured ``behavior_v2`` summary (the best-possible
interpreter output — the *ground-truth* ``measured_behavior`` serialized to ``attribute_spec_text``)
a conditioning signal AT LEAST AS GOOD as the free-text ``instruction`` the one-stage generator was
trained on? If a perfect spec cannot drive the current Generator to the target codes at or above the
one-stage token accuracy on the SAME unit-aware holdout, the summary is lossy (many LUTs collapse to
one spec, AUDIT §9) and the two-stage design is abandoned before any interpreter/generator spend.

Apples-to-apples: BOTH conditions score the SAME current adapter on the SAME P1 unit-aware holdout
(:mod:`sft.holdout`), differing ONLY in the input field:
  * baseline  → ``instruction``            (one-stage)
  * oracle    → ``attribute_spec_text``    (from ground-truth ``measured_behavior``, ADR 0021)

Prints two labelled ``METRIC=`` lines + a single ``{"oracle_gate": {...}}`` JSON line with both token
accuracies, their unit-clustered bootstrap CIs (reused from :mod:`sft.score_tokens`), the delta, and
a PASS/FAIL recommendation (PASS ⇔ oracle point accuracy ≥ baseline). Runs on the Colab A100.

Usage (on Colab, after staging + vocab-resize, with the current adapter):
    SLM_ARTIFACT_ROOT=/content/slm python -m sft.oracle_gate \
        --resized-model models/base_resized --adapter models/sft_adapters/<current_run>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import yaml

from data_pipeline.attribute_spec import measured_behavior_to_text
from data_pipeline.errors import SFTError
from sft.config import SFTConfig
from sft.score_tokens import score

_DEFAULT_CFG_PATH = Path("configs/sft_default.yaml")


def _load_config(path: str | None) -> SFTConfig:
    p = Path(path) if path else _DEFAULT_CFG_PATH
    overrides = yaml.safe_load(p.read_text(encoding="utf-8")) or {} if p.exists() else {}
    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in overrides.items() if k in fields}
    return SFTConfig(**kw)


def _stamp_attribute_spec_text(row: dict) -> None:
    """Ground-truth path: serialize the row's measured_behavior (behavior_v2) to attribute_spec_text."""
    mb = row.get("measured_behavior") or {}
    row["attribute_spec_text"] = measured_behavior_to_text(mb, route="grade")


def run(cfg: SFTConfig, resized_model: str, adapter: str, limit: int) -> dict:
    # Baseline: the one-stage generator conditioned on the free-text instruction.
    base = score(cfg, resized_model, adapter, limit, input_field="instruction")
    # Oracle: the SAME adapter conditioned on the ground-truth attribute_spec_text.
    oracle = score(cfg, resized_model, adapter, limit, input_field="attribute_spec_text",
                   prep_row=_stamp_attribute_spec_text)

    b_acc, o_acc = base["metric"], oracle["metric"]
    delta = o_acc - b_acc
    # PASS ⇔ the ground-truth spec is at least as good a conditioner as the free-text instruction.
    recommend = "PASS" if o_acc >= b_acc else "FAIL"
    return {
        "recommendation": recommend,
        "baseline_token_accuracy": b_acc,
        "oracle_token_accuracy": o_acc,
        "delta": delta,
        "baseline_ci": [base.get("overall_ci_low"), base.get("overall_ci_high")],
        "oracle_ci": [oracle.get("overall_ci_low"), oracle.get("overall_ci_high")],
        "scored_rows": base.get("scored_rows"),
        "scored_units": base.get("scored_units"),
        "baseline_summary": base,
        "oracle_summary": oracle,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(_DEFAULT_CFG_PATH))
    ap.add_argument("--resized-model", default="models/base_resized")
    ap.add_argument("--adapter", required=True, help="the CURRENT one-stage adapter dir")
    ap.add_argument("--limit", type=int, default=0, help="0 = full unit-aware holdout (honest default)")
    args = ap.parse_args(argv)
    cfg = _load_config(args.config)
    try:
        rep = run(cfg, args.resized_model, args.adapter, args.limit)
    except SFTError as exc:
        print(json.dumps({"oracle_gate": {"error": str(exc)}}))
        print(f"[oracle][ABORT] {exc}")
        return 1
    print(json.dumps({"oracle_gate": rep}))
    print(f"METRIC_baseline={rep['baseline_token_accuracy']:.6f}")
    print(f"METRIC_oracle={rep['oracle_token_accuracy']:.6f}")
    print(f"[oracle] recommendation={rep['recommendation']} "
          f"baseline={rep['baseline_token_accuracy']:.4f} oracle={rep['oracle_token_accuracy']:.4f} "
          f"delta={rep['delta']:+.4f} (PASS ⇔ oracle ≥ baseline)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
