"""Decoder-free analysis of the semantic-IR seam (ADR 0021 §8; AUDIT §9).

The oracle gate scores the seam THROUGH a fixed generator, so it conflates two questions and is
confounded when the generator was trained on a different input format. This isolates the
information-theoretic half — **does ``attribute_spec_text`` retain enough information to identify the
target LUT codes?** (the audit's "many LUTs share a summary" concern) — with NO generator and NO
decoder.

For the supported, materialized rows (each carries a ``behavior_v2`` ``measured_behavior`` + 64
``target_tokens``) it serializes each row to ``attribute_spec_text``, groups rows by that string, and
reports:

  * **collision rate** — the fraction of rows whose spec is shared by ANOTHER row with a DIFFERENT
    64-code target (a genuinely lossy collision: the spec cannot distinguish them);
  * **token-accuracy upper bound** — the ceiling a perfect spec→codes mapper could reach: within each
    spec group, predict the per-position majority code; average the per-position hit rate over all
    rows. Unique specs ⇒ 100%. This upper-bounds any generator conditioned on the spec.

It runs this for the full behavior_v2 spec AND for a 2-axis "behavior_v1-style" spec (temperature +
tint only), so the resolution gain from ADR 0022 is quantified, and separately for the P1 unit-aware
holdout (apples-to-apples with the oracle gate's 120 rows).

Pure/local (numpy only) — no GPU, no teacher, no decoder.

Usage:
    python -m scripts.analyze_seam_injectivity
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from data_pipeline.attribute_spec import AttributeSpec, serialize
from data_pipeline.attribute_spec import from_measured_behavior
from sft.holdout import is_holdout_row

_ACTIVE_ROWS = "data/active_sft/active_rows.jsonl"


def _v2_spec_text(mb: dict) -> str:
    return serialize(from_measured_behavior(mb))


def _v1_spec_text(mb: dict) -> str:
    """A behavior_v1-style 2-axis spec (temperature + tint only) — the pre-ADR-0022 resolution."""
    axes = {}
    for fld in ("temperature_delta_b", "tint_delta_a"):
        v = round(float(mb.get(fld, 0.0) or 0.0), 1)
        if abs(v) >= 0.5:
            axes[fld] = v
    return serialize(AttributeSpec(axes=axes))


def analyze(rows: list[dict], spec_fn) -> dict:
    """Collision rate + token-accuracy upper bound for a spec serialization over ``rows``."""
    groups: dict[str, list[tuple]] = defaultdict(list)
    for r in rows:
        spec = spec_fn(r["measured_behavior"])
        groups[spec].append(tuple(r["target_tokens"]))

    n = len(rows)
    n_specs = len(groups)
    code_seqs = {tuple(r["target_tokens"]) for r in rows}
    # lossy collisions: a spec bucket holding >1 DISTINCT code sequence
    lossy_rows = sum(len(seqs) for seqs in groups.values() if len(set(seqs)) > 1)

    total_pos = correct_pos = 0
    for seqs in groups.values():
        arr = np.array(seqs)                      # [g, 64]
        for pos in range(arr.shape[1]):
            col = arr[:, pos]
            _vals, counts = np.unique(col, return_counts=True)
            correct_pos += int(counts.max())      # majority code is the best a spec-mapper can do
            total_pos += col.size
    return {
        "rows": n,
        "unique_specs": n_specs,
        "unique_code_sequences": len(code_seqs),
        "spec_uniqueness": n_specs / n if n else 0.0,
        "lossy_collision_rows": lossy_rows,
        "lossy_collision_rate": lossy_rows / n if n else 0.0,
        "token_accuracy_upper_bound": correct_pos / total_pos if total_pos else 0.0,
    }


def _materialized_supported(path: str) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    out = []
    for r in rows:
        if (r.get("is_supported") and isinstance(r.get("target_tokens"), list)
                and len(r["target_tokens"]) == 64 and r.get("measured_behavior")):
            out.append(r)
    return out


def run(active_rows: str) -> dict:
    rows = _materialized_supported(active_rows)
    holdout = [r for r in rows if is_holdout_row(r)]
    report = {
        "full_corpus": {
            "behavior_v2": analyze(rows, _v2_spec_text),
            "behavior_v1_2axis": analyze(rows, _v1_spec_text),
        },
        "p1_holdout": {
            "behavior_v2": analyze(holdout, _v2_spec_text),
            "behavior_v1_2axis": analyze(holdout, _v1_spec_text),
        },
    }

    def _fmt(tag, d):
        print(f"  [{tag}] rows={d['rows']} unique_specs={d['unique_specs']} "
              f"({d['spec_uniqueness']:.1%}) lossy_collision_rate={d['lossy_collision_rate']:.3f} "
              f"token_acc_upper_bound={d['token_accuracy_upper_bound']:.4f}")

    print("=== Seam injectivity (decoder-free; can a perfect spec-mapper hit the codes?) ===")
    print("FULL CORPUS:")
    _fmt("behavior_v2   ", report["full_corpus"]["behavior_v2"])
    _fmt("behavior_v1 2ax", report["full_corpus"]["behavior_v1_2axis"])
    print("P1 UNIT-AWARE HOLDOUT (apples-to-apples with the oracle gate):")
    _fmt("behavior_v2   ", report["p1_holdout"]["behavior_v2"])
    _fmt("behavior_v1 2ax", report["p1_holdout"]["behavior_v1_2axis"])
    print(json.dumps({"seam_injectivity": report}))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--active-rows", default=_ACTIVE_ROWS)
    args = ap.parse_args(argv)
    run(args.active_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
