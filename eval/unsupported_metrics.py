"""L0 boundary + unsupported/refusal metrics.

Implements every formula in docs/eval_harness_implementation.md "Unsupported Metrics".
Operates on the refuse / not-refuse decision only, so it needs no LUT decoder.

The boundary decision is: a row is *refused* iff the model output parses to exactly
``<unsupported>`` (``ParsedOutput.kind == "unsupported"``); anything else (a valid
LUT-token sequence OR an invalid output) is *not refused*. "False support" is the
stricter event of emitting a valid LUT sequence on a gold-unsupported row.

Each metric is returned as a :class:`BinaryMetric` carrying the per-unit pass vector so
``stats.py`` can compute Wilson CIs and paired bootstraps. ``boundary_f1`` is derived
from the refusal-detector confusion (positive class = unsupported).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass
class DecisionRecord:
    """One row's gold label + the model's parsed decision."""

    id: str
    is_supported: bool
    kind: str  # "lut_tokens" | "unsupported" | "invalid"
    syntax_pass: bool = False
    mixed_prompt: bool = False
    boundary_pair_id: str | None = None

    @property
    def refused(self) -> bool:
        return self.kind == "unsupported"

    @property
    def emitted_lut(self) -> bool:
        return self.kind == "lut_tokens"


@dataclass
class BinaryMetric:
    """A rate metric over a population of units (rows or pairs) with pass indicators."""

    name: str
    unit_ids: list[str] = field(default_factory=list)
    passed: list[bool] = field(default_factory=list)

    @property
    def k(self) -> int:
        return int(sum(1 for p in self.passed if p))

    @property
    def n(self) -> int:
        return len(self.passed)

    @property
    def rate(self) -> float | None:
        return (self.k / self.n) if self.n else None

    def as_pairs(self) -> list[tuple[str, bool]]:
        return list(zip(self.unit_ids, self.passed))


def _metric(name: str, items: Iterable[tuple[str, bool]]) -> BinaryMetric:
    ids, passed = [], []
    for uid, p in items:
        ids.append(uid)
        passed.append(bool(p))
    return BinaryMetric(name=name, unit_ids=ids, passed=passed)


def compute_unsupported_metrics(records: list[DecisionRecord]) -> dict:
    """Compute all boundary/unsupported metrics.

    Returns a dict with:
      * ``metrics``: name -> BinaryMetric
      * ``confusion``: refusal-detector confusion counts (positive = unsupported)
      * ``scalars``: derived scalars (boundary_f1, unsupported_precision/recall, ...)
    """

    gold_unsup = [r for r in records if not r.is_supported]
    gold_sup = [r for r in records if r.is_supported]
    model_refusals = [r for r in records if r.refused]
    mixed_unsup = [r for r in gold_unsup if r.mixed_prompt]
    sup_non_refusals = [r for r in gold_sup if not r.refused]

    metrics: dict[str, BinaryMetric] = {}

    # Unsupported recall: correct refusals / all gold unsupported
    metrics["unsupported_recall"] = _metric(
        "unsupported_recall", [(r.id, r.refused) for r in gold_unsup]
    )
    # Unsupported precision: correct refusals / all model refusals
    metrics["unsupported_precision"] = _metric(
        "unsupported_precision", [(r.id, (not r.is_supported)) for r in model_refusals]
    )
    # False-support: valid LUT output on gold unsupported / all gold unsupported
    metrics["false_support_rate"] = _metric(
        "false_support_rate", [(r.id, r.emitted_lut) for r in gold_unsup]
    )
    # Over-refusal: refusal on gold supported / all gold supported
    metrics["over_refusal_rate"] = _metric(
        "over_refusal_rate", [(r.id, r.refused) for r in gold_sup]
    )
    # Supported coverage: non-refusal on gold supported / all gold supported
    metrics["supported_coverage"] = _metric(
        "supported_coverage", [(r.id, (not r.refused)) for r in gold_sup]
    )
    # Boundary accuracy: correct refuse/not-refuse decision over all rows
    metrics["boundary_accuracy"] = _metric(
        "boundary_accuracy",
        [(r.id, (r.refused == (not r.is_supported))) for r in records],
    )
    # Mixed unsupported recall
    metrics["mixed_unsupported_recall"] = _metric(
        "mixed_unsupported_recall", [(r.id, r.refused) for r in mixed_unsup]
    )
    # Selective risk: deterministic failures / supported non-refusals. The full spec
    # numerator is L1-L7; only L1 (syntax) is evaluable while the decoder is disabled,
    # so this is reported under an explicit *_syntax_only key to avoid being read as the
    # full selective-risk metric. Rename to `selective_risk` once L2-L7 are enabled.
    metrics["selective_risk_syntax_only"] = _metric(
        "selective_risk_syntax_only", [(r.id, (not r.syntax_pass)) for r in sup_non_refusals]
    )

    # Near-boundary pair accuracy: both members of a boundary pair correct.
    pairs: dict[str, list[DecisionRecord]] = {}
    for r in records:
        if r.boundary_pair_id:
            pairs.setdefault(r.boundary_pair_id, []).append(r)
    pair_items: list[tuple[str, bool]] = []
    for pid, members in pairs.items():
        # a valid boundary pair has both polarities; an incomplete pair cannot be
        # scored as correct (it is exactly the one-sided case this metric must catch).
        has_sup = any(m.is_supported for m in members)
        has_unsup = any(not m.is_supported for m in members)
        complete = len(members) >= 2 and has_sup and has_unsup
        all_correct = complete and all(m.refused == (not m.is_supported) for m in members)
        pair_items.append((pid, all_correct))
    metrics["near_boundary_pair_accuracy"] = _metric(
        "near_boundary_pair_accuracy", pair_items
    )

    # Refusal-detector confusion (positive class = unsupported/refuse)
    tp = sum(1 for r in gold_unsup if r.refused)
    fn = sum(1 for r in gold_unsup if not r.refused)
    fp = sum(1 for r in gold_sup if r.refused)
    tn = sum(1 for r in gold_sup if not r.refused)
    confusion = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    if precision and recall and (precision + recall) > 0:
        boundary_f1 = 2 * precision * recall / (precision + recall)
    else:
        boundary_f1 = 0.0 if (tp + fp + fn) else None

    scalars = {
        "unsupported_precision": precision,
        "unsupported_recall": recall,
        "boundary_f1": boundary_f1,
        "n_gold_supported": len(gold_sup),
        "n_gold_unsupported": len(gold_unsup),
        "n_model_refusals": len(model_refusals),
    }

    return {"metrics": metrics, "confusion": confusion, "scalars": scalars}
