"""Behavioral scoring for a frontier-produced LUT — the "is it good?" layer.

The frozen eval rows carry no target/reference LUTs (``target_lut_path`` is null), so L5
target fidelity is impossible. What *is* measurable without a target — and without the VQ
tokenizer — is whether the LUT actually performs the requested global edit. This module
scores two things on a decoded absolute LUT:

  * **Direction (L4-style)**: does the LUT move the image in the direction each explicit
    gold tag names, by a perceptible magnitude? Tag -> measured axis + sign follows the
    table in docs/detailed_behavior_spec.md ("Measured Behavior", lines ~103-110); the
    magnitude bar is that doc's ">= 1.5 Lab units for tint/temperature tags".
  * **Safety (L6-style)**: is the LUT a sane transform — finite, non-degenerate (not the
    identity), smooth, monotone (no foldover), not clipping most of the range?

Both reuse the already-implemented, decoder-free color machinery
(data_pipeline.behavior_vector.measure_behavior). Thresholds here are provisional pilot
values, not the frozen calibration_manifest windows; they are named constants so they are
easy to see and tune. Style-bundle recipe windows (L7: matte/faded/cinematic/...) are NOT
scored here — those tags are treated as non-directional and leave direction not_evaluated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from data_pipeline.behavior_vector import measure_behavior
from eval.tag_vocabulary import DIRECTIONAL_TAG_AXIS, canonicalize_tags, min_magnitude_for_axis

PASS = "pass"
FAIL = "fail"
NOT_EVALUATED = "not_evaluated"

# gold tag -> list of (behavior_vector key, required sign (+1/-1), min |magnitude|).
# A tag passes iff its measured metric has the right sign AND clears the magnitude bar. Built from
# the ONE unified tag vocabulary (ADR 0022; eval.tag_vocabulary) so it can never drift from
# instruction_gen._TAG_BEHAVIOR. Retired aliases (more_magenta/higher_contrast/desaturated/…) are
# accepted via canonicalize_tags on ingest (evaluate_direction). Style bundles are absent (they are
# non-directional). Per-axis magnitude bars come from min_magnitude_for_axis (Lab 1.5 / chroma 1.0).
TAG_DIRECTIONS: dict[str, list[tuple[str, int, float]]] = {
    tag: [(axis, sign, min_magnitude_for_axis(axis))]
    for tag, (axis, sign) in DIRECTIONAL_TAG_AXIS.items()
}

# Safety thresholds (provisional pilot values).
_MAX_FOLDOVER = 0.02        # fraction of severely non-monotone node steps
_MAX_CLIP_RATE = 0.35       # fraction of probe outputs clamped at 0/1
_MAX_SMOOTHNESS = 0.12      # p99 of |2nd differences| of the residual lattice
_MIN_RESIDUAL_NORM = 5e-4   # below this the LUT is effectively the identity (did nothing)


@dataclass
class DirectionResult:
    status: str  # PASS | FAIL | NOT_EVALUATED
    per_tag: dict[str, dict] = field(default_factory=dict)
    directional_tags: list[str] = field(default_factory=list)


@dataclass
class SafetyResult:
    status: str  # PASS | FAIL
    reasons: list[str] = field(default_factory=list)


def evaluate_direction(behavior: dict, gold_tags: list[str]) -> DirectionResult:
    """Row passes direction iff every directional gold tag moves the right way + enough.

    Rows whose tags are all non-directional (pure style bundles, or empty) are
    ``not_evaluated`` — direction is genuinely undefined for them here.
    """
    directional = [t for t in canonicalize_tags(gold_tags) if t in TAG_DIRECTIONS]
    if not directional:
        return DirectionResult(NOT_EVALUATED, directional_tags=[])

    per_tag: dict[str, dict] = {}
    all_pass = True
    for tag in directional:
        tag_pass = True
        checks = []
        for key, sign, min_mag in TAG_DIRECTIONS[tag]:
            measured = float(behavior.get(key, 0.0))
            ok = (measured * sign > 0) and (abs(measured) >= min_mag)
            checks.append({"metric": key, "measured": measured,
                           "expected_sign": sign, "min_magnitude": min_mag, "pass": ok})
            tag_pass = tag_pass and ok
        per_tag[tag] = {"pass": tag_pass, "checks": checks}
        all_pass = all_pass and tag_pass

    return DirectionResult(PASS if all_pass else FAIL, per_tag=per_tag, directional_tags=directional)


def evaluate_safety(behavior: dict) -> SafetyResult:
    reasons: list[str] = []
    foldover = float(behavior.get("foldover_rate", 0.0))
    clip = float(behavior.get("clip_rate", 0.0))
    smooth = float(behavior.get("smoothness", 0.0))
    resid = float(behavior.get("residual_norm", 0.0))

    if not np.isfinite([foldover, clip, smooth, resid]).all():
        reasons.append("non_finite_behavior")
    if resid < _MIN_RESIDUAL_NORM:
        reasons.append(f"degenerate_identity:residual_norm={resid:.2e}")
    if foldover > _MAX_FOLDOVER:
        reasons.append(f"foldover:{foldover:.3f}>{_MAX_FOLDOVER}")
    if clip > _MAX_CLIP_RATE:
        reasons.append(f"clip_rate:{clip:.3f}>{_MAX_CLIP_RATE}")
    if smooth > _MAX_SMOOTHNESS:
        reasons.append(f"not_smooth:{smooth:.3f}>{_MAX_SMOOTHNESS}")

    return SafetyResult(PASS if not reasons else FAIL, reasons=reasons)


@dataclass
class LutScore:
    behavior: dict
    direction: DirectionResult
    safety: SafetyResult

    @property
    def lut_quality_pass(self) -> Optional[bool]:
        """Headline 'good LUT' verdict on a supported, directional row.

        None when direction is not_evaluated (no directional tag) — such rows can't
        contribute to the direction-based quality headline.
        """
        if self.direction.status == NOT_EVALUATED:
            return None
        return self.direction.status == PASS and self.safety.status == PASS


def score_lut(lut_abs: np.ndarray, gold_tags: list[str]) -> LutScore:
    behavior = measure_behavior(lut_abs)
    return LutScore(
        behavior=behavior,
        direction=evaluate_direction(behavior, gold_tags),
        safety=evaluate_safety(behavior),
    )
