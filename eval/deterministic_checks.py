"""Deterministic color/LUT checks (L4 direction, L6 safety, L7 style, skin-locus).

DISABLED in this build — every check needs a decoded LUT (see :mod:`eval.lut_decoder`).
Each function returns a ``not_evaluated: decoder_disabled`` :class:`LayerResult` so
``run_eval`` records the layer without fabricating a pass. Signatures match the eventual
real ones (they will take a decoded absolute LUT + the calibration manifest).

When enabled these implement:
  * L4 direction: docs/eval_harness_implementation.md "Direction Checks" table +
    "Final eval minimum detectable movement" magnitudes.
  * L6 safety: "Safety Checks" (clip, out-of-range, foldover, smoothness, neutral
    drift) + skin-locus gate.
  * L7 style: "Style Metrics" recipe windows + discriminability (calibration_manifest).
"""

from __future__ import annotations

from typing import Optional

from .schemas import LayerResult


def direction_check(row, decoded_lut=None, calibration=None) -> LayerResult:  # noqa: ANN001
    return LayerResult.disabled("L4_direction")


def safety_check(row, decoded_lut=None, calibration=None) -> LayerResult:  # noqa: ANN001
    return LayerResult.disabled("L6_safety")


def skin_locus_check(row, decoded_lut=None, calibration=None) -> LayerResult:  # noqa: ANN001
    return LayerResult.disabled("L6_skin_locus")


def style_check(row, decoded_lut=None, calibration=None) -> LayerResult:  # noqa: ANN001
    return LayerResult.disabled("L7_style")


def run_all(row, decoded_lut=None, calibration=None) -> dict[str, LayerResult]:  # noqa: ANN001
    """Run L4/L6/L7 (+ skin-locus). All disabled here."""
    return {
        "L4_direction": direction_check(row, decoded_lut, calibration),
        "L6_safety": safety_check(row, decoded_lut, calibration),
        "L6_skin_locus": skin_locus_check(row, decoded_lut, calibration),
        "L7_style": style_check(row, decoded_lut, calibration),
    }
