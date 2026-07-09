"""Quality + safety gates on a canonical LUT (detailed_behavior_spec.md "Safety Gates").

Post-fit acceptance checks (distinct from the fit-time priors). Thresholds are the verbatim
provisional spec values. Skin-locus is evaluated intrinsically on the fixed ``skin_locus_v1``
anchors regardless of image content.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from eval import color_pipeline as cp

from .behavior_vector import SKIN_LOCUS_V1, _skin_rgb, measure_behavior
from .constants import QUALITY_FILTER_VERSION
from .lut_ops import apply_lut_trilinear

SAFETY_THRESHOLDS = {
    "clip_rate_max": 0.005,          # <= 0.5% of output channels at 0/1
    "out_of_range_max": 0.03,        # pre-clamp max violation
    # <= 0.5% non-monotone node transitions. 0.1% was punitive for pair-fit LUTs: a handful of
    # inversions in sparsely-sampled cube corners (saturated near-black/near-white colours that
    # rarely occur in images) is cosmetically negligible for a residual LUT. Gold-tier direct
    # LUTs still measure ~0 foldover, so this only admits borderline pair-fits to diagnostic.
    "foldover_rate_max": 0.005,
    "neutral_drift_max": 3.0,        # neutral-axis DeltaE00 (unless explicitly tinted)
}

# Smoothness is two-tier + demote-don't-reject (measured resample-aware on the native grid):
#  * <= DIAG: clean (gold-eligible).
#  * DIAG < s <= REJECT: caps the tier at diagnostic (a stylized/creative LUT with sharper tonal
#    transitions is still usable), but does NOT hard-reject.
#  * > REJECT: hard reject (genuine node-to-node jitter / a broken LUT; observed jitter reached ~1.7).
# DIAG widened 0.10 -> 0.15 (quality_v6): admits mildly-textured film/creative LUTs to gold
# instead of capping them at diagnostic. Reject bar unchanged, so real jitter still rejects.
SMOOTHNESS_DIAG_MAX = 0.15
SMOOTHNESS_REJECT_MAX = 0.30
SKIN_THRESHOLDS = {
    "clip_rate": 0.0,
    "hue_drift_deg_p95_max": 8.0,
    "deltaE00_p95_max": 12.0,
    "chroma_ratio_min": 0.75,
    "chroma_ratio_max": 1.35,
}

# direction-magnitude floors for tag validation (Safety Gates "Direction magnitude").
DIRECTION_FLOORS = {
    "temperature": ("temperature_delta_b", 1.5),
    "tint": ("tint_delta_a", 1.5),
    "exposure": ("mean_l_delta", 2.0),
    "shadows": ("shadow_l_delta", 2.0),
    "highlights": ("highlight_l_delta", 2.0),
    "black_point": ("black_point_l_delta", 2.0),
    "saturation": ("chroma_delta", 2.0),
    "contrast": ("contrast_l_spread_delta", 2.5),
}


@dataclass
class QualityResult:
    quality_scores: dict = field(default_factory=dict)
    safety_pass: bool = True                       # non-skin core safety gates (hard reject)
    skin_pass: bool = True                         # skin-locus gate (caps tier at diagnostic)
    safety_reasons: list = field(default_factory=list)
    skin_reasons: list = field(default_factory=list)
    cap_reasons: list = field(default_factory=list)  # diagnostic-cap (e.g. moderate smoothness)
    reasons: list = field(default_factory=list)    # combined, for reporting
    quality_filter_version: str = QUALITY_FILTER_VERSION


def _pre_clamp_violation(lut_abs: np.ndarray, pre_clamp: np.ndarray | None) -> float:
    src = pre_clamp if pre_clamp is not None else lut_abs
    return float(max(0.0, src.max() - 1.0, -src.min()))


def _skin_lightness_order_violations(lut_abs: np.ndarray) -> int:
    skin = _skin_rgb()
    before = cp.srgb_to_lab_d65(skin)[:, 0]
    after = cp.srgb_to_lab_d65(apply_lut_trilinear(lut_abs, skin))[:, 0]
    order_before = np.argsort(before)
    # count adjacent inversions of the before-order under the after lightness
    seq = after[order_before]
    return int(np.sum(np.diff(seq) < -1e-6))


def assess_quality(lut_abs: np.ndarray, behavior: dict | None = None,
                   pre_clamp: np.ndarray | None = None, tinted: bool = False,
                   smoothness_override: float | None = None) -> QualityResult:
    """Compute quality scores + evaluate safety and skin-locus gates for an absolute LUT.

    ``smoothness_override`` (the resample-aware native-resolution smoothness) is used for the
    smoothness gate/score in place of the 17^3-residual value when provided.
    """
    b = behavior if behavior is not None else measure_behavior(lut_abs)
    reasons: list[str] = []
    cap_reasons: list[str] = []

    smooth = smoothness_override if smoothness_override is not None else b["smoothness"]
    out_of_range = _pre_clamp_violation(lut_abs, pre_clamp)
    scores = {
        "clip_rate": b["clip_rate"],
        "smoothness": smooth,
        "foldover_rate": b["foldover_rate"],
        "neutral_drift_deltaE": b["neutral_drift_deltaE"],
        "pre_clamp_out_of_range": out_of_range,
        "residual_magnitude": b["residual_norm"],
        "skin_locus_deltaE00_p95": b["skin_locus_deltaE00_p95"],
        "skin_locus_hue_drift_deg_p95": b["skin_locus_hue_drift_deg_p95"],
    }

    if b["clip_rate"] > SAFETY_THRESHOLDS["clip_rate_max"]:
        reasons.append("clip_rate_exceeded")
    if out_of_range > SAFETY_THRESHOLDS["out_of_range_max"]:
        reasons.append("pre_clamp_out_of_range")
    if b["foldover_rate"] > SAFETY_THRESHOLDS["foldover_rate_max"]:
        reasons.append("foldover")
    # smoothness: demote-don't-reject unless extreme (see SMOOTHNESS_* constants)
    if smooth > SMOOTHNESS_REJECT_MAX:
        reasons.append("smoothness_extreme")
    elif smooth > SMOOTHNESS_DIAG_MAX:
        cap_reasons.append("smoothness")
    if not tinted and b["neutral_drift_deltaE"] > SAFETY_THRESHOLDS["neutral_drift_max"]:
        reasons.append("neutral_drift")

    skin_reasons: list[str] = []
    if b["skin_locus_hue_drift_deg_p95"] > SKIN_THRESHOLDS["hue_drift_deg_p95_max"]:
        skin_reasons.append("skin_hue_drift")
    if b["skin_locus_deltaE00_p95"] > SKIN_THRESHOLDS["deltaE00_p95_max"]:
        skin_reasons.append("skin_deltaE")
    if b["skin_chroma_ratio_min"] < SKIN_THRESHOLDS["chroma_ratio_min"]:
        skin_reasons.append("skin_chroma_low")
    if b["skin_chroma_ratio_max"] > SKIN_THRESHOLDS["chroma_ratio_max"]:
        skin_reasons.append("skin_chroma_high")
    if _skin_lightness_order_violations(lut_abs) > 0:
        skin_reasons.append("skin_lightness_order")

    scores["skin_lightness_order_violations"] = _skin_lightness_order_violations(lut_abs)
    # Core safety is hard-reject; skin-locus and diagnostic-cap reasons (moderate smoothness) only
    # cap the tier at diagnostic. A clean global LUT that merely shifts skin (e.g. a B&W film sim)
    # or has a slightly sharp-but-coherent transition stays diagnostic-usable, not rejected.
    return QualityResult(
        quality_scores=scores,
        safety_pass=(len(reasons) == 0),
        skin_pass=(len(skin_reasons) == 0),
        safety_reasons=list(reasons),
        skin_reasons=list(skin_reasons),
        cap_reasons=list(cap_reasons),
        reasons=reasons + skin_reasons + cap_reasons,
    )


def direction_magnitude_ok(behavior: dict, attribute: str, sign: int = 1) -> bool:
    """True if the measured behavior meets the direction-magnitude floor for ``attribute``."""
    if attribute not in DIRECTION_FLOORS:
        return True
    key, floor = DIRECTION_FLOORS[attribute]
    val = behavior.get(key, 0.0)
    return (sign * val) >= floor
