"""Measured behavior vector (data_collection_plan.md "Measured Behavior Vector").

Probes an absolute LUT on a fixed neutral ramp + chromatic chart + skin anchors + highlight/
shadow samples and measures Lab-domain deltas. This vector is the authority for prompt tags:
if a prompt says "warmer" but the measured behavior is cooler, the row is rejected/regenerated.

Deterministic (fixed probe set) and real (uses :mod:`eval.color_pipeline`).
"""

from __future__ import annotations

import numpy as np

from eval import color_pipeline as cp
from eval.cube_io import GRID_SIZE, absolute_to_residual, identity_grid

from .constants import BEHAVIOR_VECTOR_VERSION
from .lut_ops import apply_lut_trilinear, resample_lut, residual_norm

# skin_locus_v1 anchors (detailed_behavior_spec.md), sRGB8 -> [0,1].
SKIN_LOCUS_V1 = {
    "cc_dark_skin": (115, 82, 68),
    "cc_light_skin": (194, 150, 130),
    "deep_anchor": (74, 48, 38),
    "medium_anchor": (144, 98, 75),
    "tan_anchor": (173, 123, 96),
    "fair_anchor": (231, 195, 170),
}


def _skin_rgb() -> np.ndarray:
    return np.array([[r / 255.0, g / 255.0, b / 255.0] for (r, g, b) in SKIN_LOCUS_V1.values()])


def _neutral_ramp(n: int = 33) -> np.ndarray:
    v = np.linspace(0.0, 1.0, n)
    return np.stack([v, v, v], axis=1)


def _color_chart(step: int = 5) -> np.ndarray:
    axis = np.linspace(0.05, 0.95, step)
    grid = np.array([[r, g, b] for r in axis for g in axis for b in axis])
    # keep chromatic samples only (drop near-neutral)
    spread = grid.max(axis=1) - grid.min(axis=1)
    return grid[spread > 0.15]


def _clip_probe() -> np.ndarray:
    """Typical-pixel probe for the clip gate: grays + skin + moderate-chroma colors.

    Excludes near-gamut-boundary primaries, which clip under any transform and are not
    representative of real image content.
    """
    grays = _neutral_ramp(17)
    grays = grays[(grays[:, 0] > 1e-6) & (grays[:, 0] < 1.0 - 1e-6)]
    axis = np.linspace(0.2, 0.8, 5)
    grid = np.array([[r, g, b] for r in axis for g in axis for b in axis])
    spread = grid.max(axis=1) - grid.min(axis=1)
    moderate = grid[spread <= 0.4]
    return np.concatenate([grays, moderate, _skin_rgb()], axis=0)


def _pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q)) if x.size else 0.0


def _hue_drift_deg(before_lab: np.ndarray, after_lab: np.ndarray) -> np.ndarray:
    hb = np.radians(cp.hue_deg(before_lab))
    ha = np.radians(cp.hue_deg(after_lab))
    d = np.degrees(np.arctan2(np.sin(ha - hb), np.cos(ha - hb)))
    return np.abs(d)


def measure_behavior(lut_abs: np.ndarray) -> dict:
    """Return the measured behavior vector for an absolute LUT."""
    residual = absolute_to_residual(lut_abs)

    # neutral ramp
    ramp = _neutral_ramp()
    r_before = cp.srgb_to_lab_d65(ramp)
    r_after = cp.srgb_to_lab_d65(apply_lut_trilinear(lut_abs, ramp))
    dL = r_after[:, 0] - r_before[:, 0]
    da = r_after[:, 1] - r_before[:, 1]
    db = r_after[:, 2] - r_before[:, 2]

    low = ramp[:, 0] <= 0.25
    high = ramp[:, 0] >= 0.75

    spread_before = _pct(r_before[:, 0], 95) - _pct(r_before[:, 0], 5)
    spread_after = _pct(r_after[:, 0], 95) - _pct(r_after[:, 0], 5)

    # chromatic chart
    chart = _color_chart()
    c_before = cp.srgb_to_lab_d65(chart)
    c_after = cp.srgb_to_lab_d65(apply_lut_trilinear(lut_abs, chart))
    chroma_before = cp.chroma(c_before)
    chroma_after = cp.chroma(c_after)

    # highlight / shadow regions (by input luma)
    chart_L = c_before[:, 0]
    hi = chart_L >= 66.0
    lo = chart_L <= 33.0

    # split tone: chroma-weighted a/b shift in shadows vs highlights
    ab_shift = c_after[:, 1:] - c_before[:, 1:]
    shadow_ab = ab_shift[lo].mean(axis=0) if lo.any() else np.zeros(2)
    highlight_ab = ab_shift[hi].mean(axis=0) if hi.any() else np.zeros(2)

    # skin locus
    skin = _skin_rgb()
    s_before = cp.srgb_to_lab_d65(skin)
    s_after = cp.srgb_to_lab_d65(apply_lut_trilinear(lut_abs, skin))
    skin_dE = cp.ciede2000(s_before, s_after)
    skin_hue = _hue_drift_deg(s_before, s_after)
    skin_chroma_ratio = cp.chroma(s_after) / np.maximum(cp.chroma(s_before), 1e-6)

    # clip rate: fraction of OUTPUT channels clamped at 0/1 for typical (non-extreme) inputs
    # (identity -> 0; only genuine clamping from the transform counts, not boundary nodes).
    probe_out = apply_lut_trilinear(lut_abs, _clip_probe()).reshape(-1)
    clip_rate = float(np.mean((probe_out <= 1e-6) | (probe_out >= 1.0 - 1e-6)))

    # neutral drift: how far neutrals move OFF the neutral axis (chroma-driven ΔE00), so a
    # pure lightness change is not counted; a warmth/tint that colors neutrals is.
    achroma = r_after.copy()
    achroma[:, 1] = 0.0
    achroma[:, 2] = 0.0
    neutral_off = cp.ciede2000(r_after, achroma)

    return {
        "behavior_vector_version": BEHAVIOR_VECTOR_VERSION,
        "temperature_delta_b": float(np.mean(db)),
        "tint_delta_a": float(np.mean(da)),
        "mean_l_delta": float(np.mean(dL)),
        "contrast_l_spread_delta": float(spread_after - spread_before),
        "black_point_l_delta": float(np.mean(dL[low])) if low.any() else 0.0,
        "highlight_l_delta": float(np.mean((r_after[:, 0] - r_before[:, 0])[high])) if high.any() else 0.0,
        "shadow_l_delta": float(np.mean((r_after[:, 0] - r_before[:, 0])[low])) if low.any() else 0.0,
        "highlight_chroma_delta": float(np.mean((chroma_after - chroma_before)[hi])) if hi.any() else 0.0,
        "shadow_chroma_delta": float(np.mean((chroma_after - chroma_before)[lo])) if lo.any() else 0.0,
        "chroma_delta": float(np.mean(chroma_after - chroma_before)),
        "split_tone_shadow_a": float(shadow_ab[0]),
        "split_tone_shadow_b": float(shadow_ab[1]),
        "split_tone_highlight_a": float(highlight_ab[0]),
        "split_tone_highlight_b": float(highlight_ab[1]),
        "split_tone_strength": float(np.hypot(*shadow_ab) + np.hypot(*highlight_ab)),
        "neutral_drift_deltaE": float(np.mean(neutral_off)),
        "neutral_drift_deltaE_p95": float(_pct(neutral_off, 95)),
        "skin_locus_deltaE00_mean": float(np.mean(skin_dE)),
        "skin_locus_deltaE00_p95": float(_pct(skin_dE, 95)),
        "skin_locus_hue_drift_deg_p95": float(_pct(skin_hue, 95)),
        "skin_chroma_ratio_min": float(np.min(skin_chroma_ratio)),
        "skin_chroma_ratio_max": float(np.max(skin_chroma_ratio)),
        "clip_rate": clip_rate,
        "smoothness": smoothness(residual),
        "foldover_rate": foldover_rate(lut_abs),
        "residual_norm": residual_norm(residual),
    }


def smoothness(residual: np.ndarray) -> float:
    """p99 of absolute second differences of the residual across the lattice (all axes)."""
    diffs = []
    for axis in range(3):
        d2 = np.diff(residual, n=2, axis=axis)
        diffs.append(np.abs(d2).reshape(-1))
    alld = np.concatenate(diffs) if diffs else np.array([0.0])
    return float(np.percentile(alld, 99))


def smoothness_native(residual_native: np.ndarray, native_size: int) -> float:
    """Resolution-normalized smoothness on the LUT's NATIVE-resolution residual.

    Second differences scale ~ curvature * h^2, so a finer native grid yields smaller raw values;
    we rescale to canonical 17^3-equivalent curvature by ``((native_size-1)/(GRID_SIZE-1))**2`` so
    the same threshold applies. Measuring on the native grid (not the trilinearly-downsampled 17^3
    residual) removes aliasing introduced by OUR resampling, so the gate reflects the LUT's own
    bumpiness. For a LUT already at 17^3 the factor is 1 (unchanged).
    """
    if native_size <= 1:
        return smoothness(residual_native)
    scale = ((native_size - 1) / (GRID_SIZE - 1)) ** 2
    return smoothness(residual_native) * scale


# Cap the resample-aware measurement grid. Below this we measure at true native (the 32/33 packs
# where our 17^3 downsampling aliases); above it we resample down first. The cap avoids blowing up
# on huge 8-bit HaldCLUT grids (e.g. 144^3), whose per-node quantization noise (~1/255), amplified
# by the (native/16)^2 normalization, would otherwise swamp the signal (~0.9 vs a true ~0.07).
SMOOTHNESS_REF_MAX = 33


def native_lut_smoothness(lut_native: np.ndarray, ref_max: int = SMOOTHNESS_REF_MAX) -> float:
    """Resample-aware smoothness of a raw LUT, measured on its native grid but capped at ``ref_max``.

    For ``native <= ref_max`` (the packs where 17^3 downsampling aliases) it measures at native,
    unchanged. For larger grids it trilinearly resamples down to ``ref_max`` first -- which also
    averages out 8-bit quantization noise -- then normalizes to 17^3-equivalent curvature.
    """
    lut = np.asarray(lut_native, dtype=np.float64)
    n = int(lut.shape[0])
    if n > ref_max:
        lut = resample_lut(lut, ref_max)
        n = ref_max
    resid = np.clip(lut, 0.0, 1.0) - identity_grid(n)
    return smoothness_native(resid, n)


def foldover_rate(lut_abs: np.ndarray) -> float:
    """Fraction of adjacent node steps whose luma (approx via mean channel) reverses.

    A monotone tone LUT increases output as the input increases along each axis; sign
    reversals of the first difference indicate foldover / non-monotonicity.
    """
    n = lut_abs.shape[0]
    luma = lut_abs.mean(axis=3)  # [N,N,N]
    reversals = 0
    total = 0
    severe = 0.02  # only SEVERE backward steps count (spec: severe grid-cell violations)
    for axis in range(3):
        d = np.diff(luma, axis=axis)
        total += d.size
        reversals += int(np.sum(d < -severe))
    return float(reversals / total) if total else 0.0
