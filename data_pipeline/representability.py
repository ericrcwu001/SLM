"""Stage 5 representability gate (data_collection_plan.md "Derived LUT Representability Gate").

Two entry points:
  * ``assess_direct_lut`` — HaldCLUT / procedural / pack LUTs are global LUTs by construction
    (no fit error); tier follows quality/safety.
  * ``assess_pair_fit`` — PPR10K/FiveK pair-fitted LUTs: held-out CIEDE2000, spatial residual
    analysis, per-cell support, and tier assignment against the verbatim thresholds.

Fitting/evaluation are in canonical encoded sRGB; ``fit_train_*`` and ``fit_validation_*`` are
reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from eval import color_pipeline as cp
from eval.cube_io import GRID_SIZE

from .constants import QUALITY_FILTER_VERSION
from .lut_ops import apply_lut_trilinear
from .quality_filters import QualityResult, assess_quality

TIER_GOLD = "gold"
TIER_DIAGNOSTIC = "diagnostic_only"
TIER_REJECTED = "rejected"

PAIR_FIT = {
    "mean_accept": 3.0, "p95_accept": 7.0, "p99_accept": 10.0,
    "mean_gold": 2.0,
    "support_accept": 0.98, "support_gold": 0.99,
    "tile_abs_max": 6.0, "tile_rel_mult": 2.5,
    "component_pct": 0.01, "component_dE": 6.0,
    "xy_r2_max": 0.05, "coord_corr_max": 0.25, "edge_corr_max": 0.30,
    "min_support": 32,
}


@dataclass
class RepresentabilityResult:
    tier: str
    status: str                       # "accepted" | "rejected"
    reasons: list = field(default_factory=list)
    fit_deltaE00: dict = field(default_factory=dict)
    fit_train_deltaE00: dict = field(default_factory=dict)
    fit_validation_deltaE00: dict = field(default_factory=dict)
    spatial: dict = field(default_factory=dict)
    support: dict = field(default_factory=dict)
    quality_scores: dict = field(default_factory=dict)
    quality_filter_version: str = QUALITY_FILTER_VERSION


def _stats(dE: np.ndarray) -> dict:
    if dE.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(dE)), "median": float(np.median(dE)),
        "p95": float(np.percentile(dE, 95)), "p99": float(np.percentile(dE, 99)),
        "max": float(np.max(dE)),
    }


def _support_rates(source_flat: np.ndarray, size: int = GRID_SIZE, min_support: int = 32) -> dict:
    idx = np.rint(np.clip(source_flat, 0, 1) * (size - 1)).astype(np.int64)
    flat_node = (idx[:, 0] * size + idx[:, 1]) * size + idx[:, 2]
    counts = np.bincount(flat_node, minlength=size ** 3)
    supported_cells = counts >= min_support
    occupied = counts > 0
    supported_cell_rate = float(supported_cells.sum() / max(1, occupied.sum()))
    input_supported = float(np.mean(supported_cells[flat_node]))
    return {
        "supported_cell_rate": supported_cell_rate,
        "input_pixel_supported_rate": input_supported,
        "occupied_cells": int(occupied.sum()),
    }


def _spatial_residual(residual_map: np.ndarray, source_luma: np.ndarray) -> dict:
    """residual_map, source_luma are 2D (H,W). Compute the spatial gates."""
    h, w = residual_map.shape
    ys, xs = np.mgrid[0:h, 0:w]
    xn = (xs / max(1, w - 1)).reshape(-1)
    yn = (ys / max(1, h - 1)).reshape(-1)
    rad = np.hypot(xn - 0.5, yn - 0.5)
    r = residual_map.reshape(-1)
    global_mean = float(np.mean(r))

    # linear fit residual ~ [1, x, y] -> R^2
    A = np.stack([np.ones_like(xn), xn, yn], axis=1)
    ss_tot = float(np.sum((r - r.mean()) ** 2))
    if ss_tot < 1e-9:
        xy_r2 = 0.0  # near-constant residual has no spatial structure
    else:
        coef, *_ = np.linalg.lstsq(A, r, rcond=None)
        ss_res = float(np.sum((r - (A @ coef)) ** 2))
        xy_r2 = max(0.0, 1.0 - ss_res / ss_tot)

    def _corr(a, b):
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            return 0.0
        return float(abs(np.corrcoef(a, b)[0, 1]))

    # edge magnitude of source luma
    gx = ndimage.sobel(source_luma, axis=1, mode="nearest")
    gy = ndimage.sobel(source_luma, axis=0, mode="nearest")
    edge = np.hypot(gx, gy).reshape(-1)

    # tiles
    tiles = 8
    th, tw = max(1, h // tiles), max(1, w // tiles)
    tile_means = []
    for i in range(0, h, th):
        for j in range(0, w, tw):
            block = residual_map[i:i + th, j:j + tw]
            if block.size:
                tile_means.append(float(block.mean()))
    tile_means = np.array(tile_means) if tile_means else np.array([0.0])

    # largest high-residual connected component
    high = residual_map > PAIR_FIT["component_dE"]
    lbl, n = ndimage.label(high)
    comp_pct = 0.0
    if n:
        sizes = ndimage.sum(np.ones_like(lbl), lbl, index=range(1, n + 1))
        comp_pct = float(sizes.max() / residual_map.size)

    return {
        "global_mean": global_mean,
        "residual_xy_r2": float(xy_r2),
        "corr_x": _corr(r, xn), "corr_y": _corr(r, yn), "corr_radius": _corr(r, rad),
        "residual_edge_corr": _corr(r, edge),
        "residual_tile_p95": float(np.percentile(tile_means, 95)),
        "residual_tile_max": float(tile_means.max()),
        "largest_high_residual_component_pct": comp_pct,
    }


def assess_pair_fit(lut_abs: np.ndarray, source_img: np.ndarray, target_img: np.ndarray,
                    held_out_stride: int = 4, tinted: bool = False,
                    smoothness_override: float | None = None) -> RepresentabilityResult:
    src = np.clip(np.asarray(source_img, dtype=np.float64), 0, 1)
    tgt = np.clip(np.asarray(target_img, dtype=np.float64), 0, 1)
    if src.shape != tgt.shape or src.ndim != 3:
        raise ValueError("source/target must be equal-shape (H,W,3) images")
    h, w, _ = src.shape

    pred = apply_lut_trilinear(lut_abs, src)
    dE_map = cp.ciede2000(cp.srgb_to_lab_d65(pred), cp.srgb_to_lab_d65(tgt))  # (H,W)
    dE = dE_map.reshape(-1)

    # deterministic held-out split
    ys, xs = np.mgrid[0:h, 0:w]
    heldout = ((xs + ys) % held_out_stride == 0).reshape(-1)
    fit_all, fit_train, fit_val = _stats(dE), _stats(dE[~heldout]), _stats(dE[heldout])

    support = _support_rates(src.reshape(-1, 3), min_support=PAIR_FIT["min_support"])
    src_luma = cp.srgb_to_lab_d65(src)[..., 0]
    spatial = _spatial_residual(dE_map, src_luma)

    # Hard rejects: the fitted global LUT fails to *reproduce* the target (magnitude / support)
    # or a whole region is badly off (tile / connected high-residual component). These mean the
    # LUT is not a usable global approximation at all.
    hard_reasons: list[str] = []
    if fit_all["mean"] > PAIR_FIT["mean_accept"]:
        hard_reasons.append("fit_mean_exceeded")
    if fit_all["p95"] > PAIR_FIT["p95_accept"]:
        hard_reasons.append("fit_p95_exceeded")
    if fit_all["p99"] > PAIR_FIT["p99_accept"]:
        hard_reasons.append("fit_p99_exceeded")
    if support["input_pixel_supported_rate"] < PAIR_FIT["support_accept"]:
        hard_reasons.append("input_support_low")
    tile_limit = max(PAIR_FIT["tile_abs_max"], PAIR_FIT["tile_rel_mult"] * spatial["global_mean"])
    if spatial["residual_tile_max"] > tile_limit:
        hard_reasons.append("tile_residual")
    if spatial["largest_high_residual_component_pct"] > PAIR_FIT["component_pct"]:
        hard_reasons.append("high_residual_component")

    # Structure gates: the residual is *within tolerance* everywhere (it cleared the magnitude
    # gates) but is spatially correlated with position/edges -- the signature of a minor local
    # component in the source edit. The fitted LUT is still a valid global approximation, so
    # this disqualifies the row from gold/headline but does NOT reject it (diagnostic tier:
    # usable for tokenizer/warmup/SFT, never a headline eval row). See ADR / data_collection_plan
    # "Derived LUT Representability Gate".
    structure_reasons: list[str] = []
    if spatial["residual_xy_r2"] > PAIR_FIT["xy_r2_max"]:
        structure_reasons.append("residual_xy_r2")
    if max(spatial["corr_x"], spatial["corr_y"], spatial["corr_radius"]) > PAIR_FIT["coord_corr_max"]:
        structure_reasons.append("coord_correlation")
    if spatial["residual_edge_corr"] > PAIR_FIT["edge_corr_max"]:
        structure_reasons.append("edge_correlation")

    quality = assess_quality(lut_abs, tinted=tinted, smoothness_override=smoothness_override)
    # core-safety failures are hard rejects; skin-only + diagnostic-cap (moderate smoothness)
    # failures cap the tier at diagnostic.
    hard_reasons.extend(f"quality:{r}" for r in quality.safety_reasons)
    skin_reasons = [f"skin:{r}" for r in quality.skin_reasons]
    cap_reasons = [f"cap:{r}" for r in quality.cap_reasons]

    if hard_reasons:
        tier, status = TIER_REJECTED, "rejected"
        reasons = hard_reasons + structure_reasons + skin_reasons + cap_reasons
    elif (fit_all["mean"] <= PAIR_FIT["mean_gold"]
          and support["input_pixel_supported_rate"] >= PAIR_FIT["support_gold"]
          and quality.skin_pass and not structure_reasons and not quality.cap_reasons):
        tier, status, reasons = TIER_GOLD, "accepted", []
    else:
        tier, status = TIER_DIAGNOSTIC, "accepted"
        reasons = structure_reasons + skin_reasons + cap_reasons  # recorded, not fatal

    return RepresentabilityResult(
        tier=tier, status=status, reasons=reasons,
        fit_deltaE00=fit_all, fit_train_deltaE00=fit_train, fit_validation_deltaE00=fit_val,
        spatial=spatial, support=support, quality_scores=quality.quality_scores,
    )


def assess_direct_lut(lut_abs: np.ndarray, tinted: bool = False,
                      quality: QualityResult | None = None,
                      smoothness_override: float | None = None) -> RepresentabilityResult:
    """A directly-provided global LUT (HaldCLUT/procedural/pack): tier follows quality/safety.

    Skin-locus and diagnostic-cap reasons (moderate smoothness) cap the tier at diagnostic;
    only core-safety failures reject.
    """
    q = quality if quality is not None else assess_quality(
        lut_abs, tinted=tinted, smoothness_override=smoothness_override)
    if q.safety_pass and q.skin_pass and not q.cap_reasons:
        tier, status, reasons = TIER_GOLD, "accepted", []
    elif q.safety_pass:
        tier, status, reasons = TIER_DIAGNOSTIC, "accepted", list(q.reasons)
    else:
        tier, status, reasons = TIER_REJECTED, "rejected", list(q.reasons)
    return RepresentabilityResult(
        tier=tier, status=status, reasons=reasons,
        fit_deltaE00={"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0},
        support={"supported_cell_rate": 1.0, "input_pixel_supported_rate": 1.0},
        quality_scores=q.quality_scores,
    )
