"""Tests for the representability gate (direct + pair-fit)."""

import numpy as np

from data_pipeline.lut_ops import apply_lut_trilinear
from data_pipeline.representability import (
    TIER_DIAGNOSTIC,
    TIER_GOLD,
    TIER_REJECTED,
    assess_direct_lut,
    assess_pair_fit,
)
from data_pipeline.sources import procedural as proc
from data_pipeline.sources.derive import fit_global_lut
from eval.cube_io import identity_grid


def _lut(name):
    return proc.generate_lut_tensor(next(s for s in proc.catalog() if s.lut_id == name))


def test_direct_identity_is_gold():
    assert assess_direct_lut(identity_grid(17)).tier == TIER_GOLD


def test_direct_warmth_gold_when_tinted():
    assert assess_direct_lut(_lut("proc_attr_warmer"), tinted=True).tier == TIER_GOLD


def test_direct_moderate_smoothness_is_diagnostic_not_rejected():
    # demote-don't-reject: moderate smoothness (DIAG 0.15 < s <= REJECT 0.30) caps a clean LUT
    # at diagnostic, not rejected.
    r = assess_direct_lut(identity_grid(17), smoothness_override=0.20)
    assert r.tier == TIER_DIAGNOSTIC
    assert "smoothness" in r.reasons


def test_direct_extreme_smoothness_rejected():
    r = assess_direct_lut(identity_grid(17), smoothness_override=0.5)
    assert r.tier == TIER_REJECTED


def test_direct_clipping_lut_rejected():
    harsh = np.clip(identity_grid(17) * 3.0 - 1.0, 0.0, 1.0)
    assert assess_direct_lut(harsh).tier == TIER_REJECTED


def _block_image(colors, block=16, grid=8):
    side = block * grid
    img = np.zeros((side, side, 3), dtype=np.float64)
    for gi in range(grid):
        for gj in range(grid):
            c = colors[(gi * grid + gj) % len(colors)]
            img[gi * block:(gi + 1) * block, gj * block:(gj + 1) * block] = c
    return img


def test_pair_fit_global_lut_is_gold():
    rng = np.random.default_rng(0)
    colors = 0.2 + 0.6 * rng.random((64, 3))
    src = _block_image(colors)
    lut0 = _lut("proc_attr_warmer")
    tgt = apply_lut_trilinear(lut0, src)
    res = assess_pair_fit(lut0, src, tgt, tinted=True)
    assert res.tier == TIER_GOLD
    assert res.fit_deltaE00["mean"] < 0.5
    assert res.support["input_pixel_supported_rate"] >= 0.99


def _photo_like_image(seed=0, n=300):
    """A smooth, densely-sampled image covering a curved slice of the RGB cube (like a photo):
    most 17^3 nodes are unobserved (the sparse-coverage regime that broke the old fitter), yet
    the observed nodes hold enough pixels to clear the support gate."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n] / (n - 1)
    r = 0.12 + 0.75 * xx
    g = 0.18 + 0.62 * (0.5 * xx + 0.5 * yy)
    b = 0.22 + 0.55 * yy ** 1.3
    img = np.stack([r, g, b], axis=-1)
    img += 0.01 * rng.standard_normal(img.shape)     # mild texture
    return np.clip(img, 0, 1)


def test_fitted_global_edit_on_sparse_coverage_is_accepted():
    """Regression for the smooth-fill fix: a *perfectly global* edit fit from a real-coverage
    image must be accepted, not rejected on estimator-induced smoothness/foldover.

    Before the fix, most of the cube was identity-filled and the cliffs read as roughness, so a
    by-construction-global edit was rejected. See derive._smooth_fill_residual.
    """
    src = _photo_like_image(seed=1)
    lut_true = _lut("proc_attr_warmer")               # smooth, monotonic, global by construction
    tgt = apply_lut_trilinear(lut_true, src)

    fitted = fit_global_lut(src, tgt).lut_abs          # smooth fill (default)
    res = assess_pair_fit(fitted, src, tgt, tinted=True)
    # the whole point: the estimator no longer manufactures smoothness/foldover failures
    assert "smoothness" not in res.reasons and "quality:smoothness" not in res.reasons
    assert "foldover" not in res.reasons and "quality:foldover" not in res.reasons
    assert res.tier in (TIER_GOLD, TIER_DIAGNOSTIC), res.reasons
    assert res.fit_deltaE00["mean"] <= 3.0

    # ...and on this same edit the pre-fix estimator (identity fallback) is measurably rougher:
    # the smooth fill strictly lowers the smoothness score the gate keys on.
    raw = fit_global_lut(src, tgt, smooth=False).lut_abs
    raw_res = assess_pair_fit(raw, src, tgt, tinted=True)
    assert res.quality_scores["smoothness"] < raw_res.quality_scores["smoothness"]


def test_pair_fit_local_edit_rejected():
    # target = source + a column-dependent shift -> not representable by a global LUT
    side = 96
    src = np.full((side, side, 3), 0.5)
    tgt = src.copy()
    ramp = np.linspace(0.0, 0.3, side)[None, :]  # varies with column x
    tgt[..., 2] = np.clip(tgt[..., 2] + ramp, 0, 1)
    res = assess_pair_fit(identity_grid(17), src, tgt)
    assert res.tier == TIER_REJECTED
    assert any(r in ("coord_correlation", "residual_xy_r2", "tile_residual") for r in res.reasons)
