"""Tests for Stage-4 canonicalization."""

import numpy as np

from data_pipeline.canonicalize import canonicalize_lut, is_identity
from eval.cube_io import identity_grid


def test_identity_canonicalizes_to_zero_residual():
    res = canonicalize_lut(identity_grid(17), "srgb")
    assert not res.rejected
    assert is_identity(res.residual)
    assert len(res.canonical_absolute_lut_hash) == 64
    assert len(res.canonical_residual_lut_hash) == 64


def test_hashes_deterministic():
    a = canonicalize_lut(identity_grid(17), "srgb")
    b = canonicalize_lut(identity_grid(17), "srgb")
    assert a.canonical_absolute_lut_hash == b.canonical_absolute_lut_hash
    assert a.canonical_residual_lut_hash == b.canonical_residual_lut_hash


def test_resample_from_33_matches_17():
    from_33 = canonicalize_lut(identity_grid(33), "srgb")
    ref = canonicalize_lut(identity_grid(17), "srgb")
    assert from_33.canonical_absolute_lut_hash == ref.canonical_absolute_lut_hash


def test_adobe_rgb_identity_canonicalizes_to_zero_residual():
    # an identity LUT is identity in any RGB space -> color-managed AdobeRGB identity stays identity.
    res = canonicalize_lut(identity_grid(32), "adobe_rgb")
    assert not res.rejected
    assert is_identity(res.residual, atol=1e-6)
    assert "color_managed_adobe_rgb_to_srgb" in res.normalization_warnings


def test_adobe_rgb_path_differs_from_srgb_treatment():
    # a non-identity AdobeRGB LUT color-managed != the same tensor treated as sRGB.
    rng = np.random.default_rng(0)
    lut = np.clip(identity_grid(17) + (rng.random((17, 17, 17, 3)) - 0.5) * 0.1, 0, 1)
    managed = canonicalize_lut(lut, "adobe_rgb")
    as_srgb = canonicalize_lut(lut, "srgb")
    assert managed.canonical_residual_lut_hash != as_srgb.canonical_residual_lut_hash


def test_unknown_domain_assumed_srgb_with_warning():
    res = canonicalize_lut(identity_grid(17), "weird_space")
    assert not res.rejected
    assert any("assumed_srgb" in w for w in res.normalization_warnings)


def test_camera_log_domain_rejected():
    res = canonicalize_lut(identity_grid(17), "camera_log")
    assert res.rejected
    assert "camera_log" in res.reject_reason


def test_out_of_range_clipped_warning():
    lut = identity_grid(17).copy()
    lut[0, 0, 0] = [-0.5, 1.5, 0.5]
    res = canonicalize_lut(lut, "srgb")
    assert "clipped_out_of_range" in res.normalization_warnings
    assert res.absolute.min() >= 0.0 and res.absolute.max() <= 1.0
