"""Tests for core LUT operations (apply, resample, HaldCLUT decode)."""

import numpy as np

from data_pipeline.lut_ops import (
    apply_lut_trilinear,
    hald_level_and_edge,
    haldclut_to_lut,
    lut_to_hald,
    resample_lut,
    residual_norm,
)
from eval.cube_io import absolute_to_residual, identity_grid


def test_apply_identity_is_passthrough():
    lut = identity_grid(17)
    rgb = np.array([[0.1, 0.2, 0.3], [0.9, 0.5, 0.0], [0.5, 0.5, 0.5]])
    out = apply_lut_trilinear(lut, rgb)
    assert np.allclose(out, rgb, atol=1e-9)


def test_resample_identity_stays_identity():
    src = identity_grid(33)
    out = resample_lut(src, 17)
    assert np.allclose(out, identity_grid(17), atol=1e-9)
    assert residual_norm(absolute_to_residual(out)) < 1e-9


def test_hald_level_and_edge():
    assert hald_level_and_edge(8) == (2, 4)
    assert hald_level_and_edge(512) == (8, 64)
    assert hald_level_and_edge(1728) == (12, 144)


def test_haldclut_roundtrip_identity():
    ident = identity_grid(17)
    hald = lut_to_hald(ident, level=8)  # side 512
    assert hald.shape[0] == 512
    back = haldclut_to_lut(hald, target_size=17)
    assert np.allclose(back, ident, atol=1e-6)


def test_haldclut_roundtrip_nonidentity():
    # a simple channel-swap-ish LUT survives hald encode/decode within resample tolerance
    ident = identity_grid(17)
    lut = np.clip(ident * 0.8 + 0.1, 0.0, 1.0)
    hald = lut_to_hald(lut, level=8)
    back = haldclut_to_lut(hald, target_size=17)
    assert np.max(np.abs(back - lut)) < 1e-3
