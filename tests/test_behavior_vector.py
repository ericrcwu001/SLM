"""Tests for the measured behavior vector."""

import numpy as np

from data_pipeline.behavior_vector import measure_behavior, smoothness, smoothness_native
from data_pipeline.sources import procedural as proc
from eval.cube_io import GRID_SIZE, identity_grid


def _lut(name):
    return proc.generate_lut_tensor(next(s for s in proc.catalog() if s.lut_id == name))


def test_smoothness_native_normalization():
    # at the canonical size the factor is 1 (identical to plain smoothness)
    rng = np.random.default_rng(0)
    r17 = rng.random((GRID_SIZE, GRID_SIZE, GRID_SIZE, 3)) * 0.01
    assert smoothness_native(r17, GRID_SIZE) == smoothness(r17)
    # a finer native grid is rescaled by ((N-1)/(GRID_SIZE-1))**2
    r33 = rng.random((33, 33, 33, 3)) * 0.01
    expected = smoothness(r33) * ((33 - 1) / (GRID_SIZE - 1)) ** 2
    assert abs(smoothness_native(r33, 33) - expected) < 1e-12


def test_native_lut_smoothness_caps_oversized_grids():
    # capping the measurement grid keeps huge quantized LUTs from blowing up (the 8-bit HaldCLUT
    # failure mode): pure native amplifies quantization noise; capped-at-33 stays bounded.
    from data_pipeline.behavior_vector import native_lut_smoothness
    assert native_lut_smoothness(identity_grid(65)) < 1e-9  # identity stays ~0 either way
    rng = np.random.default_rng(1)
    quantized = np.clip(identity_grid(65) + np.round(rng.random((65, 65, 65, 3)) * 4) / 255.0, 0, 1)
    capped = native_lut_smoothness(quantized)
    pure = smoothness_native(quantized - identity_grid(65), 65)
    assert capped < pure           # cap denoises the quantization amplification
    assert capped < 0.30           # mild quantization noise stays out of the extreme-reject band


def test_identity_behavior_is_near_zero():
    b = measure_behavior(identity_grid(17))
    assert abs(b["temperature_delta_b"]) < 0.2
    assert abs(b["tint_delta_a"]) < 0.2
    assert abs(b["mean_l_delta"]) < 0.2
    assert b["residual_norm"] < 1e-6
    assert b["neutral_drift_deltaE"] < 0.2
    assert b["foldover_rate"] == 0.0
    assert b["clip_rate"] < 1e-6
    assert b["skin_locus_deltaE00_p95"] < 0.2


def test_warmer_positive_temperature():
    assert measure_behavior(_lut("proc_attr_warmer"))["temperature_delta_b"] > 1.5


def test_cooler_negative_temperature():
    assert measure_behavior(_lut("proc_attr_cooler"))["temperature_delta_b"] < -1.5


def test_brighter_positive_l():
    assert measure_behavior(_lut("proc_attr_brighter"))["mean_l_delta"] > 2.0


def test_saturation_signs():
    assert measure_behavior(_lut("proc_attr_more_saturated"))["chroma_delta"] > 2.0
    assert measure_behavior(_lut("proc_attr_muted"))["chroma_delta"] < -2.0


def test_tint_sign():
    assert measure_behavior(_lut("proc_attr_tint_magenta"))["tint_delta_a"] > 1.5


def test_teal_orange_has_split_tone():
    b = measure_behavior(_lut("proc_style_teal-orange"))
    assert b["split_tone_strength"] > 1.0
