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


# --- behavior_v2 axes (ADR 0022) -------------------------------------------------------------
from data_pipeline.constants import BEHAVIOR_VECTOR_VERSION  # noqa: E402
from eval.tag_vocabulary import HUE_SECTORS  # noqa: E402

_V2_FIELDS = {
    "global_hue_deg", "global_hue_magnitude", "shadow_hue_deg", "midtone_hue_deg",
    "highlight_hue_deg", "per_hue_saturation", "contrast_toe_delta", "contrast_shoulder_delta",
    "matte_strength",
}


def test_version_is_behavior_v2_and_fields_present():
    b = measure_behavior(_lut("proc_attr_warmer"))
    assert BEHAVIOR_VECTOR_VERSION == "behavior_v2"
    assert b["behavior_vector_version"] == "behavior_v2"
    assert _V2_FIELDS <= set(b)                      # all new axes present
    # all 27 behavior_v1 fields retained (spot-check the axis families)
    for k in ("temperature_delta_b", "tint_delta_a", "mean_l_delta", "chroma_delta",
              "split_tone_strength", "skin_locus_deltaE00_mean", "residual_norm"):
        assert k in b


def test_identity_v2_axes_near_zero():
    b = measure_behavior(identity_grid(17))
    assert b["global_hue_magnitude"] < 0.2
    assert abs(b["matte_strength"]) < 0.2
    assert abs(b["contrast_toe_delta"]) < 0.05 and abs(b["contrast_shoulder_delta"]) < 0.05
    assert all(abs(v) < 0.2 for v in b["per_hue_saturation"].values())


def test_global_hue_angle_warm_points_yellow():
    # A warm cast is +b* (toward yellow); Lab hue atan2(b,a) ~ 90 deg.
    b = measure_behavior(_lut("proc_attr_warmer"))
    assert b["global_hue_magnitude"] > 1.5
    assert 60.0 <= b["global_hue_deg"] <= 120.0


def test_per_hue_saturation_is_seven_sector_map():
    b = measure_behavior(_lut("proc_attr_muted"))
    phs = b["per_hue_saturation"]
    assert set(phs) == set(HUE_SECTORS)
    # muting pulls chroma down, so at least one populated sector is negative
    assert min(phs.values()) < 0.0


def test_matte_strength_positive_for_matte_and_faded():
    assert measure_behavior(_lut("proc_style_matte"))["matte_strength"] > 0.5
    assert measure_behavior(_lut("proc_style_faded"))["matte_strength"] > 0.5
    assert measure_behavior(_lut("proc_attr_warmer"))["matte_strength"] < 0.5  # a warm tint is not matte


def test_contrast_shape_toe_shoulder():
    # less_contrast softens the tone curve -> toe rises / shoulder falls relative to identity;
    # the shape axes must at least differ measurably between more- and less-contrast LUTs.
    more = measure_behavior(_lut("proc_attr_more_contrast"))
    less = measure_behavior(_lut("proc_attr_less_contrast"))
    assert more["contrast_shoulder_delta"] != less["contrast_shoulder_delta"]
    assert more["contrast_toe_delta"] != less["contrast_toe_delta"]


def test_teal_orange_region_hue_split():
    b = measure_behavior(_lut("proc_style_teal-orange"))
    # shadows teal (~180-220 deg), highlights orange (~0-60 deg) — the defining split.
    assert 150.0 <= b["shadow_hue_deg"] <= 250.0
    assert (b["highlight_hue_deg"] <= 70.0) or (b["highlight_hue_deg"] >= 330.0)
