"""Tests for quality + safety gates."""

import numpy as np

from data_pipeline.behavior_vector import measure_behavior
from data_pipeline.quality_filters import assess_quality, direction_magnitude_ok
from data_pipeline.sources import procedural as proc
from eval.cube_io import identity_grid


def _lut(name):
    return proc.generate_lut_tensor(next(s for s in proc.catalog() if s.lut_id == name))


def test_identity_passes_safety_and_skin():
    q = assess_quality(identity_grid(17))
    assert q.safety_pass is True
    assert q.skin_pass is True
    assert q.reasons == []


def test_clipping_lut_fails_safety():
    # push everything far out of range so interior inputs clamp heavily
    harsh = np.clip(identity_grid(17) * 3.0 - 1.0, 0.0, 1.0)
    q = assess_quality(harsh, pre_clamp=identity_grid(17) * 3.0 - 1.0)
    assert q.safety_pass is False
    assert any(r in ("clip_rate_exceeded", "pre_clamp_out_of_range") for r in q.reasons)


def test_moderate_smoothness_caps_not_rejects():
    # 0.15 < s <= 0.30 -> diagnostic cap (safety still passes), not a hard reject.
    q = assess_quality(identity_grid(17), smoothness_override=0.20)
    assert q.safety_pass is True
    assert "smoothness" in q.cap_reasons
    assert "smoothness_extreme" not in q.safety_reasons


def test_extreme_smoothness_hard_rejects():
    q = assess_quality(identity_grid(17), smoothness_override=0.5)
    assert q.safety_pass is False
    assert "smoothness_extreme" in q.safety_reasons


def test_smoothness_override_used_over_measured():
    # a clean identity LUT (measured smoothness ~0) is forced extreme via override
    q = assess_quality(identity_grid(17), smoothness_override=0.9)
    assert q.quality_scores["smoothness"] == 0.9
    assert q.safety_pass is False


def test_warmth_needs_tinted_flag_for_neutral_drift():
    warmer = _lut("proc_attr_warmer")
    # without tinted -> neutral drift flagged; with tinted -> allowed
    assert "neutral_drift" in assess_quality(warmer, tinted=False).reasons
    assert "neutral_drift" not in assess_quality(warmer, tinted=True).reasons


def test_direction_magnitude_floor():
    b_warm = measure_behavior(_lut("proc_attr_warmer"))
    assert direction_magnitude_ok(b_warm, "temperature", sign=1) is True
    b_id = measure_behavior(identity_grid(17))
    assert direction_magnitude_ok(b_id, "temperature", sign=1) is False
