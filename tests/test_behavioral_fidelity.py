"""Tests for the free-running behavioral-fidelity metric (eval.behavioral_fidelity).

The scoring/aggregation helpers are pure numpy + the decoder-free color machinery, so
they run without torch or the frozen VQ weights. The one decode-path test is skipped when
the frozen weights are absent (they are gitignored; present only on a staged corpus).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from data_pipeline.attribute_spec import AttributeSpec, from_measured_behavior
from data_pipeline.behavior_vector import measure_behavior
from eval import cube_io
from eval.behavioral_fidelity import (
    behavioral_agreement,
    code_histogram_stats,
    decoded_delta_e,
    score_from_lut,
    summarize_fidelity,
)


# --- helpers ---------------------------------------------------------------------
def _warm_dark_lut() -> np.ndarray:
    """A clearly non-trivial LUT: less blue (warmer, +b*) + all channels down (darker)."""
    lut = cube_io.identity_grid(17).copy()
    lut[..., 2] = np.clip(lut[..., 2] - 0.12, 0, 1)   # drop blue -> warmer
    lut = np.clip(lut - 0.04, 0, 1)                    # pull everything down -> darker
    return lut


# --- code histogram / collapse diagnostics ---------------------------------------
def test_code_histogram_neutral_is_zero_entropy():
    stats = code_histogram_stats([160] * 64)
    assert stats["unique_codes"] == 1
    assert stats["dominant_code"] == 160
    assert stats["dominant_share"] == 1.0
    assert stats["entropy_bits"] == 0.0
    assert stats["entropy_norm"] == 0.0


def test_code_histogram_diverse_is_high_entropy():
    stats = code_histogram_stats(list(range(64)))  # all distinct
    assert stats["unique_codes"] == 64
    assert stats["dominant_share"] == pytest.approx(1 / 64)
    assert stats["entropy_norm"] == pytest.approx(1.0)


# --- behavioral agreement (no LUT, hand-built spec + measured dict) ---------------
def test_agreement_counts_backed_axes():
    spec = AttributeSpec(route="grade", axes={"temperature_delta_b": 2.0, "mean_l_delta": -2.0})
    mb = {"temperature_delta_b": 2.0,   # right sign + magnitude -> backed
          "mean_l_delta": +2.0}          # wrong sign -> unbacked
    agree = behavioral_agreement(spec, mb)
    assert agree["axes_total"] == 2
    assert agree["axes_backed"] == 1
    assert agree["fidelity"] == pytest.approx(0.5)
    assert agree["per_axis"]["temperature_delta_b"]["backed"] is True
    assert agree["per_axis"]["mean_l_delta"]["backed"] is False


def test_agreement_none_when_no_measurable_axis():
    # A hue-only spec asserts nothing is_backed magnitude-checks -> fidelity undefined.
    spec = AttributeSpec(route="grade", axes={"global_hue_deg": 210.0})
    assert behavioral_agreement(spec, {"global_hue_deg": 210.0})["fidelity"] is None


# --- score_from_lut: self-consistency + collapse ---------------------------------
def test_self_consistent_spec_is_fully_backed():
    """A spec derived from a LUT's OWN measured behavior is fully backed by that LUT."""
    lut = _warm_dark_lut()
    spec = from_measured_behavior(measure_behavior(lut))
    rec = score_from_lut(lut, spec)
    assert rec["agreement"]["axes_total"] >= 1  # the test LUT must assert something
    assert rec["behavioral_fidelity"] == pytest.approx(1.0)
    assert rec["collapsed"] is False


def test_identity_lut_collapses_and_fails_a_real_spec():
    lut = cube_io.identity_grid(17)
    rec = score_from_lut(lut, "route=grade | warmer=+3.0 darker=+2.0")
    assert rec["collapsed"] is True
    assert rec["degenerate_identity"] is True
    assert rec["residual_norm"] == pytest.approx(0.0, abs=1e-9)
    assert rec["behavioral_fidelity"] == pytest.approx(0.0)


def test_refuse_spec_has_no_fidelity():
    rec = score_from_lut(_warm_dark_lut(), "route=refuse | refuse=out_of_scope")
    assert rec["route"] == "refuse"
    assert rec["behavioral_fidelity"] is None


def test_dominant_code_collapse_flagged_even_with_moderate_rms():
    """The greedy failure mode: a non-trivial residual but one code owning most positions."""
    lut = _warm_dark_lut()  # a real edit -> residual well above the collapse floor
    greedy_like = [8] * 48 + [124] * 16          # dominant_share 0.75 (the observed Phase-0 greedy)
    diverse = list(range(64))                    # dominant_share ~0.016
    assert score_from_lut(lut, "route=grade | warmer=+2.0", codes=greedy_like)["collapsed"] is True
    assert score_from_lut(lut, "route=grade | warmer=+2.0", codes=diverse)["collapsed"] is False


# --- decoded ΔE ------------------------------------------------------------------
def test_decoded_delta_e_zero_for_identical_and_positive_for_shifted():
    a = _warm_dark_lut()
    assert decoded_delta_e(a, a)["mean"] == pytest.approx(0.0, abs=1e-9)
    assert decoded_delta_e(a, cube_io.identity_grid(17))["mean"] > 1.0


# --- summary aggregation ---------------------------------------------------------
def test_summarize_fidelity_aggregates():
    recs = [
        {"behavioral_fidelity": 0.9, "residual_norm": 0.2, "collapsed": False,
         "degenerate_identity": False, "code_stats": {"entropy_norm": 0.8, "dominant_share": 0.1}},
        {"behavioral_fidelity": 0.2, "residual_norm": 0.001, "collapsed": True,
         "degenerate_identity": False, "code_stats": {"entropy_norm": 0.0, "dominant_share": 1.0}},
        {"behavioral_fidelity": None, "residual_norm": 0.15, "collapsed": False,
         "degenerate_identity": False},  # a refuse row: no fidelity
    ]
    s = summarize_fidelity(recs)
    assert s["rows"] == 3
    assert s["grade_rows"] == 2
    assert s["behavioral_fidelity_mean"] == pytest.approx(0.55)  # mean(0.9, 0.2)
    assert s["collapse_rate"] == pytest.approx(1 / 3)


# --- decode path (needs the frozen VQ weights; gitignored) -----------------------
_WEIGHTS = Path("tokenizer/final/model.pt").is_file()


@pytest.mark.skipif(not _WEIGHTS, reason="frozen VQ weights absent (staged-corpus only)")
def test_decode_neutral_code_collapses():
    pytest.importorskip("torch")
    from eval.behavioral_fidelity import score_generation

    rec = score_generation([160] * 64, "route=grade | warmer=+3.0", target_codes=[160] * 64)
    assert rec["collapsed"] is True
    assert rec["residual_norm"] < 0.01
    assert rec["code_stats"]["dominant_share"] == 1.0
    assert rec["decoded_delta_e"]["mean"] == pytest.approx(0.0, abs=1e-9)  # same codes
