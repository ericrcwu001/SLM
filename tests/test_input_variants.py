"""Tests for the Phase-2 input levers: hybrid conditioning + bucketized ordinal magnitudes.

Verifies the bucketized rendering is input-only and lossy (no decimals, ordinal words), that the
canonical ``serialize``/``parse`` round-trip is untouched, that ``input_text_for`` produces the
hybrid/bucketized text, and that ``SFTConfig`` accepts the new knobs.
"""

from __future__ import annotations

import pytest

import random

from data_pipeline.attribute_spec import (
    augment_spec,
    from_measured_behavior,
    ground_truth_attribute_spec_text,
    parse,
    serialize,
    serialize_bucketed,
    shuffle_axis_order,
)
from sft.config import SFTConfig
from sft.example import input_text_for

_MB = {
    "temperature_delta_b": 1.0,     # slight, warmer
    "chroma_delta": -2.0,           # moderate, muted
    "matte_strength": 8.0,          # extreme (unipolar)
    "global_hue_deg": 210.0,
    "global_hue_magnitude": 5.0,    # gate open -> global_hue emitted
    "per_hue_saturation": {"green": -2.0},
}


def _spec():
    return from_measured_behavior(_MB)


# --- bucketized rendering --------------------------------------------------------
def test_bucketized_uses_ordinals_and_no_decimals():
    out = serialize_bucketed(_spec())
    assert "warmer=slight" in out
    assert "muted=moderate" in out
    assert "matte=extreme" in out
    assert "global_hue=210" in out       # hue stays integer degrees
    assert "sat_green=-moderate" in out   # per-hue sat keeps a sign, buckets the magnitude
    assert "." not in out                 # no shredded floats anywhere


def test_bucket_boundaries():
    # 1.5 -> moderate (not slight), 3.0 -> strong, 6.0 -> extreme (upper-inclusive bands)
    assert "warmer=moderate" in serialize_bucketed(from_measured_behavior({"temperature_delta_b": 1.5}))
    assert "warmer=strong" in serialize_bucketed(from_measured_behavior({"temperature_delta_b": 3.0}))
    assert "warmer=extreme" in serialize_bucketed(from_measured_behavior({"temperature_delta_b": 6.0}))


def test_canonical_roundtrip_unaffected():
    spec = _spec()
    assert parse(serialize(spec)) == spec   # bucketize must not perturb the canonical seam


# --- ground-truth spec text ------------------------------------------------------
def test_ground_truth_bucketize_flag():
    row = {"is_supported": True, "measured_behavior": _MB}
    assert ground_truth_attribute_spec_text(row) == serialize(_spec())
    assert ground_truth_attribute_spec_text(row, bucketize=True) == serialize_bucketed(_spec())


# --- input_text_for --------------------------------------------------------------
def _row():
    return {"id": "r1", "instruction": "make it warmer", "is_supported": True, "measured_behavior": _MB}


def test_input_text_instruction_and_spec_hybrid():
    row = _row()
    txt = input_text_for(row, "instruction_and_spec")
    assert txt.startswith("make it warmer\n")
    assert txt.endswith(serialize(_spec()))


def test_input_text_bucketized_paths():
    row = _row()
    assert input_text_for(row, "attribute_spec_text", bucketize=True) == serialize_bucketed(_spec())
    hybrid = input_text_for(row, "instruction_and_spec", bucketize=True)
    assert "make it warmer" in hybrid and "warmer=slight" in hybrid and "." not in hybrid


def test_input_text_prefers_prestamped_only_when_canonical():
    row = _row()
    row["attribute_spec_text"] = "route=grade | warmer=+9.9"   # a pre-stamped canonical spec
    assert input_text_for(row, "attribute_spec_text") == "route=grade | warmer=+9.9"
    # bucketized path re-renders from measured behavior, ignoring the pre-stamped float spec
    assert input_text_for(row, "attribute_spec_text", bucketize=True) == serialize_bucketed(_spec())


# --- spec augmentation (train-only input smoothing) ------------------------------
def test_augment_preserves_sign_and_bounds():
    spec = from_measured_behavior({"temperature_delta_b": 2.0, "chroma_delta": -3.0, "matte_strength": 5.0})
    aug = augment_spec(spec, random.Random(0), jitter=0.3)
    assert aug.axes["temperature_delta_b"] > 0        # sign preserved
    assert aug.axes["chroma_delta"] < 0
    assert abs(abs(aug.axes["temperature_delta_b"]) - 2.0) <= 0.4   # within jitter (+ rounding)


def test_augment_is_deterministic_per_seed():
    spec = from_measured_behavior({"temperature_delta_b": 2.0})
    assert augment_spec(spec, random.Random(7)) == augment_spec(spec, random.Random(7))


def test_shuffle_axis_order_reparses_to_same_spec():
    spec = from_measured_behavior({"temperature_delta_b": 2.0, "chroma_delta": -3.0, "matte_strength": 5.0})
    text = serialize(spec)
    shuffled = shuffle_axis_order(text, random.Random(1))
    assert parse(shuffled) == spec                    # parse is order-insensitive
    assert set(shuffled.split(" | ")[1].split()) == set(text.split(" | ")[1].split())


# --- config ----------------------------------------------------------------------
def test_config_accepts_new_knobs():
    assert SFTConfig().spec_bucketize is False
    assert SFTConfig().soft_label_weight == 0.0
    assert SFTConfig().spec_augment is False
    SFTConfig(input_field="instruction_and_spec")            # no raise
    SFTConfig(input_field="attribute_spec_text", spec_bucketize=True)
    SFTConfig(soft_label_weight=0.5, soft_label_tau=0.7, spec_augment=True)
    with pytest.raises(ValueError):
        SFTConfig(input_field="bogus")
    with pytest.raises(ValueError):
        SFTConfig(soft_label_weight=-1.0)
    with pytest.raises(ValueError):
        SFTConfig(soft_label_tau=0.0)
