"""AttributeSpec schema / serializer / parser / backing gate (ADR 0021; attribute_spec.md)."""

from __future__ import annotations

import pytest

from data_pipeline import attribute_spec as A
from data_pipeline.behavior_vector import measure_behavior
from data_pipeline.sources import procedural as proc
from eval.cube_io import identity_grid


def _lut(name):
    return proc.generate_lut_tensor(next(s for s in proc.catalog() if s.lut_id == name))


@pytest.mark.parametrize("name", [
    "proc_style_teal-orange", "proc_attr_warmer", "proc_style_matte", "proc_attr_muted",
    "proc_attr_more_contrast",
])
def test_round_trip_identity_from_measured_behavior(name):
    mb = measure_behavior(_lut(name))
    spec = A.from_measured_behavior(mb, confidence=0.82)
    text = A.serialize(spec)
    spec2 = A.parse(text)
    # serialize -> parse is identity on the canonical spec (attribute_spec.md §7)
    assert spec2.axes == spec.axes
    assert spec2.sat == spec.sat
    assert spec2.route == spec.route
    # and serialize is idempotent (deterministic canonical text)
    assert A.serialize(spec2) == text


def test_bipolar_tag_encodes_direction_positive_magnitude():
    # A muted (negative chroma) look serializes with the `muted` tag and a POSITIVE magnitude.
    spec = A.AttributeSpec(axes={"chroma_delta": -4.8})
    text = A.serialize(spec)
    assert "muted=+4.8" in text and "more_saturated" not in text
    # ... and parses back to the signed axis value.
    assert A.parse(text).axes["chroma_delta"] == -4.8
    warm = A.parse("route=grade | warmer=+2.3")
    assert warm.axes["temperature_delta_b"] == 2.3


def test_omit_below_threshold_and_determinism():
    mb = {"temperature_delta_b": 0.2, "chroma_delta": -3.0}   # temp below _MAG_EPS -> omitted
    spec = A.from_measured_behavior(mb)
    text = A.serialize(spec)
    assert "warmer" not in text and "cooler" not in text
    assert "muted=+3.0" in text
    assert A.serialize(A.from_measured_behavior(mb)) == text  # deterministic


def test_identity_lut_is_empty_grade():
    spec = A.from_measured_behavior(measure_behavior(identity_grid(17)))
    text = A.serialize(spec)
    assert text.startswith("route=grade")
    assert spec.axes == {} and spec.sat == {}


def test_refuse_and_clarify_routes():
    r = A.AttributeSpec(route="refuse", refuse_reason="out_of_gamut")
    assert A.parse(A.serialize(r)).refuse_reason == "out_of_gamut"
    assert A.parse(A.serialize(r)).route == "refuse"
    c = A.AttributeSpec(route="clarify", confidence=0.3)
    assert A.parse(A.serialize(c)).route == "clarify"
    with pytest.raises(ValueError):
        A.AttributeSpec(route="not_a_route")


def test_backing_gate():
    mb = measure_behavior(_lut("proc_attr_warmer"))
    spec = A.from_measured_behavior(mb)
    ok, issues = A.is_backed(spec, mb)
    assert ok, issues
    # a spec asserting the OPPOSITE sign of a measured axis is not backed
    bad = A.AttributeSpec(axes={"temperature_delta_b": -3.0})   # measured is warm (+)
    ok2, issues2 = A.is_backed(bad, mb)
    assert not ok2 and any("unbacked_sign" in i for i in issues2)


def test_backing_gate_per_hue_saturation_magnitude():
    # per-hue saturation must be backed by BOTH sign AND magnitude (same as main axes).
    mb = {"per_hue_saturation": {"red": 2.0}}
    # correct sign, within tolerance -> backed
    ok, issues = A.is_backed(A.AttributeSpec(sat={"red": 3.0}), mb)
    assert ok, issues
    # correct sign but wildly over-claimed magnitude (25x) -> unbacked
    ok2, issues2 = A.is_backed(A.AttributeSpec(sat={"red": 50.0}), mb)
    assert not ok2 and any("unbacked_sat_magnitude" in i for i in issues2)
    # opposite sign is still flagged as a sign issue, not a magnitude one
    ok3, issues3 = A.is_backed(A.AttributeSpec(sat={"red": -3.0}), mb)
    assert not ok3 and any("unbacked_sat_sign" in i for i in issues3)


def test_measured_behavior_to_text_matches_pipeline_field():
    mb = measure_behavior(_lut("proc_attr_warmer"))
    text = A.measured_behavior_to_text(mb, confidence=0.9)
    assert text.startswith("route=grade")
    assert text.endswith("conf=0.90")
    assert "warmer=+" in text


def test_ground_truth_attribute_spec_text_grade_and_refuse():
    # supported row -> grade spec from measured behavior
    mb = measure_behavior(_lut("proc_attr_warmer"))
    grade = A.ground_truth_attribute_spec_text({"is_supported": True, "measured_behavior": mb})
    assert grade.startswith("route=grade") and "warmer=+" in grade
    # refuse row -> refuse spec carrying its kind (no LUT needed)
    oog = A.ground_truth_attribute_spec_text(
        {"is_supported": False, "refuse_kind": "out_of_gamut"})
    assert oog == "route=refuse | refuse=out_of_gamut"
    # refuse row w/o an explicit kind defaults to out_of_scope
    oos = A.ground_truth_attribute_spec_text({"is_supported": False})
    assert oos == "route=refuse | refuse=out_of_scope"
