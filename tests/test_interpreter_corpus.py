"""Interpreter corpus unification + the leakage-fix regression (ADR 0021/0024)."""

from __future__ import annotations

import importlib

from data_pipeline.attribute_spec import parse

bic = importlib.import_module("scripts.build_interpreter_corpus")


def _caption(lut, style, target="route=grade | warmer=+2.0"):
    return {"id": f"cap_{lut}_{style}", "source_lut_id": lut, "caption": f"{style} phrasing",
            "style": style, "route": "grade", "attribute_spec_text": target}


def _active_supported(lut, unit):
    return {"id": lut, "source_lut_id": lut, "is_supported": True, "split_unit_id": unit}


def test_all_styles_of_one_lut_share_one_unit():
    # THE leakage regression: 5 style-captions of one LUT must collapse to a single split unit,
    # so they get one holdout decision (not 5 independent coin-flips).
    styles = ["literal", "metaphor", "mood", "concept", "slang"]
    caption_rows = [_caption("lut_a", s) for s in styles]
    active = [_active_supported("lut_a", "unit_XYZ")]
    rows, stats = bic.build_rows(caption_rows, active, [])
    grade = [r for r in rows if r["route"] == "grade"]
    assert len(grade) == 5
    assert {r["split_unit_id"] for r in grade} == {"unit_XYZ"}  # ONE unit for all styles
    assert stats["dropped_missing_unit"] == 0 and stats["fallback_key_count"] == 0
    assert all(r["source_family"] == "caption" for r in grade)


def test_caption_without_matching_active_lut_is_dropped():
    caption_rows = [_caption("orphan_lut", "literal")]
    rows, stats = bic.build_rows(caption_rows, [], [])
    assert rows == [] and stats["dropped_missing_unit"] == 1


def test_out_of_scope_from_active_rows():
    active = [{"id": "unsup_train_000001", "is_supported": False, "route": "refuse",
               "refuse_kind": "out_of_scope", "instruction_natural": "remove the background",
               "split_unit_id": "unsup:img1", "source_family": "unsupported_teacher"}]
    rows, stats = bic.build_rows([], active, [])
    assert len(rows) == 1
    r = rows[0]
    assert r["text"] == "remove the background" and r["split_unit_id"] == "unsup:img1"
    spec = parse(r["attribute_spec_text"])
    assert spec.route == "refuse" and spec.refuse_reason == "out_of_scope"


def test_supplement_clarify_and_gamut_targets():
    supp = [
        {"id": "unsup_clarify_000001", "route": "clarify", "refuse_kind": None,
         "instruction_natural": "make it look nicer", "split_unit_id": "unsup:unsup_clarify_000001"},
        {"id": "unsup_gamut_000001", "route": "refuse", "refuse_kind": "out_of_gamut",
         "instruction_natural": "make it infrared false color", "split_unit_id": "unsup:unsup_gamut_000001"},
    ]
    rows, stats = bic.build_rows([], [], supp)
    by_id = {r["id"]: r for r in rows}
    clarify = parse(by_id["unsup_clarify_000001"]["attribute_spec_text"])
    gamut = parse(by_id["unsup_gamut_000001"]["attribute_spec_text"])
    assert clarify.route == "clarify"
    assert gamut.route == "refuse" and gamut.refuse_reason == "out_of_gamut"
    assert stats["by_route"] == {"clarify": 1, "refuse": 1}


def test_full_three_way_stats():
    caption_rows = [_caption("lut_a", s) for s in ("literal", "slang")]
    active = [_active_supported("lut_a", "unit_A"),
              {"id": "unsup_train_1", "is_supported": False, "route": "refuse",
               "refuse_kind": "out_of_scope", "instruction_natural": "erase the sign",
               "split_unit_id": "unsup:i1"}]
    supp = [{"id": "unsup_clarify_1", "route": "clarify", "refuse_kind": None,
             "instruction_natural": "fix the colors", "split_unit_id": "unsup:c1"}]
    rows, stats = bic.build_rows(caption_rows, active, supp)
    assert stats["by_route"] == {"grade": 2, "refuse": 1, "clarify": 1}
    assert stats["fallback_key_count"] == 0
    assert stats["units_per_route"]["grade"] == 1  # both captions → one LUT unit
