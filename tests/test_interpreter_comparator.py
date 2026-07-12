"""Spec-vs-spec comparator — including the two adversarial-review blockers (None fidelity, parse)."""

from __future__ import annotations

from data_pipeline.attribute_spec import AttributeSpec, serialize
from interpreter.comparator import compare_specs, joint_score

WARM_DARK = serialize(AttributeSpec(route="grade",
                                    axes={"temperature_delta_b": 3.0, "mean_l_delta": -2.0}))
COOL = serialize(AttributeSpec(route="grade", axes={"temperature_delta_b": -3.0}))
EMPTY_GRADE = serialize(AttributeSpec(route="grade"))
REFUSE_SCOPE = serialize(AttributeSpec(route="refuse", refuse_reason="out_of_scope"))
REFUSE_GAMUT = serialize(AttributeSpec(route="refuse", refuse_reason="out_of_gamut"))
CLARIFY = serialize(AttributeSpec(route="clarify"))


def test_identical_grade_is_perfect():
    c = compare_specs(WARM_DARK, WARM_DARK)
    assert c["parse_ok"] and c["route_correct"] and c["attribute_f1"] == 1.0
    assert joint_score(c) == 1.0


def test_opposite_sign_axis_tanks_f1():
    c = compare_specs(COOL, serialize(AttributeSpec(route="grade", axes={"temperature_delta_b": 3.0})))
    assert c["route_correct"] and c["attribute_f1"] == 0.0  # sign mismatch -> unbacked both ways


def test_empty_pred_vs_empty_gold_is_perfect():
    # BLOCKER 1: behavioral_agreement returns fidelity=None here; must NOT crash, must score 1.0.
    c = compare_specs(EMPTY_GRADE, EMPTY_GRADE)
    assert c["attribute_f1"] == 1.0 and joint_score(c) == 1.0


def test_empty_pred_vs_nonempty_gold_zero_recall():
    c = compare_specs(EMPTY_GRADE, WARM_DARK)
    assert c["precision"] == 1.0 and c["recall"] == 0.0 and c["attribute_f1"] == 0.0


def test_malformed_pred_is_a_hard_miss():
    # BLOCKER 2a: parse() raises on a bad float -> must be a miss, not a crash.
    c = compare_specs("route=grade | warmer=abc", WARM_DARK)
    assert not c["parse_ok"] and not c["route_correct"] and c["attribute_f1"] == 0.0
    assert joint_score(c) == 0.0


def test_gibberish_without_route_is_not_a_silent_grade():
    # BLOCKER 2b: parse() defaults gibberish to route=grade{}; the guard must reject it.
    c = compare_specs("total garbage no equals sign", WARM_DARK)
    assert not c["parse_ok"] and c["route_pred"] is None and not c["route_correct"]


def test_route_mismatch_grade_vs_refuse():
    c = compare_specs(WARM_DARK, REFUSE_SCOPE)
    assert not c["route_correct"] and joint_score(c) == 0.0


def test_refuse_kind_tracked_separately():
    same = compare_specs(REFUSE_SCOPE, REFUSE_SCOPE)
    assert same["route_correct"] and same["refuse_kind_correct"] and joint_score(same) == 1.0
    wrong_kind = compare_specs(REFUSE_GAMUT, REFUSE_SCOPE)
    assert wrong_kind["route_correct"]           # both 'refuse'
    assert not wrong_kind["refuse_kind_correct"]  # gamut vs scope
    assert joint_score(wrong_kind) == 1.0         # route-correct refuse -> joint 1.0 (kind reported apart)


def test_direction_f1_separates_direction_from_magnitude():
    # Right DIRECTION (warmer), wrong MAGNITUDE (+1.0 vs +3.0): attribute_f1 fails on tol,
    # but direction_f1 credits the correct sign.
    gold = serialize(AttributeSpec(route="grade", axes={"temperature_delta_b": 3.0}))
    pred = serialize(AttributeSpec(route="grade", axes={"temperature_delta_b": 1.0}))
    c = compare_specs(gold, pred)
    assert c["attribute_f1"] == 0.0        # |3-1|=2 > tol and > 25% of 3
    assert c["direction_f1"] == 1.0        # same sign -> direction correct
    # opposite sign -> both zero
    wrong = compare_specs(serialize(AttributeSpec(route="grade", axes={"temperature_delta_b": -1.0})), gold)
    assert wrong["direction_f1"] == 0.0
    # refuse/clarify -> direction undefined
    assert compare_specs(REFUSE_SCOPE, REFUSE_SCOPE)["direction_f1"] is None


def test_clarify_route_only():
    c = compare_specs(CLARIFY, CLARIFY)
    assert c["route_correct"] and c["attribute_f1"] is None and joint_score(c) == 1.0
    miss = compare_specs(WARM_DARK, CLARIFY)
    assert not miss["route_correct"] and joint_score(miss) == 0.0
