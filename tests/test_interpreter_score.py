"""Pure aggregation of interpreter scoring (summarize_interpreter), no model/GPU."""

from __future__ import annotations

from interpreter.score import summarize_interpreter


def _rec(route_gold, route_pred, *, f1=None, refuse_kind_correct=True, unit="u", lut=None,
         style=None, parse_ok=True):
    route_correct = route_gold == route_pred
    joint = 0.0 if not route_correct else (1.0 if f1 is None else f1)
    return {"route_gold": route_gold, "route_pred": route_pred, "route_correct": route_correct,
            "refuse_kind_correct": refuse_kind_correct, "attribute_f1": f1, "joint": joint,
            "split_unit_id": unit, "source_lut_id": lut, "style": style, "parse_ok": parse_ok}


def test_metric_is_unit_macro_mean_of_joints():
    # unit A: three grade caption rows joints [1,1,0] -> unit mean 0.667; unit B: one refuse -> 1.0.
    recs = [
        _rec("grade", "grade", f1=1.0, unit="A", lut="lutA", style="literal"),
        _rec("grade", "grade", f1=1.0, unit="A", lut="lutA", style="mood"),
        _rec("grade", "grade", f1=0.0, unit="A", lut="lutA", style="slang"),
        _rec("refuse", "refuse", unit="B", lut=None),
    ]
    s = summarize_interpreter(recs)
    assert s["n"] == 4 and s["n_units"] == 2
    assert abs(s["metric"] - ((2 / 3) + 1.0) / 2) < 1e-9  # equal weight per unit, NOT per row


def test_over_refusal_and_route_recall():
    recs = [
        _rec("grade", "refuse", unit="A", lut="lutA"),   # over-refusal (grade -> refuse)
        _rec("grade", "grade", f1=0.5, unit="B", lut="lutB"),
        _rec("refuse", "refuse", unit="C"),
        _rec("clarify", "grade", unit="D"),              # clarify missed
    ]
    s = summarize_interpreter(recs)
    assert s["interpreter_over_refusal_rate"]["n"] == 2   # two grade golds
    assert s["interpreter_over_refusal_rate"]["rate"] == 0.5
    assert s["per_route_recall"]["refuse"]["recall"] == 1.0
    assert s["per_route_recall"]["clarify"]["recall"] == 0.0


def test_refuse_kind_accuracy_tracks_wrong_kind():
    recs = [
        _rec("refuse", "refuse", refuse_kind_correct=True, unit="A"),
        _rec("refuse", "refuse", refuse_kind_correct=False, unit="B"),  # right route, wrong kind
    ]
    s = summarize_interpreter(recs)
    assert s["refuse_kind_accuracy"]["accuracy"] == 0.5 and s["refuse_kind_accuracy"]["n"] == 2


def test_attribute_f1_split_real_vs_procedural():
    recs = [
        _rec("grade", "grade", f1=1.0, unit="A", lut="proc_attr_warmer_m0"),   # procedural
        _rec("grade", "grade", f1=0.0, unit="B", lut="fivek_1234"),            # real LUT
    ]
    s = summarize_interpreter(recs)
    assert s["attribute_f1"]["procedural"]["mean"] == 1.0
    assert s["attribute_f1"]["real_lut"]["mean"] == 0.0
    assert s["attribute_f1"]["overall"]["mean"] == 0.5


def test_empty_records():
    assert summarize_interpreter([])["metric"] is None
