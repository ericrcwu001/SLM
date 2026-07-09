"""Known-answer tests for boundary / unsupported metrics."""

import math

from eval.unsupported_metrics import DecisionRecord, compute_unsupported_metrics


def _rec(i, is_sup, kind, mixed=False, pair=None, syntax=True):
    return DecisionRecord(id=f"r{i}", is_supported=is_sup, kind=kind,
                          syntax_pass=syntax, mixed_prompt=mixed, boundary_pair_id=pair)


def test_hand_built_confusion():
    # 4 gold-supported: 3 emit lut (correct non-refusal), 1 refuses (over-refusal)
    # 4 gold-unsupported: 3 refuse (correct), 1 emits lut (false support)
    recs = [
        _rec(1, True, "lut_tokens"),
        _rec(2, True, "lut_tokens"),
        _rec(3, True, "lut_tokens"),
        _rec(4, True, "unsupported"),        # over-refusal
        _rec(5, False, "unsupported"),
        _rec(6, False, "unsupported"),
        _rec(7, False, "unsupported"),
        _rec(8, False, "lut_tokens"),        # false support
    ]
    out = compute_unsupported_metrics(recs)
    m, s, c = out["metrics"], out["scalars"], out["confusion"]

    assert c == {"tp": 3, "fp": 1, "fn": 1, "tn": 3}
    assert m["unsupported_recall"].rate == 3 / 4       # 3 correct refusals / 4 gold unsup
    assert m["over_refusal_rate"].rate == 1 / 4        # 1 refusal / 4 gold sup
    assert m["supported_coverage"].rate == 3 / 4
    assert m["false_support_rate"].rate == 1 / 4
    assert m["boundary_accuracy"].rate == 6 / 8        # tp+tn / all
    # precision = tp/(tp+fp) = 3/4 ; recall = 3/4 ; f1 = 3/4
    assert math.isclose(s["unsupported_precision"], 0.75)
    assert math.isclose(s["unsupported_recall"], 0.75)
    assert math.isclose(s["boundary_f1"], 0.75)


def test_mixed_recall():
    recs = [
        _rec(1, False, "unsupported", mixed=True),   # correct mixed refusal
        _rec(2, False, "lut_tokens", mixed=True),     # mixed false support
        _rec(3, False, "unsupported", mixed=False),   # non-mixed unsup (ignored by mixed metric)
    ]
    m = compute_unsupported_metrics(recs)["metrics"]
    assert m["mixed_unsupported_recall"].n == 2
    assert m["mixed_unsupported_recall"].rate == 0.5


def test_near_boundary_pair_accuracy():
    # pair A: both correct; pair B: one wrong
    recs = [
        _rec(1, True, "lut_tokens", pair="A"),        # correct non-refusal
        _rec(2, False, "unsupported", pair="A"),       # correct refusal -> pair A correct
        _rec(3, True, "unsupported", pair="B"),        # over-refusal -> pair B wrong
        _rec(4, False, "unsupported", pair="B"),
    ]
    m = compute_unsupported_metrics(recs)["metrics"]
    assert m["near_boundary_pair_accuracy"].n == 2      # 2 pairs
    assert m["near_boundary_pair_accuracy"].rate == 0.5


def test_selective_risk_uses_syntax_only():
    # supported non-refusals: one valid (syntax ok), one invalid (syntax fail)
    recs = [
        _rec(1, True, "lut_tokens", syntax=True),
        _rec(2, True, "invalid", syntax=False),
    ]
    m = compute_unsupported_metrics(recs)["metrics"]
    # reported under the explicit *_syntax_only key while L2-L7 are disabled
    assert "selective_risk" not in m
    assert m["selective_risk_syntax_only"].n == 2
    assert m["selective_risk_syntax_only"].rate == 0.5


def test_incomplete_boundary_pair_not_counted_correct():
    # a pair missing its unsupported member must NOT score as correct
    recs = [
        _rec(1, True, "lut_tokens", pair="solo"),   # only the supported side present
    ]
    m = compute_unsupported_metrics(recs)["metrics"]
    assert m["near_boundary_pair_accuracy"].n == 1
    assert m["near_boundary_pair_accuracy"].rate == 0.0


def test_all_refuse_precision_and_recall():
    recs = [_rec(1, False, "unsupported"), _rec(2, True, "unsupported")]
    out = compute_unsupported_metrics(recs)
    # precision = tp/(tp+fp) = 1/2 ; recall = 1/1
    assert out["scalars"]["unsupported_precision"] == 0.5
    assert out["scalars"]["unsupported_recall"] == 1.0
