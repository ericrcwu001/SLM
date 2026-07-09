"""Tests for the teacher/judge interaction eval (eval/instruction_quality.py)."""

from eval import instruction_quality as iq
from eval.instruction_quality import (
    agreement_matrix,
    deterministic_catches,
    run_catch_rate,
    synthesize_negatives,
)


class _Verdict:
    def __init__(self, status):
        self.status = status


# --- A. agreement matrix ----------------------------------------------------------
def test_agreement_matrix_counts():
    rows = [
        {"id": "a", "validation_ok": True, "judge": {"status": "pass"}},        # agree accept
        {"id": "b", "validation_ok": True, "judge": {"status": "not_evaluated"}},
        {"id": "c", "validation_ok": False, "judge": {"status": "pass"}},        # judge missed
        {"id": "d", "validation_ok": True, "judge": {"status": "fail"}},         # judge-only flag
        {"id": "e", "validation_ok": False, "judge": {"status": "fail"}},        # agree reject
        {"id": "f", "instruction_status": "source_authored"},                    # skipped
        {"id": "g", "validation_ok": True, "judge": None},                       # agree accept (judge off)
    ]
    m = agreement_matrix(rows)
    assert m["authored_skipped"] == 1
    assert m["agree_accept"] == 2                 # a + g
    assert m["agree_reject"] == 1                 # e
    assert m["judge_missed_measurable"] == 1      # c
    assert m["judge_only_flag"] == 1              # d
    assert m["judge_not_evaluated"] == 1          # b
    assert m["disagreement_examples"]["judge_missed_measurable"] == ["c"]
    assert m["disagreement_examples"]["judge_only_flag"] == ["d"]


# --- C. synthetic negatives -------------------------------------------------------
def test_clean_seeds_are_not_rejected():
    for s in iq.DEFAULT_SEEDS:
        assert not deterministic_catches(s), s["id"]


def test_direction_flip_caught_deterministically_language_defect_not():
    negs = synthesize_negatives()
    wrong = [n for n in negs if n["defect"] == "wrong_direction"]
    assert wrong
    for n in wrong:
        assert deterministic_catches(n["record"]), n["record"]
    # a pure language defect leaves the tags/behavior intact, so the deterministic gate can't see it
    for n in (n for n in negs if n["defect"] == "local_edit"):
        assert not deterministic_catches(n["record"])


def test_catch_rate_with_fake_judge():
    negs = synthesize_negatives()
    markers = ("blur", "background", "sharpen", "exactly the same", "most beautiful", "best possible")
    dirw = {"warm", "warmer", "cool", "cooler", "bright", "brighter", "dark", "darker",
            "muted", "contrast", "saturat"}

    def toks(s):
        s = (s or "").lower()
        return {w for w in dirw if w in s}

    def fake_judge(rec):
        text = (rec.get("concise", "") + " " + (rec.get("natural", "") or "")).lower()
        if any(m in text for m in markers):
            return _Verdict("fail")
        if rec.get("natural") and not (toks(rec["concise"]) & toks(rec["natural"])):
            return _Verdict("fail")     # concise/natural describe different edits
        return _Verdict("pass")

    rates = run_catch_rate(negs, judge_fn=fake_judge)
    for defect in ("wrong_direction", "local_edit", "impossible_preservation",
                   "aesthetic_ranking", "concise_natural_divergence"):
        assert rates[defect]["catch_rate"] == 1.0, (defect, rates[defect])
    assert rates["_overall"]["catch_rate"] == 1.0
    # direction flips are caught by the deterministic gate, not the judge
    assert rates["wrong_direction"]["by_det"] == rates["wrong_direction"]["n"]
    # language defects are the judge's job (deterministic gate blind to them)
    assert rates["local_edit"]["by_det"] == 0


def test_catch_rate_deterministic_only_floor():
    # No judge -> only the direction flip is catchable; language defects slip through.
    rates = run_catch_rate(synthesize_negatives(), judge_fn=None)
    assert rates["wrong_direction"]["catch_rate"] == 1.0
    assert rates["local_edit"]["catch_rate"] == 0.0
