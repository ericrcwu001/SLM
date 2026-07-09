"""Known-answer tests for the statistics module."""

import math

from eval.stats import (
    NOT_EVALUABLE,
    holm_bonferroni,
    mcnemar,
    paired_delta_bootstrap,
    seed_summary,
    wilson_ci,
    wilson_gate,
)


def test_wilson_known_value():
    w = wilson_ci(8, 10, 0.95)
    assert math.isclose(w.point, 0.8)
    # published Wilson 95% for 8/10 ~ [0.4902, 0.9433]
    assert math.isclose(w.low, 0.4902, abs_tol=1e-3)
    assert math.isclose(w.high, 0.9433, abs_tol=1e-3)


def test_wilson_zero_n():
    w = wilson_ci(0, 0)
    assert w.point is None and w.low is None and w.high is None


def test_paired_delta_all_better():
    a = [1.0] * 20
    b = [0.0] * 20
    r = paired_delta_bootstrap(a, b, seed=0)
    assert math.isclose(r.delta, 1.0)
    assert r.ci_low == 1.0 and r.ci_high == 1.0


def test_paired_delta_reproducible():
    a = [1, 0, 1, 1, 0, 1, 0, 1, 1, 0]
    b = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    r1 = paired_delta_bootstrap(a, b, seed=123)
    r2 = paired_delta_bootstrap(a, b, seed=123)
    assert (r1.ci_low, r1.ci_high) == (r2.ci_low, r2.ci_high)
    assert math.isclose(r1.delta, (sum(a) - sum(b)) / len(a))


def test_mcnemar_balanced_is_p1():
    a = [1, 0, 1, 0]
    b = [0, 1, 0, 1]
    r = mcnemar(a, b)
    assert r.b == 2 and r.c == 2
    assert math.isclose(r.p_value, 1.0)


def test_mcnemar_lopsided():
    a = [1] * 9 + [0]
    b = [0] * 10
    r = mcnemar(a, b)
    assert r.b == 9 and r.c == 0
    assert math.isclose(r.p_value, 2 * 0.5 ** 9, rel_tol=1e-6)


def test_holm_bonferroni_stepdown():
    pv = {"a": 0.001, "b": 0.02, "c": 0.049, "d": 0.5}
    res = {r.name: r.reject for r in holm_bonferroni(pv, alpha=0.05)}
    # thresholds: 0.0125, 0.0167, 0.025, 0.05
    assert res["a"] is True     # 0.001 <= 0.0125
    assert res["b"] is False    # 0.02  > 0.0167 -> stop
    assert res["c"] is False
    assert res["d"] is False


def test_seed_summary():
    s = seed_summary("m", [0.8, 0.9, 1.0])
    assert s.seed_count == 3
    assert math.isclose(s.mean, 0.9)
    assert s.min == 0.8 and s.max == 1.0 and s.median == 0.9


def test_wilson_gate_min_n_gating():
    # n below min_N -> not evaluable
    g = wilson_gate(8, 10, bound="lower", threshold=0.85, min_n=800, name="m")
    assert g.status == NOT_EVALUABLE
    # n at/above min_N, lower bound below threshold -> fail
    g2 = wilson_gate(80, 100, bound="lower", threshold=0.85, min_n=50, name="m")
    assert g2.status == "fail"
    # clearly clears
    g3 = wilson_gate(99, 100, bound="lower", threshold=0.85, min_n=50, name="m")
    assert g3.status == "pass"


def test_wilson_gate_upper_bound():
    # over-refusal style: upper bound must be <= threshold
    g = wilson_gate(2, 100, bound="upper", threshold=0.10, min_n=50, name="orr")
    assert g.status == "pass"
    g2 = wilson_gate(15, 100, bound="upper", threshold=0.10, min_n=50, name="orr")
    assert g2.status == "fail"
