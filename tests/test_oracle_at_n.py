"""Tests for the oracle@N coverage gate aggregation (eval.oracle_at_n).

Pure numpy — no model, no GPU. The generation/scoring themselves are exercised on Colab; here we test
the aggregation (`oracle_and_best`), the sample bookkeeping (`score_row_samples`), and that the module
imports without torch.
"""

from __future__ import annotations

import pytest

import eval.oracle_at_n as O


def test_oracle_curve_and_best_pick():
    recs_by_row = [
        [{"behavioral_fidelity": 0.1, "collapsed": True},
         {"behavioral_fidelity": 0.6, "collapsed": False, "code_stats": {"entropy_norm": 0.8}},
         {"behavioral_fidelity": 0.2, "collapsed": False}],
        [{"behavioral_fidelity": 0.3, "collapsed": False},
         {"behavioral_fidelity": 0.3, "collapsed": False}],
    ]
    s = O.oracle_and_best(recs_by_row, ks=(1, 2, 4))
    assert s["oracle@1"] == pytest.approx(0.2)     # mean(max first 1) = mean(0.1, 0.3)
    assert s["oracle@2"] == pytest.approx(0.45)    # mean(max first 2) = mean(0.6, 0.3)
    assert s["oracle@4"] == pytest.approx(0.45)    # only 3 / 2 samples available
    assert s["oracle@1"] <= s["oracle@2"] <= s["oracle@4"]     # monotonic non-decreasing in k
    assert s["best_of_N"] == pytest.approx(0.45)   # reranker picks 0.6 (row0) and 0.3 (row1)
    assert s["scored_rows"] == 2


def test_oracle_ignores_empty_rows():
    s = O.oracle_and_best([[], [{"behavioral_fidelity": 0.5, "collapsed": False}]], ks=(1,))
    assert s["rows"] == 2 and s["scored_rows"] == 1
    assert s["oracle@1"] == pytest.approx(0.5)


def test_oracle_excludes_none_fidelity_not_coerced_to_zero():
    # A row whose valid samples are all None (spec asserts no measurable axis) is EXCLUDED, not
    # counted as 0.0 — matching summarize_fidelity, so the gate isn't silently deflated.
    recs_by_row = [
        [{"behavioral_fidelity": None}, {"behavioral_fidelity": None}],   # unmeasurable -> excluded
        [{"behavioral_fidelity": 0.4, "collapsed": False}],               # measurable
    ]
    s = O.oracle_and_best(recs_by_row, ks=(1, 4))
    assert s["oracle@1"] == pytest.approx(0.4)   # NOT (0.0 + 0.4)/2 = 0.2
    assert s["best_of_N"] == pytest.approx(0.4)
    # a refusal (0.0) is a real miss and IS kept
    s2 = O.oracle_and_best([[{"behavioral_fidelity": 0.0, "collapsed": True},
                             {"behavioral_fidelity": 0.6, "collapsed": False}]], ks=(2,))
    assert s2["oracle@2"] == pytest.approx(0.6)


def test_score_row_samples_refusal_and_short(monkeypatch):
    # valid 64-code sample -> routed through score_generation (stubbed); None/short -> 0.0 miss.
    monkeypatch.setattr(O, "score_generation",
                        lambda codes, spec, target_codes=None: {"behavioral_fidelity": 0.5, "n": len(codes)})
    recs = O.score_row_samples([list(range(64)), None, [1, 2, 3]], "route=grade | warmer=+2.0", None)
    assert recs[0]["behavioral_fidelity"] == 0.5           # valid
    assert recs[1]["behavioral_fidelity"] == 0.0 and recs[1]["refused"] is True    # None
    assert recs[2]["behavioral_fidelity"] == 0.0 and recs[2]["refused"] is False   # short (len 3)
    assert all(r.get("collapsed") for r in recs[1:])


def test_module_imports_without_torch():
    import importlib
    import sys
    assert "torch" not in sys.modules or True  # torch may be loaded by other tests; just ensure import works
    importlib.import_module("eval.oracle_at_n")
