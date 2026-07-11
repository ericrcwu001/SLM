"""Tests for best-of-N reranking (eval.best_of_n) — pure orchestration, no model/GPU.

`generate_codes_batch` and `score_generation` are monkeypatched; we test that the reranker returns the
highest-fidelity VALID candidate, skips refusals/malformed samples, and falls back when all refuse.
"""

from __future__ import annotations

import eval.best_of_n as B
import sft.generate as G


def test_best_of_n_picks_highest_valid(monkeypatch):
    # candidates: valid(64), None(refusal), short(3), valid(64) — only the two 64-len ones are scored
    monkeypatch.setattr(G, "generate_codes_batch",
                        lambda *a, **k: [list(range(64)), None, [1, 2, 3], [i % 256 for i in range(64)]])
    fids = iter([0.2, 0.7])   # first valid -> 0.2, second valid -> 0.7
    monkeypatch.setattr(B, "score_generation",
                        lambda codes, spec, **k: {"behavioral_fidelity": next(fids), "collapsed": False})
    codes, rec = B.best_of_n_codes(None, None, image="img", cond_text="route=grade | warmer=+2.0", n=4)
    assert rec["behavioral_fidelity"] == 0.7      # reranker picks the higher-fidelity valid candidate
    assert codes is not None and len(codes) == 64


def test_best_of_n_all_refused_fallback(monkeypatch):
    monkeypatch.setattr(G, "generate_codes_batch", lambda *a, **k: [None, None, [1, 2, 3]])
    codes, rec = B.best_of_n_codes(None, None, image="img", cond_text="s", n=3)
    assert codes is None and rec.get("refused_all") is True


def test_best_of_n_defaults_spec_to_cond(monkeypatch):
    seen = {}
    monkeypatch.setattr(G, "generate_codes_batch", lambda *a, **k: [list(range(64))])

    def _score(codes, spec, **k):
        seen["spec"] = spec
        return {"behavioral_fidelity": 0.5, "collapsed": False}

    monkeypatch.setattr(B, "score_generation", _score)
    B.best_of_n_codes(None, None, image="img", cond_text="COND", n=1)   # no spec_text -> defaults to cond
    assert seen["spec"] == "COND"
