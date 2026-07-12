"""Reward-shaping + group-advantage tests (eval.grpo_reward; docs/grpo/01_reward.md §8).

Pure numpy — no GPU, no frozen weights. The refusal/None accounting is verified by SPYING on
``score_batch`` (so we can assert malformed samples never reach the decoder). The end-to-end parity
vs the canonical ruler (needs the frozen VQ weights) lives in ``tests/test_fast_reward.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.grpo_reward import group_advantages, shaped_rewards

_SPEC = "route=grade | warmer=+2.3 muted=+2.0 matte=+2.5 contrast=-1.0"


def _spy(records):
    """A fake score_batch that records the codes it was handed and returns canned ``records``."""
    seen = []

    def fn(codes_batch, spec, *, device=None, **kw):
        seen.append([list(c) for c in codes_batch])
        assert len(records) == len(codes_batch), "spy: record/codes length mismatch"
        return records

    fn.seen = seen
    return fn


# --- refusal / malformed accounting (must NOT hit the decoder) ------------------------------------
def test_refusal_and_non64_short_circuit_before_decode(monkeypatch):
    valid = list(range(64))
    spy = _spy([{"behavioral_fidelity": 0.5, "collapsed": False}])   # only the 1 valid sample decoded
    monkeypatch.setattr("eval.grpo_reward.score_batch", spy)

    out = shaped_rewards([None, [0] * 63, valid], _SPEC)

    assert spy.seen == [[valid]], "malformed/refusal codes must never reach score_batch"
    assert out[0][0] == 0.0 and out[0][1].get("refused") is True      # refusal -> 0
    assert out[1][0] == 0.0 and out[1][1].get("refused") is True      # len-63 -> 0
    assert out[2][0] == pytest.approx(0.5)                            # valid -> fidelity
    assert out[2][1]["behavioral_fidelity"] == 0.5


def test_none_fidelity_excluded_not_zero(monkeypatch):
    monkeypatch.setattr("eval.grpo_reward.score_batch",
                        _spy([{"behavioral_fidelity": None, "collapsed": False}] * 2))
    out = shaped_rewards([list(range(64)), list(range(64))], _SPEC)
    assert out[0][0] is None and out[1][0] is None                   # excluded, NOT scored 0.0


def test_none_sample_does_not_shift_group_stats():
    """A None-fidelity sample must not change the advantages of the measurable samples."""
    with_none = group_advantages([0.75, 0.25, None])
    without = group_advantages([0.75, 0.25])
    assert with_none[0] == pytest.approx(without[0])
    assert with_none[1] == pytest.approx(without[1])
    assert with_none[2] is None


# --- collapse penalty (Doc 01 §5 worked example) --------------------------------------------------
def test_collapse_penalty_suppresses_dominant_code(monkeypatch):
    # 6 samples: #4 (index 3) refuses (None codes, handled before decode). The 5 valid samples map to
    # these records IN ORDER; sample #5 (dominant code, index 4) has the SAME raw fidelity 0.50 as the
    # healthy sample #2 (index 1) but is `collapsed`.
    valid = list(range(64))
    codes = [valid, valid, valid, None, valid, valid]
    recs = [
        {"behavioral_fidelity": 0.75, "collapsed": False},   # #1 healthy
        {"behavioral_fidelity": 0.50, "collapsed": False},   # #2 healthy   (raw 0.50)
        {"behavioral_fidelity": 0.25, "collapsed": False},   # #3 healthy
        {"behavioral_fidelity": 0.50, "collapsed": True},    # #5 dominant  (raw 0.50, collapsed)
        {"behavioral_fidelity": 0.00, "collapsed": True},    # #6 neutral collapse
    ]
    monkeypatch.setattr("eval.grpo_reward.score_batch", _spy(recs))

    out = shaped_rewards(codes, _SPEC, collapse_penalty=0.25)
    rewards = [r for r, _ in out]
    # shaped rewards: refusal 0.0; #5 0.50-0.25=0.25; #6 max(0,0-0.25)=0.0
    assert rewards == [pytest.approx(x) for x in (0.75, 0.50, 0.25, 0.0, 0.25, 0.0)]

    adv = group_advantages(rewards, eps=1e-4)
    healthy_050 = adv[1]      # #2, raw fidelity 0.50, healthy
    dominant_050 = adv[4]     # #5, raw fidelity 0.50, collapsed
    assert dominant_050 < 0.0, "the dominant-code collapse must get a negative advantage"
    assert dominant_050 < healthy_050, "penalty must push the collapse below the equal-raw healthy sample"
    assert adv[0] > adv[1] > adv[2]      # ordering among the healthy samples preserved


def test_collapse_penalty_zero_recovers_pure_fidelity(monkeypatch):
    monkeypatch.setattr("eval.grpo_reward.score_batch",
                        _spy([{"behavioral_fidelity": 0.5, "collapsed": True}]))
    out = shaped_rewards([list(range(64))], _SPEC, collapse_penalty=0.0)
    assert out[0][0] == pytest.approx(0.5)      # penalty 0 -> collapsed flag doesn't dock the reward


# --- group_advantages math ------------------------------------------------------------------------
def test_group_advantages_standardize():
    adv = group_advantages([0.0, 1.0], eps=0.0)
    # mean 0.5, population std 0.5 -> +/-1
    assert adv[0] == pytest.approx(-1.0) and adv[1] == pytest.approx(1.0)


def test_group_advantages_zero_std_gives_zero():
    adv = group_advantages([0.3, 0.3, 0.3], eps=1e-4)
    assert all(a == pytest.approx(0.0) for a in adv)     # no variation -> no learning signal


def test_group_advantages_all_none():
    assert group_advantages([None, None]) == [None, None]


def test_group_advantages_all_refused_zero():
    # all refused -> all reward 0 -> std 0 -> advantage 0 (prompt wasted, math safe)
    assert all(a == pytest.approx(0.0) for a in group_advantages([0.0, 0.0, 0.0]))


def test_reward_determinism(monkeypatch):
    monkeypatch.setattr("eval.grpo_reward.score_batch",
                        _spy([{"behavioral_fidelity": 0.42, "collapsed": False}]))
    a = shaped_rewards([list(range(64))], _SPEC)
    monkeypatch.setattr("eval.grpo_reward.score_batch",
                        _spy([{"behavioral_fidelity": 0.42, "collapsed": False}]))
    b = shaped_rewards([list(range(64))], _SPEC)
    assert a[0][0] == b[0][0]
