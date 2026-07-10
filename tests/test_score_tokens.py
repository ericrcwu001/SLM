"""Unit tests for :mod:`sft.score_tokens` — the second previously-untested SFT choke point
(ADR 0024). GPU-free: exercises the pure scoring-summary aggregation, the unit-clustered
group-bootstrap CI, and the exact-64 constant. The heavy :func:`sft.score_tokens.score` (torch)
is not exercised here — its record-building loop is thin glue over these tested pure functions.
"""

from __future__ import annotations

from sft import score_tokens as st


def _rec(unit, family, correct, total=64):
    return {"unit": unit, "family": family, "correct": correct, "total": total,
            "exact": correct == total}


# --- summarize_scores ----------------------------------------------------------

def test_summary_overall_micro_and_macro():
    # ppr: 2 rows, 124/128 correct. fivek: 2 rows, 96/128. micro = 220/256; macro = mean(.96875,.75)
    recs = [_rec("u1", "ppr10k_derived", 60), _rec("u2", "ppr10k_derived", 64),
            _rec("u3", "fivek_derived", 32), _rec("u4", "fivek_derived", 64)]
    rep = st.summarize_scores(recs, B=300, seed=0)
    assert abs(rep["metric"] - 220 / 256) < 1e-9
    assert rep["token_accuracy"] == rep["metric"]
    assert abs(rep["per_family"]["ppr10k_derived"]["accuracy"] - 124 / 128) < 1e-9
    assert abs(rep["per_family"]["fivek_derived"]["accuracy"] - 96 / 128) < 1e-9
    macro = (124 / 128 + 96 / 128) / 2
    assert abs(rep["macro_family_accuracy"] - macro) < 1e-9
    assert rep["exact_match_rate"] == 0.5          # 2 of 4 rows exact
    assert rep["scored_rows"] == 4 and rep["scored_units"] == 4
    assert rep["code_positions"] == 256 and rep["correct"] == 220


def test_summary_is_deterministic():
    recs = [_rec(f"u{i}", "ppr10k_derived", 50 + i % 10) for i in range(20)]
    assert st.summarize_scores(recs, B=500, seed=0) == st.summarize_scores(recs, B=500, seed=0)


def test_summary_empty_does_not_crash():
    rep = st.summarize_scores([], B=100, seed=0)
    assert rep["metric"] == 0.0 and rep["scored_rows"] == 0 and rep["per_family"] == {}


# --- group-bootstrap CI (clustered on split_unit_id) ---------------------------

def test_group_bootstrap_point_and_ci_bracket():
    units = [f"u{i}" for i in range(40)]
    corr = [50] * 40
    tot = [64] * 40
    point, lo, hi = st._group_bootstrap_ratio(units, corr, tot, B=1000, seed=0)
    assert abs(point - 50 / 64) < 1e-9
    assert lo is not None and hi is not None
    assert lo <= point <= hi


def test_group_bootstrap_ci_none_with_single_unit():
    # One unit -> CI not estimable (cluster bootstrap needs >=2 clusters); point still returned.
    point, lo, hi = st._group_bootstrap_ratio(["u1", "u1"], [60, 64], [64, 64], B=500, seed=0)
    assert point is not None and lo is None and hi is None


def test_group_bootstrap_clusters_by_unit_not_row():
    # 100 rows but only 2 units -> CI reflects 2 clusters (wide), not 100 independent rows.
    units = ["u1"] * 50 + ["u2"] * 50
    corr = [64] * 50 + [0] * 50
    tot = [64] * 100
    point, lo, hi = st._group_bootstrap_ratio(units, corr, tot, B=2000, seed=0)
    assert abs(point - 0.5) < 1e-9
    # With only 2 clusters (one all-right, one all-wrong) the bootstrap spans ~[0,1].
    assert lo is not None and hi - lo > 0.5


def test_per_family_ci_none_when_family_has_one_unit():
    recs = [_rec("u1", "solo_family", 60), _rec("u2", "multi", 60), _rec("u3", "multi", 64)]
    rep = st.summarize_scores(recs, B=300, seed=0)
    assert rep["per_family"]["solo_family"]["ci_low"] is None
    assert rep["per_family"]["multi"]["ci_low"] is not None


# --- exact-64 invariant --------------------------------------------------------

def test_exact_64_constant():
    assert st._CODES_PER_ROW == 64


# --- config loader -------------------------------------------------------------

def test_load_config_defaults_ok():
    cfg = st._load_config(None)
    assert cfg.num_new_tokens == 259          # locked knob survives the round-trip
