"""Interpreter holdout key: leakage-safe (split_unit_id, NO id fallback)."""

from __future__ import annotations

import pytest

from interpreter.corpus import (
    interpreter_holdout_key,
    is_holdout_row,
    split_train_holdout,
)


def test_key_is_split_unit_id():
    assert interpreter_holdout_key({"id": "cap_x_literal", "split_unit_id": "unit_9"}) == "unit_9"


def test_missing_unit_raises_no_id_fallback():
    # The crux: unlike sft.holdout.holdout_key, there is NO fallback to the row id.
    with pytest.raises(ValueError):
        interpreter_holdout_key({"id": "cap_x_literal"})


def test_all_captions_of_a_lut_share_one_holdout_decision():
    styles = ["literal", "metaphor", "mood", "concept", "slang"]
    rows = [{"id": f"cap_lutA_{s}", "split_unit_id": "unitA"} for s in styles]
    decisions = {is_holdout_row(r) for r in rows}
    assert len(decisions) == 1  # one unit -> one decision for all 5 styles


def test_split_is_deterministic_and_carves_a_minority():
    rows = [{"id": f"r{i}", "split_unit_id": f"unit_{i}"} for i in range(2000)]
    train, holdout = split_train_holdout(rows, frac=0.06)
    assert len(train) + len(holdout) == 2000
    assert 0 < len(holdout) < len(train)               # a real minority carve
    # deterministic across calls
    assert split_train_holdout(rows, frac=0.06)[1] == holdout
