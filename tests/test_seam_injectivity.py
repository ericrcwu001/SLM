"""Decoder-free seam-injectivity metric math (collision rate + token-accuracy upper bound)."""

from __future__ import annotations

import importlib

sa = importlib.import_module("scripts.analyze_seam_injectivity")


def _row(spec: str, codes: list[int]) -> dict:
    return {"measured_behavior": {"spec": spec}, "target_tokens": codes}


def _by_spec(mb: dict) -> str:
    return mb["spec"]


def test_unique_specs_upper_bound_is_one():
    rows = [_row("A", [1] * 64), _row("B", [2] * 64), _row("C", [3] * 64)]
    d = sa.analyze(rows, _by_spec)
    assert d["spec_uniqueness"] == 1.0
    assert d["lossy_collision_rate"] == 0.0
    assert d["token_accuracy_upper_bound"] == 1.0


def test_same_spec_same_codes_is_not_a_lossy_collision():
    rows = [_row("A", [5] * 64), _row("A", [5] * 64)]   # identical codes -> spec is fine
    d = sa.analyze(rows, _by_spec)
    assert d["unique_specs"] == 1
    assert d["lossy_collision_rate"] == 0.0
    assert d["token_accuracy_upper_bound"] == 1.0


def test_same_spec_divergent_codes_is_lossy():
    a = [1] * 64
    b = [1] * 64
    b[0] = 9                                            # differ at exactly one of 64 positions
    d = sa.analyze([_row("A", a), _row("A", b)], _by_spec)
    assert d["lossy_collision_rate"] == 1.0             # both rows share a spec with divergent codes
    # per-position majority: 63 positions agree (2/2), 1 position splits (best 1/2)
    assert abs(d["token_accuracy_upper_bound"] - (63 * 2 + 1) / (64 * 2)) < 1e-9
