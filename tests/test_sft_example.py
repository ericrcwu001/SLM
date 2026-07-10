"""Unit tests for :mod:`sft.example` — one of the two previously-untested SFT choke points
(ADR 0024). GPU-free: exercises the pure row helpers, the **unit-aware** holdout partitioning,
and the exact-64 survival counter (AUDIT F8) with a fake tokenizer + numpy tensors — no torch,
transformers, or `sft` extra required.
"""

from __future__ import annotations

import numpy as np
import pytest

from eval.vocab import code_token
from sft import example as ex
from sft.holdout import holdout_key, is_holdout_row


class _Cfg:
    def __init__(self, dt):
        self.bnb_4bit_compute_dtype = dt


def test_resolve_compute_dtype_float16_config():
    import torch
    assert ex.resolve_compute_dtype(_Cfg("float16")) is torch.float16


def test_resolve_compute_dtype_bf16_falls_back_on_t4(monkeypatch):
    # Simulate a T4: CUDA present but no hardware bf16 -> must fall back to float16.
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert ex.resolve_compute_dtype(_Cfg("bfloat16")) is torch.float16


def test_resolve_compute_dtype_bf16_kept_on_ampere(monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert ex.resolve_compute_dtype(_Cfg("bfloat16")) is torch.bfloat16


def _supported_row(rid: str, unit: str, *, family: str = "ppr10k_derived") -> dict:
    return {
        "id": rid,
        "split_unit_id": unit,
        "source_family": family,
        "is_supported": True,
        "target_tokens": list(range(64)),
        "image_path": "images/x.png",
        "instruction": "warmer +2",
        "assistant_target": "<lut_bos>" + "".join(code_token(i % 256) for i in range(64)) + "<lut_eos>",
    }


# --- is_supported_materialized -------------------------------------------------

def test_is_supported_materialized_requires_all_fields():
    row = _supported_row("a", "u1")
    assert ex.is_supported_materialized(row)
    for missing in ("target_tokens", "image_path", "instruction", "assistant_target"):
        bad = dict(row)
        bad.pop(missing)
        assert not ex.is_supported_materialized(bad), missing
    short = dict(row)
    short["target_tokens"] = list(range(63))       # not 64 -> rejected
    assert not ex.is_supported_materialized(short)
    unsup = dict(row)
    unsup["is_supported"] = False
    assert not ex.is_supported_materialized(unsup)


# --- unit-aware holdout partitioning (the F4 fix) ------------------------------

def test_holdout_is_unit_aware_not_row_aware():
    # Two distinct rows in the SAME split unit must land on the SAME side of the holdout boundary.
    a = _supported_row("row-a", "unit_shared")
    b = _supported_row("row-b", "unit_shared")
    assert holdout_key(a) == holdout_key(b) == "unit_shared"
    assert is_holdout_row(a) == is_holdout_row(b)


def test_supported_rows_split_has_zero_unit_crossing():
    # Build many rows across many units (2 rows per unit) and confirm the train/holdout carve never
    # splits a unit — the core leakage invariant of ADR 0024.
    rows = []
    for u in range(400):
        unit = f"unit_{u:04d}"
        rows.append(_supported_row(f"r{u}a", unit))
        rows.append(_supported_row(f"r{u}b", unit))
    train = ex.supported_rows(rows, holdout=False)
    held = ex.supported_rows(rows, holdout=True)
    assert train and held and len(train) + len(held) == len(rows)
    train_units = {r["split_unit_id"] for r in train}
    held_units = {r["split_unit_id"] for r in held}
    assert train_units.isdisjoint(held_units), "a split unit straddled the holdout boundary (leak)"
    # both rows of any given unit are on the same side
    for u in range(400):
        unit = f"unit_{u:04d}"
        sides = {is_holdout_row(r) for r in rows if r["split_unit_id"] == unit}
        assert len(sides) == 1


def test_supported_rows_none_returns_all_materialized():
    rows = [_supported_row("a", "u1"), _supported_row("b", "u2")]
    rows.append({"id": "c", "is_supported": False, "instruction": "x", "image_path": "y"})
    assert len(ex.supported_rows(rows, holdout=None)) == 2


# --- exact-64 surviving-code counter (AUDIT F8) --------------------------------

class _FakeTokenizer:
    """Maps ``<lut_NNN>`` -> 1000+N and everything else to a fixed non-code id."""

    def convert_tokens_to_ids(self, token: str) -> int:
        from eval.vocab import code_index

        idx = code_index(token)
        return 1000 + idx if idx is not None else 5


def test_surviving_code_positions_counts_only_code_tokens():
    tok = _FakeTokenizer()
    # prompt = 3 non-code tokens (id 5), assistant span = 64 code tokens (1000..1063) + eos (5)
    prompt = [5, 5, 5]
    assistant = [1000 + i for i in range(64)] + [5]
    input_ids = np.array([prompt + assistant])
    n_prompt = len(prompt)
    assert ex.surviving_code_positions(tok, input_ids, n_prompt) == 64


def test_surviving_code_positions_detects_partial_truncation():
    tok = _FakeTokenizer()
    prompt = [5, 5, 5]
    assistant = [1000 + i for i in range(60)]     # only 60 of 64 codes survived
    input_ids = np.array([prompt + assistant])
    assert ex.surviving_code_positions(tok, input_ids, len(prompt)) == 60


def test_code_token_id_cache_is_stable():
    tok = _FakeTokenizer()
    a = ex._code_token_ids(tok)
    b = ex._code_token_ids(tok)
    assert a is b and len(a) == 256          # cached + exactly the 256 code ids
