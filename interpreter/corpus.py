"""Interpreter corpus loader + the leakage-safe holdout key.

Reuses the generator's deterministic unit-aware bucketing (:func:`sft.holdout.is_holdout`) but with a
**strict** key: an interpreter row MUST carry ``split_unit_id`` (stamped by
``scripts.build_interpreter_corpus``). Unlike :func:`sft.holdout.holdout_key`, this does NOT fall
back to the row id — a fallback here would give each of a LUT's captions an independent holdout
coin-flip and re-open the 48.5% row-id-carve leak (ADR 0024). Missing unit -> hard error.
"""

from __future__ import annotations

import json
from pathlib import Path

from sft.holdout import DEFAULT_HOLDOUT_FRAC, is_holdout


def load_interpreter_rows(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def interpreter_holdout_key(row: dict) -> str:
    """The leakage-safe holdout key: the row's ``split_unit_id`` (NO id fallback).

    Raises if absent — the corpus builder guarantees every row has one; a missing unit means the
    corpus was built wrong and scoring would leak, so we fail loud instead of silently degrading.
    """
    unit = row.get("split_unit_id")
    if not unit:
        raise ValueError(
            f"interpreter row {row.get('id')!r} has no split_unit_id; holdout would fall back to a "
            f"per-row id and leak same-LUT captions across the boundary. Rebuild the corpus "
            f"(scripts.build_interpreter_corpus).")
    return unit


def is_holdout_row(row: dict, frac: float = DEFAULT_HOLDOUT_FRAC) -> bool:
    return is_holdout(interpreter_holdout_key(row), frac)


def split_train_holdout(rows: list[dict],
                        frac: float = DEFAULT_HOLDOUT_FRAC) -> tuple[list[dict], list[dict]]:
    """Partition rows into (train, holdout) by the leakage-safe unit key."""
    train, holdout = [], []
    for r in rows:
        (holdout if is_holdout_row(r, frac) else train).append(r)
    return train, holdout
