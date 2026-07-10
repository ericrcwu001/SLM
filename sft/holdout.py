"""Deterministic, **unit-aware** held-out split for SFT token-accuracy scoring.

The active SFT corpus ships every row with ``split='train'`` (no validation slice), so we carve a
stable holdout ourselves. Per ADR 0024 (eval-honesty contract) the holdout keys on the pipeline's
leakage-safe ``split_unit_id`` — **not** the row id — so near-duplicate LUTs (rows unioned into one
split unit by shared base identity or leakage near-neighbors, see :mod:`data_pipeline.splits`) can
never straddle the train/holdout boundary. The prior row-id carve leaked: 82/169 held-out rows
(48.5%) shared a split unit with training (``AUDIT_claude_codex_prompt_to_lut.md`` F4/finding B).

The SAME predicate is used by :mod:`sft.train` (to EXCLUDE holdout units from training) and
:mod:`sft.score_tokens` (to score ONLY holdout units), guaranteeing the metric measures
generalization, never memorization.

Pure + stdlib-only (no torch/transformers) so it is import- and unit-test-safe without the ``sft``
extra. Determinism is by SHA-1 of the split-unit key, so the split is identical across runs and
machines and independent of file order or corpus size.
"""

from __future__ import annotations

import hashlib

DEFAULT_HOLDOUT_FRAC = 0.06
_BUCKETS = 10_000


def _bucket(key: str) -> int:
    """Stable bucket in [0, _BUCKETS) from a key (order- and machine-independent)."""
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % _BUCKETS


def is_holdout(key: str, frac: float = DEFAULT_HOLDOUT_FRAC) -> bool:
    """True for ~``frac`` of keys (the held-out scoring slice); deterministic and stable.

    ``key`` is the leakage-safe ``split_unit_id`` in production (see :func:`holdout_key`); the pure
    hash predicate is key-agnostic so the low-level bucketing stays unit-testable with plain strings.

    ``frac<=0`` → nothing is holdout (all units train); ``frac>=1`` → everything is holdout.
    An empty/missing key is never holdout (it would collide across rows).
    """
    if not key:
        return False
    if frac <= 0.0:
        return False
    if frac >= 1.0:
        return True
    return _bucket(key) < int(round(frac * _BUCKETS))


def holdout_key(row: dict) -> str:
    """The unit-aware holdout key for a row: its ``split_unit_id`` (leakage-safe).

    Falls back to the row id only when a row carries no split unit (legacy/fixture rows); every
    supported+materialized corpus row has a ``split_unit_id``, so in production this is always the
    unit key and near-duplicates share a holdout decision.
    """
    return row.get("split_unit_id") or row.get("id", "")


def is_holdout_row(row: dict, frac: float = DEFAULT_HOLDOUT_FRAC) -> bool:
    """True iff ``row``'s split unit is in the held-out slice (unit-aware; see :func:`holdout_key`)."""
    return is_holdout(holdout_key(row), frac)
