"""Deterministic held-out split for SFT token-accuracy scoring.

The active SFT corpus ships every row with ``split='train'`` (no validation slice), so we carve a
stable holdout by hashing the row id. The SAME predicate is used by :mod:`sft.train` (to EXCLUDE
holdout rows from training) and :mod:`sft.score_tokens` (to score ONLY holdout rows), guaranteeing
the metric measures generalization, never memorization.

Pure + stdlib-only (no torch/transformers) so it is import- and unit-test-safe without the ``sft``
extra. Determinism is by SHA-1 of the row id, so the split is identical across runs and machines and
independent of file order or corpus size.
"""

from __future__ import annotations

import hashlib

DEFAULT_HOLDOUT_FRAC = 0.06
_BUCKETS = 10_000


def _bucket(row_id: str) -> int:
    """Stable bucket in [0, _BUCKETS) from the row id (order- and machine-independent)."""
    digest = hashlib.sha1(row_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % _BUCKETS


def is_holdout(row_id: str, frac: float = DEFAULT_HOLDOUT_FRAC) -> bool:
    """True for ~``frac`` of ids (the held-out scoring slice); deterministic and stable.

    ``frac<=0`` → nothing is holdout (all rows train); ``frac>=1`` → everything is holdout.
    An empty/missing id is never holdout (it would collide across rows).
    """
    if not row_id:
        return False
    if frac <= 0.0:
        return False
    if frac >= 1.0:
        return True
    return _bucket(row_id) < int(round(frac * _BUCKETS))
