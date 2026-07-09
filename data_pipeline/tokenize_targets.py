"""Token materialization interface (GATED — needs a frozen VQ tokenizer).

The 64 ``target_tokens`` require encoding the canonical residual through the frozen VQ
codebook (master-plan Stage 8). No tokenizer exists yet, so :func:`encode_residual_to_codes`
raises :class:`RequiresTokenizer`; downstream rows carry ``target_tokens=None`` and
``token_status = pending_tokenizer``. Signature matches the eventual real encoder.
"""

from __future__ import annotations

import numpy as np

from .constants import TOKEN_STATUS_PENDING
from .errors import RequiresTokenizer

ENABLED = False


def is_available() -> bool:
    return ENABLED


def encode_residual_to_codes(residual: np.ndarray, manifest: dict | None = None) -> list[int]:
    """Encode a canonical 17^3 residual -> 64 codebook ids (0..255). Gated until freeze."""
    raise RequiresTokenizer(
        "VQ tokenizer not frozen: cannot materialize target_tokens "
        f"(rows carry token_status={TOKEN_STATUS_PENDING!r}). Train+freeze the tokenizer "
        "(Stages 7-8), then wire the encoder here."
    )


def token_status() -> str:
    return TOKEN_STATUS_PENDING
