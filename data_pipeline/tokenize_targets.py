"""Token materialization interface — WIRED to the frozen VQ tokenizer (master-plan Stage 8).

The 64 ``target_tokens`` for a supported row are its canonical 17^3 residual LUT encoded
through the frozen VQ codebook. :func:`encode_residual_to_codes` delegates to
:meth:`tokenizer.model.VQVAE.encode` on the frozen weights loaded and integrity-checked
against the manifest by :func:`tokenizer.frozen.load_frozen_vqvae`.

Torch/tokenizer are imported lazily inside the function so importing this module stays
dependency-light (and honest when the frozen weights are not staged: a missing tokenizer
still surfaces as :class:`RequiresTokenizer`, so callers record ``pending_tokenizer`` rather
than fabricating targets).
"""

from __future__ import annotations

import numpy as np

from .constants import TOKEN_STATUS_MATERIALIZED, TOKEN_STATUS_PENDING
from .errors import RequiresTokenizer

ENABLED = True


def is_available() -> bool:
    return ENABLED


def encode_residual_to_codes(residual: np.ndarray, manifest: dict | None = None) -> list[int]:
    """Encode a canonical 17^3 residual ``[r,g,b,3]`` -> 64 codebook ids (0..255).

    Raises :class:`RequiresTokenizer` if the tokenizer runtime is unavailable (torch missing
    or the frozen weights not staged) so downstream code degrades to ``pending_tokenizer``.
    """
    if not ENABLED:
        raise RequiresTokenizer("tokenize_targets is disabled (ENABLED=False).")
    try:
        from tokenizer.frozen import FrozenTokenizerError, load_frozen_vqvae
    except Exception as exc:  # torch / tokenizer package unavailable
        raise RequiresTokenizer(f"tokenizer runtime unavailable: {exc}") from exc
    try:
        model, _manifest = load_frozen_vqvae()
    except FrozenTokenizerError as exc:
        raise RequiresTokenizer(str(exc)) from exc
    return model.encode(residual)


def token_status(materialized: bool = False) -> str:
    """Row token-status marker: pending until the encoder has actually run over the row."""
    return TOKEN_STATUS_MATERIALIZED if materialized else TOKEN_STATUS_PENDING
