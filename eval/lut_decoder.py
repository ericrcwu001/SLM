"""LUT decoder interface — DISABLED in this build.

Real responsibility (model_architecture.md "LUT Tokenizer", eval L2): map 64 codebook
ids through the frozen VQ decoder to a canonical 17x17x17 residual LUT, then add the
identity grid to obtain the absolute LUT.

There is no trained/frozen VQ tokenizer yet (training Stages 7-8), so this module is
disabled. It exposes the eventual signature so that L2-L7 and the CLI can call it and
degrade cleanly; enabling is a drop-in once the tokenizer manifest is frozen (flip
``ENABLED`` and implement ``decode_tokens_to_residual`` against the frozen decoder).
"""

from __future__ import annotations

import numpy as np

ENABLED = False


class DecoderDisabled(RuntimeError):
    """Raised when a decode is attempted while the VQ decoder is disabled."""


def is_enabled() -> bool:
    return ENABLED


def decode_tokens_to_residual(token_ids: list[int], manifest: dict | None = None) -> "np.ndarray":
    """Decode 64 codebook ids -> canonical 17x17x17x3 residual LUT.

    DISABLED: raises :class:`DecoderDisabled`. When enabled it must (a) assert exactly
    64 ids in 0..255, (b) map token suffix -> codebook index per the frozen manifest,
    (c) run the frozen VQ decoder, (d) return a finite residual tensor.
    """
    raise DecoderDisabled(
        "lut_decoder is disabled: no frozen VQ tokenizer/decoder exists yet "
        "(training Stages 7-8). L2-L7 report not_evaluated:decoder_disabled."
    )
