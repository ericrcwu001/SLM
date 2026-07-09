"""LUT VQ-tokenizer package (Stages 7-8).

Import-safe: ``import tokenizer`` pulls in only the torch-free config. The torch
models/losses are imported lazily (or directly from their submodules,
``from tokenizer.model import VQVAE``) so that merely importing the package never
loads torch, reads a file, or runs compute. Nothing here starts training.
"""

from __future__ import annotations

from .config import (  # torch-free
    CODEBOOK_SIZE,
    DEFAULT_CONFIG,
    GRID,
    LATENT_FLATTEN_ORDER,
    LATENT_GRID,
    TENSOR_AXIS_ORDER,
    TOKEN_COUNT,
    TOKEN_SUFFIX_TO_CODEBOOK_INDEX,
    TOKENIZER_ARCH_VERSION,
    TokenizerConfig,
)

__all__ = [
    "TokenizerConfig",
    "DEFAULT_CONFIG",
    "GRID",
    "LATENT_GRID",
    "TOKEN_COUNT",
    "CODEBOOK_SIZE",
    "TENSOR_AXIS_ORDER",
    "LATENT_FLATTEN_ORDER",
    "TOKEN_SUFFIX_TO_CODEBOOK_INDEX",
    "TOKENIZER_ARCH_VERSION",
    "VQVAE",  # lazy (see __getattr__)
]


def __getattr__(name: str):
    # Lazy torch-dependent exports so package import stays torch-free.
    if name == "VQVAE":
        from .model import VQVAE

        return VQVAE
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
