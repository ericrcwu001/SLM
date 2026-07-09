"""MLX -> torch weight conversion (v2) — turns an MLX-trained tokenizer into the torch VQVAE.

The bridge that keeps everything downstream (freeze, eval decoder, CLI, Colab) on the torch
implementation. Correctness is guaranteed by the round-trip parity test.

v2 has NO transposed convs — the decoder is trilinear-resize + Conv3d — so BOTH encoder and
decoder convs are regular Conv3d and use the same weight permutation. The trilinear resize is
parameter-free. Convs use replicate padding, which carries no weights either.

Weight-layout facts (verified on mlx 0.32.0):
  * MLX Conv3d weight (out, kD, kH, kW, in) -> torch (out, in, kD, kH, kW) = transpose(0,4,1,2,3)
  * GroupNorm weight/bias and conv bias are 1-D, copied as-is.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from ..config import TokenizerConfig
from ..manifest import hash_state_dict
from ..model import VQVAE
from ..train import lut_corpus_hash


def _np(a) -> np.ndarray:
    return np.array(a)


def _conv_w(mlx_w) -> torch.Tensor:
    # MLX (out,kD,kH,kW,in) -> torch Conv3d (out,in,kD,kH,kW)
    return torch.from_numpy(np.ascontiguousarray(_np(mlx_w).transpose(0, 4, 1, 2, 3))).float()


def _vec(mlx_w) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(_np(mlx_w))).float()


def mlx_to_torch_vqvae(mlx_model, cfg: TokenizerConfig) -> VQVAE:
    """Build a torch VQVAE(cfg) and load the MLX v2 model's weights into it (strict)."""
    tm = VQVAE(cfg)
    sd = tm.state_dict()

    def setw(key: str, tensor: torch.Tensor) -> None:
        want = tuple(sd[key].shape)
        got = tuple(tensor.shape)
        if want != got:
            raise ValueError(f"shape mismatch at {key}: torch expects {want}, converted {got}")
        sd[key] = tensor

    # encoder: Conv3d blocks (b3 has no norm)
    for b in ("b1", "b2", "b3"):
        blk = getattr(mlx_model.encoder, b)
        setw(f"encoder.{b}.conv.weight", _conv_w(blk.conv.weight))
        setw(f"encoder.{b}.conv.bias", _vec(blk.conv.bias))
        if b != "b3":
            setw(f"encoder.{b}.norm.weight", _vec(blk.norm.weight))
            setw(f"encoder.{b}.norm.bias", _vec(blk.norm.bias))

    # decoder: v2 resize+Conv3d blocks -> torch `.conv` (regular Conv3d), b3 no norm.
    for b in ("b1", "b2", "b3"):
        blk = getattr(mlx_model.decoder, b)
        setw(f"decoder.{b}.conv.weight", _conv_w(blk.conv.weight))
        setw(f"decoder.{b}.conv.bias", _vec(blk.conv.bias))
        if b != "b3":
            setw(f"decoder.{b}.norm.weight", _vec(blk.norm.weight))
            setw(f"decoder.{b}.norm.bias", _vec(blk.norm.bias))

    setw("vq.codebook", _vec(mlx_model.vq._codebook))
    setw("vq.cluster_size", _vec(mlx_model.vq._cluster_size))
    setw("vq.embed_avg", _vec(mlx_model.vq._embed_avg))

    tm.load_state_dict(sd)  # strict=True
    tm.eval()
    return tm


def save_torch_checkpoint(mlx_model, cfg: TokenizerConfig, records, path: str) -> VQVAE:
    """Convert + write a freeze-compatible torch checkpoint (consumed by tokenizer.freeze)."""
    tm = mlx_to_torch_vqvae(mlx_model, cfg)
    sd = tm.state_dict()
    ck = {
        "config": cfg.to_dict(),
        "model_state": sd,
        "lut_corpus_hash": lut_corpus_hash(records),
        "tokenizer_weights_hash": hash_state_dict(sd),
        "trained_with": "mlx",
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(ck, path)
    return tm
