"""Frozen tokenizer manifest (model_architecture.md "LUT Tokenizer" fields) + hashing.

The manifest binds the tokenizer identity that the whole system commits to at Stage 8:
geometry, orders, codebook/decoder hashes, and the corpus/weights hashes. Changing any
of these requires a new manifest and regenerated targets (Canonical LUT Contract).
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

from .config import TokenizerConfig


def hash_tensor(t: torch.Tensor) -> str:
    """Deterministic SHA-256 of a tensor's float64 bytes (device/dtype-independent)."""
    arr = np.ascontiguousarray(t.detach().cpu().to(torch.float64).numpy())
    return hashlib.sha256(arr.tobytes()).hexdigest()


def hash_state_dict(sd: dict) -> str:
    """SHA-256 over a state_dict, key-sorted so it is order-independent."""
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        h.update(k.encode("utf-8"))
        h.update(np.ascontiguousarray(sd[k].detach().cpu().to(torch.float64).numpy()).tobytes())
    return h.hexdigest()


def encoder_decoder_layer_table(cfg: TokenizerConfig) -> dict:
    """Static layer table (geometry is pinned in config)."""
    c1, c2 = cfg.enc_channels
    d1, d2 = cfg.dec_channels
    return {
        "encoder": [
            {"op": "Conv3d", "in": 3, "out": c1, "k": 3, "s": 2, "p": 1, "size": "17->9"},
            {"op": "Conv3d", "in": c1, "out": c2, "k": 3, "s": 2, "p": 1, "size": "9->5"},
            {"op": "Conv3d", "in": c2, "out": cfg.code_dim, "k": 2, "s": 1, "p": 0, "size": "5->4"},
        ],
        "decoder": [
            {"op": "ConvTranspose3d", "in": cfg.code_dim, "out": d1, "k": 2, "s": 1, "p": 0, "op_pad": 0, "size": "4->5"},
            {"op": "ConvTranspose3d", "in": d1, "out": d2, "k": 3, "s": 2, "p": 1, "op_pad": 0, "size": "5->9"},
            {"op": "ConvTranspose3d", "in": d2, "out": 3, "k": 3, "s": 2, "p": 1, "op_pad": 0, "size": "9->17"},
        ],
    }


def build_frozen_manifest(
    model,                       # noqa: ANN001 (VQVAE)
    cfg: TokenizerConfig,
    *,
    lut_corpus_hash: str,
    tokenizer_weights_hash: str,
    gate_report: dict | None = None,
) -> dict:
    """Assemble the full frozen manifest dict (model_architecture.md field list)."""
    vq_codebook_sha256 = hash_tensor(model.vq.codebook)
    vq_decoder_sha256 = hash_state_dict(model.decoder.state_dict())
    tokenizer_version = f"{cfg.arch_version}__w{tokenizer_weights_hash[:12]}"

    return {
        "tokenizer_version": tokenizer_version,
        "arch_version": cfg.arch_version,
        # geometry / representation
        "lut_grid_size": f"{cfg.grid}x{cfg.grid}x{cfg.grid}",
        "representation": "residual_after_identity",
        "canonical_domain_id": cfg.canonical_domain_id,
        "interpolation": "trilinear",
        "latent_grid": f"{cfg.latent_grid}x{cfg.latent_grid}x{cfg.latent_grid}",
        "token_count": cfg.token_count,
        "codebook_size": cfg.codebook_size,
        # orders (verbatim pinned strings)
        "tensor_axis_order": cfg.tensor_axis_order,
        "cube_table_order": "rgb_r_fastest__b_outer_g_mid_r_inner",
        "latent_flatten_order": cfg.latent_flatten_order,
        "token_suffix_to_codebook_index": cfg.token_suffix_to_codebook_index,  # "identity"
        "code_id_to_codebook_row": "identity",  # code id k -> codebook row k
        # weights / hashes
        "vq_codebook_sha256": vq_codebook_sha256,
        "vq_decoder_sha256": vq_decoder_sha256,
        "tokenizer_weights_hash": tokenizer_weights_hash,
        "lut_corpus_hash": lut_corpus_hash,
        # layer table + pipeline versions
        "encoder_decoder_layer_table": encoder_decoder_layer_table(cfg),
        "color_pipeline_version": cfg.color_pipeline_version,
        "cube_serialization_version": cfg.cube_serialization_version,
        # gate provenance (informational; the pass decision lives in the freeze step)
        "gate_report": gate_report or {},
    }
