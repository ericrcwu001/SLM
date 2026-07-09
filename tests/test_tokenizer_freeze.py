"""Checkpoint roundtrip, frozen-manifest fields, and fail-closed freeze gate.

Synthetic + CPU. An UNtrained model must fail the reconstruction gate, so freeze()
must abort and write nothing — verifying the gate actually guards the freeze.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tokenizer import data as D
from tokenizer import freeze as F
from tokenizer.config import TokenizerConfig
from tokenizer.manifest import build_frozen_manifest
from tokenizer.model import VQVAE
from tokenizer.train import save_checkpoint


REQUIRED_MANIFEST_FIELDS = {
    "tokenizer_version", "lut_grid_size", "representation", "canonical_domain_id",
    "interpolation", "latent_grid", "token_count", "codebook_size", "tensor_axis_order",
    "cube_table_order", "latent_flatten_order", "token_suffix_to_codebook_index",
    "code_id_to_codebook_row", "vq_codebook_sha256", "vq_decoder_sha256",
    "encoder_decoder_layer_table", "lut_corpus_hash", "tokenizer_weights_hash",
    "color_pipeline_version", "cube_serialization_version",
}


def _records(tmp_path, n=12):
    recs = []
    for i in range(n):
        arr = np.random.default_rng(i).standard_normal((17, 17, 17, 3)) * 0.05
        p = tmp_path / f"k{i}.npy"
        np.save(p, arr)
        recs.append(D.LutRecord(f"k{i}", str(p), "gmic_rawtherapee", "gold"))
    return recs


def test_manifest_has_all_required_fields():
    torch.manual_seed(0)
    model = VQVAE(TokenizerConfig())
    m = build_frozen_manifest(model, model.cfg, lut_corpus_hash="abc", tokenizer_weights_hash="def0123456789")
    assert REQUIRED_MANIFEST_FIELDS <= set(m), REQUIRED_MANIFEST_FIELDS - set(m)
    assert len(m["vq_codebook_sha256"]) == 64 and len(m["vq_decoder_sha256"]) == 64
    assert m["token_suffix_to_codebook_index"] == "identity"
    assert m["codebook_size"] == 256 and m["token_count"] == 64


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    cfg = TokenizerConfig()
    model = VQVAE(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ck = str(tmp_path / "ckpt_0.pt")
    save_checkpoint(ck, model, opt, step=0, cfg=cfg, corpus_hash="deadbeef")
    m2, loaded, cfg2 = F.load_model_from_checkpoint(ck)
    assert loaded["lut_corpus_hash"] == "deadbeef"
    assert torch.allclose(m2.vq.codebook, model.vq.codebook)
    assert cfg2.arch_version == cfg.arch_version


def test_freeze_aborts_on_failing_gate(tmp_path):
    """Untrained model -> huge ΔE -> gate fails -> freeze writes nothing."""
    torch.manual_seed(0)
    cfg = TokenizerConfig()
    model = VQVAE(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ck = str(tmp_path / "ckpt_0.pt")
    save_checkpoint(ck, model, opt, step=0, cfg=cfg, corpus_hash="x")
    dev = _records(tmp_path, 8)
    out = str(tmp_path / "final")
    ok, report = F.freeze(ck, out, dev, allow_exception=True, log_fn=lambda *_: None)
    assert ok is False
    assert not os.path.exists(os.path.join(out, "manifest.json"))
    assert "gate" in report and report["gate"]["pass"] is False
