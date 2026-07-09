"""VQ-VAE geometry, code contract, and flatten-order invariants on synthetic tensors.

CPU-only, no training, no data — verifies the encoder/decoder resolve to the pinned
17->4->17 geometry, that encode/decode honor the numpy [r,g,b,3] contract and the
64-code range, and that the latent flatten order round-trips (quantize∘embed == id).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tokenizer.config import TokenizerConfig
from tokenizer.model import VQVAE, output_to_residual, residual_to_input


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return VQVAE(TokenizerConfig())


def test_encoder_decoder_geometry(model):
    x = torch.randn(2, 3, 17, 17, 17)
    z = model.encoder(x)
    assert z.shape == (2, model.cfg.code_dim, 4, 4, 4), z.shape
    recon = model.decoder(z)
    assert recon.shape == (2, 3, 17, 17, 17), recon.shape


def test_forward_shapes_and_code_range(model):
    model.train()
    x = torch.randn(4, 3, 17, 17, 17)
    out = model(x)
    assert out["recon"].shape == (4, 3, 17, 17, 17)
    assert out["codes"].shape == (4, 64)
    assert int(out["codes"].min()) >= 0 and int(out["codes"].max()) < 256
    assert out["commit_loss"].ndim == 0 and torch.isfinite(out["commit_loss"])
    assert out["perplexity"].ndim == 0 and torch.isfinite(out["perplexity"])


def test_axis_roundtrip():
    res = np.random.default_rng(0).random((17, 17, 17, 3)).astype(np.float64)
    back = output_to_residual(residual_to_input(res))[0].numpy()
    assert np.allclose(res, back, atol=1e-6)


def test_encode_returns_64_valid_codes(model):
    res = np.random.default_rng(1).standard_normal((17, 17, 17, 3)) * 0.05
    codes = model.encode(res)
    assert isinstance(codes, list) and len(codes) == 64
    assert all(isinstance(c, int) and 0 <= c < 256 for c in codes)


def test_decode_shape_and_dtype(model):
    codes = list(range(64))
    res = model.decode(codes)
    assert res.shape == (17, 17, 17, 3) and res.dtype == np.float64


def test_decode_rejects_bad_codes(model):
    with pytest.raises(ValueError):
        model.decode(list(range(63)))          # wrong count
    with pytest.raises(ValueError):
        model.decode([300] + list(range(63)))  # out of range


def test_encode_is_deterministic(model):
    res = np.random.default_rng(2).standard_normal((17, 17, 17, 3)) * 0.05
    assert model.encode(res) == model.encode(res)


def test_flatten_order_roundtrip(model):
    """quantize_indices(embed_codes(codes)) == codes when codebook rows are distinct.

    This is the invariant that the latent flatten order (token = x*16+y*4+z) is used
    consistently by both the encode and decode paths.
    """
    # make codebook rows well-separated so nearest-code of an exact row is itself
    with torch.no_grad():
        model.vq.codebook.copy_(torch.randn_like(model.vq.codebook) * 10.0)
    codes = torch.randint(0, 256, (3, 64))
    latent = model.vq.embed_codes(codes, model.cfg.latent_grid)
    recovered = model.vq.quantize_indices(latent)
    assert torch.equal(recovered, codes)
