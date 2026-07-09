"""MLX v2 tokenizer: color parity, geometry, MLX->torch round-trip fidelity, overfit.

Skipped entirely if mlx is not installed (Apple-Silicon only). The round-trip tests are
the correctness guarantee for training on MLX then freezing the torch model: the converted
torch VQVAE must reproduce the MLX model's encoder latent + decode within tolerance and its
codes exactly.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as mnn          # noqa: E402
import mlx.optimizers as optim  # noqa: E402
import torch                  # noqa: E402

from eval import color_pipeline as npc          # noqa: E402
from tokenizer.config import TokenizerConfig     # noqa: E402
from tokenizer.mlx import color_mlx as cm        # noqa: E402
from tokenizer.mlx import losses_mlx as L        # noqa: E402
from tokenizer.mlx.convert import mlx_to_torch_vqvae  # noqa: E402
from tokenizer.mlx.model_mlx import VQVAEmlx     # noqa: E402


def _spread_model(seed=0):
    m = VQVAEmlx(TokenizerConfig())
    m.vq._codebook = mx.random.normal((256, 64)) * 0.5   # separate codes so assignment is well-defined
    mx.eval(m.parameters(), m.vq._codebook)
    return m


# --- color parity (float32 MLX vs float64 numpy) ---
def test_color_ciede2000_parity():
    rng = np.random.default_rng(0)
    la = npc.srgb_to_lab_d65(rng.random((2000, 3)))
    lb = npc.srgb_to_lab_d65(rng.random((2000, 3)))
    de_np = npc.ciede2000(la, lb)
    de_mx = np.array(cm.ciede2000(mx.array(la.astype(np.float32)), mx.array(lb.astype(np.float32))))
    assert np.abs(de_np - de_mx).max() < 1e-3          # float32 tolerance


# --- geometry (v2: resize decoder) ---
def test_geometry():
    m = VQVAEmlx(TokenizerConfig())
    x = mx.zeros((2, 17, 17, 17, 3))
    z = m.encoder(x); mx.eval(z)
    assert z.shape == (2, 4, 4, 4, 64)
    out = m(x); mx.eval(out["recon"])
    assert out["recon"].shape == (2, 17, 17, 17, 3)
    assert out["codes"].shape == (2, 64)


# --- MLX -> torch round-trip fidelity (the linchpin) ---
def test_convert_strict_load_and_roundtrip():
    cfg = TokenizerConfig()
    m = _spread_model()
    tm = mlx_to_torch_vqvae(m, cfg)                    # strict load must succeed

    rng = np.random.default_rng(1)
    z_max = d_max = 0.0
    code_match = 0
    N = 6
    for _ in range(N):
        res = rng.standard_normal((17, 17, 17, 3)) * 0.05
        zl_mlx = np.array(m.encoder(mx.array(res[None].astype(np.float32))))
        zl_t = tm.encoder(torch.tensor(res[None]).float().permute(0, 4, 1, 2, 3)) \
                 .permute(0, 2, 3, 4, 1).detach().numpy()
        z_max = max(z_max, float(np.abs(zl_mlx - zl_t).max()))
        c_mlx, c_t = m.encode(res), tm.encode(res)
        code_match += int(c_mlx == c_t)
        d_max = max(d_max, float(np.abs(m.decode(c_mlx) - tm.decode(c_mlx)).max()))

    assert z_max < 1e-4, f"encoder-latent parity {z_max}"
    assert d_max < 1e-4, f"decode parity {d_max}"
    assert code_match == N, f"codes matched {code_match}/{N} (float32/float64 tie?)"


# --- 7-term loss trains (no NaN, decreases) ---
def test_mlx_overfit_reduces_loss():
    cfg = replace(TokenizerConfig(), batch_size=4)
    m = VQVAEmlx(cfg); mx.eval(m.parameters())
    opt = optim.AdamW(learning_rate=3e-4, weight_decay=1e-4)
    identity = L.identity_chlast(cfg.grid)
    x = mx.array(np.random.default_rng(0).standard_normal((4, 17, 17, 17, 3)).astype(np.float32) * 0.05)

    def loss_fn(model, xb):
        return L.total_loss(model(xb), xb, cfg, identity)

    lag = mnn.value_and_grad(m, loss_fn)
    first = None
    for step in range(40):
        loss, grads = lag(m, x)
        grads, _ = optim.clip_grad_norm(grads, cfg.grad_clip)
        opt.update(m, grads)
        m.vq.ema_update(m.encoder(x))
        mx.eval(m.parameters(), opt.state, m.vq._codebook, loss)
        if step == 0:
            first = float(loss)
    last = float(loss)
    assert np.isfinite(last) and last < first, f"loss {first:.4f} -> {last:.4f}"


def test_all_seven_loss_terms_present():
    cfg = TokenizerConfig()
    m = VQVAEmlx(cfg)
    x = mx.array(np.random.default_rng(0).standard_normal((2, 17, 17, 17, 3)).astype(np.float32) * 0.05)
    comp = L.components(m(x), x, cfg)
    for k in ("loss", "recon", "deltaE", "tail", "smooth", "clip", "neutral", "commit", "perplexity"):
        assert k in comp and np.isfinite(comp[k])
