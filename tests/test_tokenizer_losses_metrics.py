"""Loss integration (forward/backward, single-batch overfit) + numpy gate metrics.

All synthetic, CPU, seconds. No dataset, no checkpoints, no real training run.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tokenizer import metrics
from tokenizer.config import TokenizerConfig
from tokenizer.losses import total_loss
from tokenizer.model import VQVAE, residual_to_input


def _batch(n=4, seed=0):
    rng = np.random.default_rng(seed)
    res = rng.standard_normal((n, 17, 17, 17, 3)).astype(np.float32) * 0.05
    return residual_to_input(res)


def test_total_loss_is_finite_with_all_terms():
    torch.manual_seed(0)
    model = VQVAE(TokenizerConfig())
    model.train()
    x = _batch()
    out = model(x)
    loss, comp = total_loss(out, x, model.cfg)
    assert torch.isfinite(loss)
    for k in ("recon", "deltaE", "smooth", "clip", "neutral", "commit", "perplexity"):
        assert k in comp and np.isfinite(comp[k])


def test_backward_populates_finite_grads():
    torch.manual_seed(0)
    model = VQVAE(TokenizerConfig())
    model.train()
    x = _batch()
    loss, _ = total_loss(model(x), x, model.cfg)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)


def test_single_batch_overfit_reduces_loss():
    """~60 steps on one fixed 4-LUT batch should drive the loss down — sanity that the
    encoder/decoder/VQ + losses actually learn (not a quality gate)."""
    torch.manual_seed(0)
    model = VQVAE(TokenizerConfig())
    model.train()
    x = _batch()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    first = None
    for step in range(60):
        opt.zero_grad()
        loss, _ = total_loss(model(x), x, model.cfg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 0:
            first = float(loss.detach())
    last = float(loss.detach())
    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"


def test_metrics_perfect_reconstruction_passes_gate():
    rng = np.random.default_rng(0)
    targets = [rng.standard_normal((17, 17, 17, 3)) * 0.05 for _ in range(40)]
    recons = [t.copy() for t in targets]                      # perfect
    agg = metrics.aggregate_reconstruction(targets, recons)
    assert agg["overall"]["mean_deltae"] < 1e-6
    assert agg["overall"]["mean_psnr"] >= 35.0
    cb = metrics.codebook_stats(np.arange(256), 256)
    res = metrics.evaluate_gate(agg, cb)
    assert res["pass"] is True


def test_codebook_stats_collapse_detected():
    all_dead_but_one = np.zeros(1000, dtype=np.int64)         # every code == 0
    cb = metrics.codebook_stats(all_dead_but_one, 256)
    assert cb["active_codes"] == 1
    assert cb["dead_code_count"] == 255
    assert cb["top_code_share"] == 1.0


def test_roundtrip_contracts():
    torch.manual_seed(0)
    model = VQVAE(TokenizerConfig())
    res = metrics.roundtrip_contracts(model)
    assert res["pass"] is True, res["checks"]
