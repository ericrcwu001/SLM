"""Parity: tokenizer.color_torch (differentiable) vs eval.color_pipeline (authoritative).

The training L_deltaE must optimize the same CIEDE2000 the eval/gate reports, so the
torch port has to match the NumPy reference. We check sRGB->Lab and CIEDE2000 on random
in-gamut samples, plus that the loss path is differentiable and finite.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eval import color_pipeline as npc
from tokenizer import color_torch as tc


def _rng():
    return np.random.default_rng(1234)


def test_srgb_to_lab_matches_numpy():
    rgb = _rng().random((512, 3)).astype(np.float64)
    lab_np = npc.srgb_to_lab_d65(rgb)
    lab_t = tc.srgb_to_lab_d65(torch.tensor(rgb, dtype=torch.float64)).numpy()
    assert np.allclose(lab_np, lab_t, atol=1e-8, rtol=0), np.abs(lab_np - lab_t).max()


def test_ciede2000_matches_numpy():
    rng = _rng()
    lab_a = npc.srgb_to_lab_d65(rng.random((1000, 3)))
    lab_b = npc.srgb_to_lab_d65(rng.random((1000, 3)))
    de_np = npc.ciede2000(lab_a, lab_b)
    de_t = tc.ciede2000(
        torch.tensor(lab_a, dtype=torch.float64), torch.tensor(lab_b, dtype=torch.float64)
    ).numpy()
    # 1e-4 tolerance absorbs the tiny sqrt(+1e-12) gradient guard in the torch port.
    assert np.allclose(de_np, de_t, atol=1e-4, rtol=0), np.abs(de_np - de_t).max()


def test_ciede2000_zero_on_identical():
    lab = npc.srgb_to_lab_d65(_rng().random((64, 3)))
    de = tc.ciede2000(torch.tensor(lab, dtype=torch.float64), torch.tensor(lab, dtype=torch.float64))
    assert float(de.max()) < 1e-5


def test_deltae_is_differentiable_and_finite():
    a = torch.rand(256, 3, dtype=torch.float64, requires_grad=True)
    b = torch.rand(256, 3, dtype=torch.float64)
    loss = tc.deltae2000_srgb(a, b).mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert a.grad is not None and torch.isfinite(a.grad).all()
