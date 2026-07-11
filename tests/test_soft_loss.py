"""Tests for the codebook-embedding-weighted soft-target loss (sft.soft_loss).

Pure torch (available in the test env); no model or frozen weights needed — the codebook is a small
synthetic tensor. Verifies weight=0 is byte-identical to plain token CE (safe-by-default) and that
the soft-target matrix is row-stochastic and peaked at the gold code.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sft.soft_loss import code_soft_targets, soft_label_loss


def test_soft_targets_row_stochastic_and_peaked():
    cb = torch.tensor([[0.0, 0.0], [1.0, 0.0], [5.0, 0.0]])
    st = code_soft_targets(cb, tau=1.0)
    assert st.shape == (3, 3)
    assert torch.allclose(st.sum(-1), torch.ones(3), atol=1e-5)   # each row is a distribution
    assert (st.argmax(-1) == torch.arange(3)).all()               # peaked at self (distance 0)
    # nearer code gets more mass than farther code
    assert st[0, 1] > st[0, 2]


def test_weight_zero_equals_plain_cross_entropy():
    import torch.nn.functional as F
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 6)
    labels = torch.tensor([[-100, 4, 5, 4]])          # 3 assistant positions, all code tokens
    code_ids = torch.tensor([4, 5])
    st = code_soft_targets(torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
    got = soft_label_loss(logits, labels, code_ids, st, weight=0.0)
    ref = F.cross_entropy(logits[:, :-1, :].reshape(-1, 6), labels[:, 1:].reshape(-1), ignore_index=-100)
    assert torch.allclose(got, ref)


def test_weight_adds_positive_soft_term():
    torch.manual_seed(1)
    logits = torch.randn(1, 4, 6)
    labels = torch.tensor([[-100, 4, 5, 4]])
    code_ids = torch.tensor([4, 5])
    st = code_soft_targets(torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
    l0 = soft_label_loss(logits, labels, code_ids, st, weight=0.0)
    l1 = soft_label_loss(logits, labels, code_ids, st, weight=0.5)
    assert float(l1) > float(l0)      # the auxiliary soft CE is strictly positive here


def test_no_code_positions_falls_back_to_hard():
    import torch.nn.functional as F
    logits = torch.randn(1, 3, 6)
    labels = torch.tensor([[-100, 1, 2]])             # no code tokens (codes are 4,5)
    code_ids = torch.tensor([4, 5])
    st = code_soft_targets(torch.tensor([[0.0, 0.0], [1.0, 0.0]]))
    got = soft_label_loss(logits, labels, code_ids, st, weight=0.9)
    ref = F.cross_entropy(logits[:, :-1, :].reshape(-1, 6), labels[:, 1:].reshape(-1), ignore_index=-100)
    assert torch.allclose(got, ref)
