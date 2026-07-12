"""GRPO loss math tests (sft.grpo_loss; docs/grpo/03_grpo_loss.md §9).

Pure torch on synthetic tensors — no model, no frozen weights. Verifies the ρ≡1 first-step identity,
that ρ moves off 1 when the policy differs, KL non-negativity, masked-mean parity, and the stats.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sft.grpo_loss import grpo_loss


def _tensors(B=3, T=65, seed=0):
    """logp over [B,T-1] with exactly 64 selected code positions per row (positions 1..64)."""
    g = torch.Generator().manual_seed(seed)
    logp_old = -torch.rand(B, T - 1, generator=g)          # <= 0 (log-probs)
    sel = torch.zeros(B, T - 1, dtype=torch.bool)
    sel[:, 1:65] = True                                    # 64 code positions
    logp_old = logp_old.masked_fill(~sel, 0.0)
    adv = torch.tensor([[1.0], [-0.5], [0.0]][:B])
    return logp_old, sel, adv


def test_rho_identity_first_step():
    """logp_new == logp_old => ρ ≡ 1, clip inactive, L^clip == A per token; with ref==new, KL == 0."""
    logp, sel, adv = _tensors()
    loss, stats = grpo_loss(logp, logp, logp, adv, sel, clip_eps=0.2, kl_beta=0.05)
    assert stats["ratio_mean"] == pytest.approx(1.0, abs=1e-4)
    assert stats["clip_fraction"] == pytest.approx(0.0, abs=1e-6)
    assert stats["kl_mean"] == pytest.approx(0.0, abs=1e-6)
    # loss == -mean_token(A): every selected token carries its row's advantage A_i
    expected = -float((adv * sel.float()).sum() / sel.sum())
    assert float(loss) == pytest.approx(expected, abs=1e-6)


def test_rho_moves_off_one_when_policy_differs():
    logp_old, sel, adv = _tensors()
    logp_new = logp_old + 0.3 * sel.float()                # policy drifted on the code span
    _, stats = grpo_loss(logp_new, logp_old, logp_new, adv, sel, clip_eps=0.2, kl_beta=0.05)
    assert stats["ratio_mean"] == pytest.approx(float(torch.exp(torch.tensor(0.3))), rel=1e-3)


def test_kl_nonnegative_elementwise():
    """The k3 estimator exp(s)-s-1 is >= 0 for any s; a positive KL must appear when ref != new."""
    logp_old, sel, adv = _tensors(seed=1)
    logp_new = logp_old.clone()
    logp_ref = logp_old - 0.4 * sel.float()                # ref differs from new on the span
    _, stats = grpo_loss(logp_new, logp_old, logp_ref, adv, sel, clip_eps=0.2, kl_beta=0.1)
    assert stats["kl_mean"] >= 0.0
    assert stats["kl_mean"] > 0.0                          # genuinely positive when ref != new
    # exhaustive elementwise check on a random s
    s = torch.randn(1000)
    kl = torch.exp(s.clamp(-20, 20)) - s.clamp(-20, 20) - 1.0
    assert bool((kl >= -1e-6).all())


def test_masked_mean_equals_per_sequence_mean_at_64():
    """Token-mean over the group == mean over sequences of the per-sequence mean (coincide at |o|=64)."""
    logp_old, sel, adv = _tensors(B=3)
    logp_new = logp_old + 0.1 * sel.float()
    loss, _ = grpo_loss(logp_new, logp_old, logp_new, adv, sel, clip_eps=0.2, kl_beta=0.0)

    # per-sequence: each row normalized by its own 64 positions, then averaged over rows
    logratio = (logp_new - logp_old).clamp(-20, 20)
    ratio = torch.exp(logratio)
    policy = torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv)
    per_seq = -(policy * sel.float()).sum(dim=1) / sel.float().sum(dim=1)   # [B]
    assert float(loss) == pytest.approx(float(per_seq.mean()), abs=1e-6)


def test_adv_accepts_1d_or_2d():
    """grpo_loss must accept adv as [B] or [B,1] (defensive shape guard) — same loss either way."""
    logp, sel, adv = _tensors()
    l2, _ = grpo_loss(logp, logp, logp, adv, sel, clip_eps=0.2, kl_beta=0.05)          # [B,1]
    l1, _ = grpo_loss(logp, logp, logp, adv.squeeze(1), sel, clip_eps=0.2, kl_beta=0.05)  # [B]
    assert float(l1) == pytest.approx(float(l2))


def test_loss_detaches_old_and_ref():
    """The cached old/ref logprobs must not receive gradient even if a caller passes grad-carrying ones."""
    B, T = 2, 65
    sel = torch.zeros(B, T - 1, dtype=torch.bool)
    sel[:, 1:65] = True
    base = torch.randn(B, T - 1).masked_fill(~sel, 0.0)
    logp_old = base.detach().clone().requires_grad_(True)
    logp_ref = base.detach().clone().requires_grad_(True)
    logp_new = (base.detach().clone() + 0.1 * sel.float()).requires_grad_(True)
    adv = torch.tensor([[1.0], [-0.5]])
    loss, _ = grpo_loss(logp_new, logp_old, logp_ref, adv, sel, clip_eps=0.2, kl_beta=0.1)
    loss.backward()
    assert logp_new.grad is not None            # grad flows into the current policy
    assert logp_old.grad is None                # detached inside the loss
    assert logp_ref.grad is None


def test_clip_fraction_counts_out_of_band_tokens():
    logp_old, sel, adv = _tensors()
    logp_new = logp_old + 1.0 * sel.float()                # ratio ~ e^1 >> 1+eps everywhere selected
    _, stats = grpo_loss(logp_new, logp_old, logp_new, adv, sel, clip_eps=0.2, kl_beta=0.0)
    assert stats["clip_fraction"] == pytest.approx(1.0, abs=1e-6)
    assert stats["n_tokens"] == int(sel.sum())
    assert stats["n_samples"] == adv.shape[0]


# --- gradient locality: grads flow to the trainable param, never the frozen one -------------------
class _TinyPolicy(torch.nn.Module):
    """Logits = frozen_base(x) + trainable_delta(x). Mimics 'grad through LoRA only, base frozen'."""

    def __init__(self, vocab: int):
        super().__init__()
        self.frozen = torch.nn.Embedding(vocab, vocab)          # stands in for the NF4 base
        self.delta = torch.nn.Embedding(vocab, vocab)           # stands in for the LoRA/head params
        for p in self.frozen.parameters():
            p.requires_grad_(False)
        self.device = torch.device("cpu")

    def forward(self, *, input_ids, attention_mask=None, **kw):
        from types import SimpleNamespace
        logits = self.frozen(input_ids) + self.delta(input_ids)
        return SimpleNamespace(logits=logits)


def test_gradient_locality_base_frozen():
    from sft.rollout import code_logprobs, init_code_maps
    from tests.test_generate import _FakeTokenizer

    init_code_maps(_FakeTokenizer())        # codes 2000..2255
    vocab = 2300
    model = _TinyPolicy(vocab)
    n_prompt = 3
    codes = list(range(64))
    seq = [1, 2, 3, 1000] + [2000 + c for c in codes] + [1001]     # prompt + BOS + 64 codes + EOS
    input_ids = torch.tensor([seq])
    labels = input_ids.clone()
    labels[:, :n_prompt] = -100
    batch = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels}

    logp_new, sel = code_logprobs(model, batch)
    assert int(sel.sum()) == 64
    with torch.no_grad():
        logp_old = logp_new.clone()
    adv = torch.ones(1, 1)
    loss, _ = grpo_loss(logp_new, logp_old, logp_new, adv, sel, clip_eps=0.2, kl_beta=0.05)
    loss.backward()

    assert model.delta.weight.grad is not None and model.delta.weight.grad.abs().sum() > 0
    assert model.frozen.weight.grad is None      # nothing flows into the frozen 'base'
