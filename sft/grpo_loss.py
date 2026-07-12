"""GRPO clipped-surrogate + KL loss over the 64-code assistant span (docs/grpo/03_grpo_loss.md).

Given a rollout buffer's per-token logprobs (current / old / reference), the group-relative advantages,
and the 64-code mask, produce the scalar to ``.backward()`` plus logging stats for the Doc 05
anti-hacking watch. Analogous to :mod:`sft.soft_loss` — torch is lazy-imported so the module imports
without the ``sft`` extra.

The math (Doc 03 §2):

  * per-token importance ratio ``ρ = exp(logp_new − logp_old)`` (log-ratio clamped to ±20);
  * clipped surrogate ``min(ρ·A, clip(ρ, 1−ε, 1+ε)·A)`` with the scalar advantage ``A_i`` broadcast to
    every one of completion ``i``'s 64 code tokens (outcome supervision, no per-token credit);
  * k3 KL to the frozen reference ``exp(s) − s − 1`` with ``s = logp_ref − logp_new`` (clamped),
    which is ``≥ 0`` always so it can never push KL negative and destabilize;
  * masked token-mean over the 64 code positions:
    ``loss = −((surr − β·kl)·sel).sum() / sel.sum().clamp(min=1)``.

Because every grade completion has exactly 64 masked code positions, this token-mean equals the Doc 03
§2 per-sequence ``(1/|o|)Σ`` average (they coincide at ``|o| = 64``).
"""

from __future__ import annotations

_LOGRATIO_CLAMP = 20.0


def grpo_loss(logp_new, logp_old, logp_ref, adv, sel, *, clip_eps: float, kl_beta: float):
    """GRPO loss over the masked 64-code span. Returns ``(loss, stats)``.

    Shapes: ``logp_new`` / ``logp_old`` / ``logp_ref`` / ``sel`` are ``[B, T-1]`` (logprobs are 0.0 off
    the code span; ``sel`` is the boolean/0-1 mask). ``adv`` is ``[B, 1]`` (the scalar advantage of each
    completion, broadcast over its tokens). ``clip_eps`` = ε, ``kl_beta`` = β.

    ``stats`` (all detached floats, for the guard panel): ``ratio_mean``, ``clip_fraction``, ``kl_mean``,
    ``adv_abs_mean``, ``entropy_proxy`` (mean ``−logp_new`` over selected tokens — a monotone
    entropy-adjacent signal; the TRUE rollout entropy over the 256-code support is measured in
    :mod:`sft.rollout` and merged by the loop), ``n_tokens``, ``n_samples``.
    """
    import torch

    selm = sel.to(logp_new.dtype)

    logratio = (logp_new - logp_old).clamp(-_LOGRATIO_CLAMP, _LOGRATIO_CLAMP)   # overflow guard (§5)
    ratio = torch.exp(logratio)
    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    policy = torch.min(surr1, surr2)                               # per-token clipped surrogate

    s = (logp_ref - logp_new).clamp(-_LOGRATIO_CLAMP, _LOGRATIO_CLAMP)
    kl = torch.exp(s) - s - 1.0                                    # k3, per token, >= 0

    per_tok = policy - kl_beta * kl
    n = selm.sum().clamp(min=1)
    loss = -(per_tok * selm).sum() / n                             # masked token-mean (== §2 at |o|=64)

    with torch.no_grad():
        clipped = ((ratio - 1.0).abs() > clip_eps).to(logp_new.dtype)
        stats = {
            "loss": float(loss),
            "ratio_mean": float((ratio * selm).sum() / n),
            "clip_fraction": float((clipped * selm).sum() / n),
            "kl_mean": float((kl * selm).sum() / n),
            "adv_abs_mean": float(adv.abs().mean()) if adv.numel() else 0.0,
            "entropy_proxy": float((-logp_new * selm).sum() / n),
            "n_tokens": int(selm.sum()),
            "n_samples": int(adv.shape[0]),
        }
    return loss, stats
