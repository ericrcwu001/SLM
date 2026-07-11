"""Codebook-embedding-weighted soft-target loss (Phase 3 C — a collapse fix).

Plain hard-label cross-entropy treats all 256 codes as equidistant: predicting a perceptually-near
code is penalized exactly as much as a far one, leaving the model's 256-way code distribution flat
and easy to collapse under free-running greedy decode. This adds an AUXILIARY term on the code
positions whose target is a softmax over codebook *distances* — near codes (in the frozen [256,64]
embedding space) share probability mass — so the gradient rewards perceptual closeness.

The total is ``hard_ce + weight * soft_ce`` where ``hard_ce`` is the standard token CE over the whole
assistant span (so the ``<lut_bos>``/``<lut_eos>`` control tokens stay supervised) and ``soft_ce`` is
the soft-target CE over code positions only. ``weight == 0`` returns exactly ``hard_ce`` — identical
to the baseline trainer — so the knob is safe-by-default. A methodology knob, NOT in the locked
bilevel search space (AGENTS.md).

torch is imported lazily inside the functions so the module imports without the ``sft`` extra.
"""

from __future__ import annotations


def load_codebook_tensor(device=None, dtype=None):
    """Load the frozen ``[256,64]`` codebook as a tensor (``codebook.npy``, else ``model.pt``)."""
    import numpy as np
    import torch

    from tokenizer.frozen import frozen_final_dir
    dtype = dtype or torch.float32
    try:
        cb = np.load(frozen_final_dir() / "codebook.npy")
        return torch.tensor(cb, dtype=dtype, device=device)
    except Exception:  # noqa: BLE001
        from tokenizer.frozen import load_frozen_vqvae
        model, _ = load_frozen_vqvae()
        return model.vq.codebook.detach().to(device=device, dtype=dtype)


def code_soft_targets(codebook, tau: float = 1.0):
    """Row-stochastic ``[K,K]`` soft-target matrix: row ``g`` = softmax(-dist(cb[g], cb[k]) / tau).

    Distances are normalized by their median first, so ``tau`` is scale-free (interpretable across
    codebooks). Each row is peaked at ``g`` itself (distance 0) with mass on perceptually-near codes.
    """
    import torch

    cb = codebook.float()
    d = torch.cdist(cb, cb)                         # [K, K]
    med = d[d > 0].median()
    d = d / (med + 1e-8)
    return torch.softmax(-d / max(float(tau), 1e-6), dim=-1)


def soft_label_loss(logits, labels, code_token_ids, soft_targets, *, weight: float):
    """``hard_ce`` (whole assistant span) + ``weight`` * soft-target CE on code positions.

    ``logits`` ``[B,T,V]``; ``labels`` ``[B,T]`` (``-100`` ignored, assistant-only); ``code_token_ids``
    ``LongTensor[256]`` = vocab ids of the codes in codebook-index order; ``soft_targets`` ``[256,256]``
    (gold codebook-index -> target distribution over codebook indices). ``weight == 0`` -> exactly the
    standard token CE.
    """
    import torch
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    vocab = shift_logits.size(-1)
    flat_logits = shift_logits.reshape(-1, vocab)
    flat_labels = shift_labels.reshape(-1)
    hard = F.cross_entropy(flat_logits, flat_labels, ignore_index=-100)
    if weight <= 0:
        return hard

    is_code = torch.isin(flat_labels, code_token_ids)
    if not bool(is_code.any()):
        return hard
    code_logits = flat_logits[is_code][:, code_token_ids]              # [N, 256] over codes only
    gold_tok = flat_labels[is_code]                                    # [N] vocab ids
    gold_idx = (code_token_ids.unsqueeze(0) == gold_tok.unsqueeze(1)).float().argmax(1)  # [N]
    q = soft_targets[gold_idx]                                         # [N, 256] target dists
    logp = F.log_softmax(code_logits, dim=-1)
    soft = -(q * logp).sum(-1).mean()
    return hard + float(weight) * soft
