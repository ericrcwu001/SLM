"""VQ-tokenizer training loss (model_architecture.md / training_plan_colab.md Stage 1).

Seven terms, weighted by :class:`tokenizer.config.TokenizerConfig`:

  L_recon   MSE on the reconstructed residual grid (drives PSNR; up-weighted after audit)
  L_deltaE  CIEDE2000 (differentiable torch port), mean over the absolute LUT nodes
  L_tail    mean of the worst ``tail_frac`` per-node ΔE per LUT (targets the p95/p99 tail
            that the uniform mean ignores but the acceptance gate thresholds)
  L_smooth  3D second-difference (Laplacian) smoothness on the reconstructed residual
  L_clip    penalty for absolute LUT values outside [0,1] (pre-clamp)
  L_neutral TARGET-RELATIVE neutral-diagonal chroma penalty: |chroma(recon)-chroma(target)|
            on r=g=b (was: recon chroma toward absolute zero, which wrongly fought the
            ~82% of LUTs that legitimately tint neutrals and dominated the gradient)
  L_commit  VQ commitment loss (already scaled by commit_beta inside the quantizer)

All terms operate on conv tensors ``[N,3,17,17,17]`` (channel-first). Absolute LUT =
residual + encoded-sRGB identity grid; ΔE / neutral use a [0,1]-clamped absolute,
mirroring the deterministic gamut clip in :func:`eval.color_pipeline.to_canonical_srgb`.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from eval.cube_io import identity_grid

from . import color_torch as tc
from .config import TokenizerConfig


def _identity_chlast(grid: int, device, dtype) -> torch.Tensor:
    """Encoded-sRGB identity as a channel-last tensor ``[grid,grid,grid,3]``."""
    return torch.as_tensor(identity_grid(grid), device=device, dtype=dtype)


def _to_channel_last(x: torch.Tensor) -> torch.Tensor:
    # [N,3,X,Y,Z] -> [N,X,Y,Z,3]
    return x.permute(0, 2, 3, 4, 1)


def smoothness(residual_conv: torch.Tensor) -> torch.Tensor:
    """Mean squared 3D second difference of the residual along X, Y, Z."""
    total = residual_conv.new_zeros(())
    for axis in (2, 3, 4):
        lo = residual_conv.narrow(axis, 0, residual_conv.shape[axis] - 2)
        mid = residual_conv.narrow(axis, 1, residual_conv.shape[axis] - 2)
        hi = residual_conv.narrow(axis, 2, residual_conv.shape[axis] - 2)
        total = total + (lo - 2.0 * mid + hi).pow(2).mean()
    return total / 3.0


def total_loss(
    out: dict,
    target_conv: torch.Tensor,
    cfg: TokenizerConfig,
    identity_chlast: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted sum of the six terms.

    ``out`` is the dict returned by :meth:`VQVAE.forward`; ``target_conv`` is the input
    residual (conv tensor). Returns (scalar loss tensor, per-term float dict for logging).
    """
    recon = out["recon"]                      # [N,3,17,17,17]
    device, dtype = recon.device, recon.dtype
    if identity_chlast is None:
        identity_chlast = _identity_chlast(cfg.grid, device, dtype)

    # -- L_recon (residual-grid MSE) --
    l_recon = F.mse_loss(recon, target_conv)

    # -- absolute LUTs (channel-last), clamped to [0,1] for color ops --
    recon_abs = (_to_channel_last(recon) + identity_chlast).clamp(0.0, 1.0)
    target_abs = (_to_channel_last(target_conv) + identity_chlast).clamp(0.0, 1.0)

    # -- L_deltaE (CIEDE2000 over every LUT node = the "chart") --
    node_de = tc.deltae2000_srgb(recon_abs, target_abs)      # [N,17,17,17]
    l_deltae = node_de.mean()

    # -- L_tail (mean of the worst tail_frac node ΔE per LUT; targets p95/p99/max) --
    n_nodes = node_de.shape[1] * node_de.shape[2] * node_de.shape[3]
    k = max(1, int(cfg.tail_frac * n_nodes))
    worst = node_de.reshape(node_de.shape[0], -1).topk(k, dim=1).values   # [N, k]
    l_tail = worst.mean()

    # -- L_smooth --
    l_smooth = smoothness(recon)

    # -- L_clip (penalize pre-clamp absolute outside [0,1]) --
    recon_abs_raw = _to_channel_last(recon) + identity_chlast
    over = F.relu(recon_abs_raw - 1.0)
    under = F.relu(-recon_abs_raw)
    l_clip = (over.pow(2) + under.pow(2)).mean()

    # -- L_neutral (target-relative: match the target's neutral-diagonal chroma) --
    # Penalizing recon chroma toward absolute 0 optimizes the wrong quantity — most
    # graded LUTs legitimately tint neutrals — so we penalize the *deviation from the
    # target's* diagonal chroma, which is 0 at perfect reconstruction.
    n = cfg.grid
    diag = torch.arange(n, device=device)
    lab_recon = tc.srgb_to_lab_d65(recon_abs[:, diag, diag, diag, :])   # [N, n, 3]
    lab_tgt = tc.srgb_to_lab_d65(target_abs[:, diag, diag, diag, :])
    chroma_recon = torch.sqrt(lab_recon[..., 1] ** 2 + lab_recon[..., 2] ** 2 + 1e-12)
    chroma_tgt = torch.sqrt(lab_tgt[..., 1] ** 2 + lab_tgt[..., 2] ** 2 + 1e-12)
    l_neutral = (chroma_recon - chroma_tgt).abs().mean()

    # -- L_commit (already beta-scaled inside the VQ) --
    l_commit = out["commit_loss"]

    loss = (
        cfg.w_recon * l_recon
        + cfg.w_deltaE * l_deltae
        + cfg.w_tail * l_tail
        + cfg.w_smooth * l_smooth
        + cfg.w_clip * l_clip
        + cfg.w_neutral * l_neutral
        + cfg.w_commit * l_commit
    )

    components = {
        "loss": float(loss.detach()),
        "recon": float(l_recon.detach()),
        "deltaE": float(l_deltae.detach()),
        "tail": float(l_tail.detach()),
        "smooth": float(l_smooth.detach()),
        "clip": float(l_clip.detach()),
        "neutral": float(l_neutral.detach()),
        "commit": float(l_commit.detach()),
        "perplexity": float(out["perplexity"].detach()),
    }
    return loss, components
