"""VQ-tokenizer training loss (model_architecture.md / training_plan_colab.md Stage 1).

Six terms, weighted by :class:`tokenizer.config.TokenizerConfig`:

  L_recon   MSE on the reconstructed residual grid
  L_deltaE  CIEDE2000 (differentiable torch port) on the absolute LUT nodes
  L_smooth  3D second-difference (Laplacian) smoothness on the reconstructed residual
  L_clip    penalty for absolute LUT values outside [0,1] (pre-clamp)
  L_neutral chroma penalty on the neutral diagonal (r=g=b should stay neutral)
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
    l_deltae = tc.deltae2000_srgb(recon_abs, target_abs).mean()

    # -- L_smooth --
    l_smooth = smoothness(recon)

    # -- L_clip (penalize pre-clamp absolute outside [0,1]) --
    recon_abs_raw = _to_channel_last(recon) + identity_chlast
    over = F.relu(recon_abs_raw - 1.0)
    under = F.relu(-recon_abs_raw)
    l_clip = (over.pow(2) + under.pow(2)).mean()

    # -- L_neutral (chroma on the r=g=b diagonal should stay ~0) --
    n = cfg.grid
    diag = torch.arange(n, device=device)
    neutral_abs = recon_abs[:, diag, diag, diag, :]   # [N, n, 3]
    lab = tc.srgb_to_lab_d65(neutral_abs)
    chroma = torch.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2 + 1e-12)
    l_neutral = chroma.mean()

    # -- L_commit (already beta-scaled inside the VQ) --
    l_commit = out["commit_loss"]

    loss = (
        cfg.w_recon * l_recon
        + cfg.w_deltaE * l_deltae
        + cfg.w_smooth * l_smooth
        + cfg.w_clip * l_clip
        + cfg.w_neutral * l_neutral
        + cfg.w_commit * l_commit
    )

    components = {
        "loss": float(loss.detach()),
        "recon": float(l_recon.detach()),
        "deltaE": float(l_deltae.detach()),
        "smooth": float(l_smooth.detach()),
        "clip": float(l_clip.detach()),
        "neutral": float(l_neutral.detach()),
        "commit": float(l_commit.detach()),
        "perplexity": float(out["perplexity"].detach()),
    }
    return loss, components
