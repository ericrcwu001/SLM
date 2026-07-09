"""MLX 7-term VQ-tokenizer loss — mirror of tokenizer.losses (v2).

Terms (weights from TokenizerConfig): L_recon (MSE on residual, up-weighted in v2 to drive
PSNR), L_deltaE (mean CIEDE2000 over all nodes), L_tail (mean of the worst `tail_frac`
node ΔE per LUT — targets the p95/p99 tail), L_smooth (3D 2nd-difference), L_clip
(pre-clamp out-of-range), L_neutral (TARGET-RELATIVE neutral-diagonal chroma =
|chroma(recon)-chroma(target)|, zero at perfect reconstruction), L_commit (from the VQ).

Channels-last (NDHWC). Absolute LUT = residual + identity, clamped to [0,1] for color
terms. ``total_loss`` returns the differentiable MLX scalar (for value_and_grad);
``components`` returns a float breakdown for logging (call OUTSIDE the grad transform).
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from eval.cube_io import identity_grid

from . import color_mlx as cm
from ..config import TokenizerConfig

_IDENTITY_CACHE: dict[int, mx.array] = {}


def identity_chlast(grid: int) -> mx.array:
    if grid not in _IDENTITY_CACHE:
        _IDENTITY_CACHE[grid] = mx.array(np.asarray(identity_grid(grid), dtype=np.float32))
    return _IDENTITY_CACHE[grid]


def smoothness(recon: mx.array) -> mx.array:
    total = mx.zeros(())
    total = total + ((recon[:, :-2] - 2.0 * recon[:, 1:-1] + recon[:, 2:]) ** 2).mean()
    total = total + ((recon[:, :, :-2] - 2.0 * recon[:, :, 1:-1] + recon[:, :, 2:]) ** 2).mean()
    total = total + ((recon[:, :, :, :-2] - 2.0 * recon[:, :, :, 1:-1] + recon[:, :, :, 2:]) ** 2).mean()
    return total / 3.0


def _terms(out: dict, target: mx.array, cfg: TokenizerConfig, identity: mx.array) -> dict:
    recon = out["recon"]                                  # [N,r,g,b,3]
    l_recon = mx.mean((recon - target) ** 2)

    recon_abs_raw = recon + identity
    recon_abs = mx.clip(recon_abs_raw, 0.0, 1.0)
    target_abs = mx.clip(target + identity, 0.0, 1.0)

    node_de = cm.deltae2000_srgb(recon_abs, target_abs)   # [N,r,g,b]
    l_deltae = node_de.mean()

    # L_tail: mean of the worst tail_frac node ΔE per LUT (topk = k largest).
    flat_de = node_de.reshape(node_de.shape[0], -1)
    n_nodes = flat_de.shape[1]
    k = max(1, int(cfg.tail_frac * n_nodes))
    l_tail = mx.topk(flat_de, k, axis=1).mean()

    l_smooth = smoothness(recon)

    over = mx.maximum(recon_abs_raw - 1.0, 0.0)
    under = mx.maximum(-recon_abs_raw, 0.0)
    l_clip = (over ** 2 + under ** 2).mean()

    # L_neutral: target-relative diagonal chroma match (0 at perfect reconstruction).
    n = cfg.grid
    idx = list(range(n))
    rec_diag = mx.stack([recon_abs[:, i, i, i, :] for i in idx], axis=1)   # [N,n,3]
    tgt_diag = mx.stack([target_abs[:, i, i, i, :] for i in idx], axis=1)
    lab_r = cm.srgb_to_lab_d65(rec_diag)
    lab_t = cm.srgb_to_lab_d65(tgt_diag)
    chroma_r = mx.sqrt(lab_r[..., 1] ** 2 + lab_r[..., 2] ** 2 + 1e-12)
    chroma_t = mx.sqrt(lab_t[..., 1] ** 2 + lab_t[..., 2] ** 2 + 1e-12)
    l_neutral = mx.abs(chroma_r - chroma_t).mean()

    return {"recon": l_recon, "deltaE": l_deltae, "tail": l_tail, "smooth": l_smooth,
            "clip": l_clip, "neutral": l_neutral, "commit": out["commit_loss"]}


def total_loss(out: dict, target: mx.array, cfg: TokenizerConfig, identity: mx.array | None = None) -> mx.array:
    if identity is None:
        identity = identity_chlast(cfg.grid)
    t = _terms(out, target, cfg, identity)
    return (cfg.w_recon * t["recon"] + cfg.w_deltaE * t["deltaE"] + cfg.w_tail * t["tail"]
            + cfg.w_smooth * t["smooth"] + cfg.w_clip * t["clip"] + cfg.w_neutral * t["neutral"]
            + cfg.w_commit * t["commit"])


def components(out: dict, target: mx.array, cfg: TokenizerConfig, identity: mx.array | None = None) -> dict[str, float]:
    if identity is None:
        identity = identity_chlast(cfg.grid)
    t = _terms(out, target, cfg, identity)
    total = total_loss(out, target, cfg, identity)
    mx.eval(total, *t.values(), out["perplexity"])
    d = {k: float(v) for k, v in t.items()}
    d["loss"] = float(total)
    d["perplexity"] = float(out["perplexity"])
    return d
