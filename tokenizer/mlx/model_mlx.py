"""MLX VQ-VAE v2 — the Apple-GPU mirror of tokenizer.model.VQVAE (arch_version vq_v2_...).

Mirrors the v2 torch model exactly:
  * Encoder Conv3d 17->9->5->4 with `replicate` boundary padding.
  * Decoder trilinear-resize(align_corners=True) + Conv3d 4->5->9->17 (no ConvTranspose3d,
    so no Odena checkerboard), also `replicate` padding.
  * EMA vector-quantizer (codebook 256x64, dead-code revival, straight-through), nearest
    code = argmin_e(||e||^2 - 2 x.e) with lowest-index tie-break.

MLX specifics vs torch:
  * Channels-LAST (NDHWC): a residual LUT is [r,g,b,3] == [D,H,W,C], so no data permute.
  * `replicate` padding: MLX convs have no padding_mode, emulated with mx.pad(mode='edge')
    then Conv3d(padding=0). Verified parity vs torch replicate conv (~1e-5).
  * VQ nearest runs in float32 (float64 is unsupported on the Metal GPU). The torch model
    does the AUTHORITATIVE float64 search at inference; float32 assignment matches float64
    in practice (ties are rare), and this is only the training-time codebook assignment.

Codebook + EMA stats are `_`-prefixed buffers so MLX does not treat them as trainable
parameters; they are updated manually in `ema_update`. Attribute layout mirrors the torch
model (encoder.b{1,2,3}.conv/.norm, decoder.b{1,2,3}.conv/.norm) so convert.py walks it 1:1.
"""

from __future__ import annotations

from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from ..config import DEFAULT_CONFIG, TokenizerConfig


def _silu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(x)


def _edge_pad(x: mx.array, p: int) -> mx.array:
    """Replicate (edge) padding on the 3 spatial axes of an NDHWC tensor."""
    if p <= 0:
        return x
    return mx.pad(x, [(0, 0), (p, p), (p, p), (p, p), (0, 0)], mode="edge")


class _ConvBlock(nn.Module):
    """Conv3d (down) with replicate padding + GroupNorm + SiLU."""

    def __init__(self, cin, cout, k, s, p, groups, act=True):
        super().__init__()
        self.p = p
        self.conv = nn.Conv3d(cin, cout, kernel_size=k, stride=s, padding=0)  # manual replicate pad
        self.act = act
        if act:
            self.norm = nn.GroupNorm(groups, cout, pytorch_compatible=True)

    def __call__(self, x):
        x = self.conv(_edge_pad(x, self.p))
        if self.act:
            x = _silu(self.norm(x))
        return x


class _ResizeConvBlock(nn.Module):
    """Trilinear resize to a fixed grid size, then replicate-padded Conv3d (k3,s1) + norm/act."""

    def __init__(self, cin, cout, out_size, in_size, groups, act=True):
        super().__init__()
        self.out_size = out_size
        self.up = nn.Upsample(scale_factor=(out_size / in_size,) * 3, mode="linear", align_corners=True)
        self.conv = nn.Conv3d(cin, cout, kernel_size=3, stride=1, padding=0)  # manual replicate pad p=1
        self.act = act
        if act:
            self.norm = nn.GroupNorm(groups, cout, pytorch_compatible=True)

    def __call__(self, x):
        x = self.up(x)
        x = self.conv(_edge_pad(x, 1))
        if self.act:
            x = _silu(self.norm(x))
        return x


class Encoder(nn.Module):
    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        c1, c2 = cfg.enc_channels
        g = cfg.norm_groups
        self.b1 = _ConvBlock(3, c1, 3, 2, 1, g)                       # 17 -> 9
        self.b2 = _ConvBlock(c1, c2, 3, 2, 1, g)                      # 9 -> 5
        self.b3 = _ConvBlock(c2, cfg.code_dim, 2, 1, 0, g, act=False)  # 5 -> 4 (linear)

    def __call__(self, x):
        return self.b3(self.b2(self.b1(x)))


class Decoder(nn.Module):
    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        d1, d2 = cfg.dec_channels
        g = cfg.norm_groups
        self.b1 = _ResizeConvBlock(cfg.code_dim, d1, out_size=5, in_size=4, groups=g)        # 4 -> 5
        self.b2 = _ResizeConvBlock(d1, d2, out_size=9, in_size=5, groups=g)                  # 5 -> 9
        self.b3 = _ResizeConvBlock(d2, 3, out_size=cfg.grid, in_size=9, groups=g, act=False)  # 9 -> 17 (linear)

    def __call__(self, z):
        return self.b3(self.b2(self.b1(z)))


class VectorQuantizerEMA(nn.Module):
    """EMA VQ (v2). Codebook + EMA stats are ``_``-buffers (not trainable params)."""

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.K = cfg.codebook_size
        self.D = cfg.code_dim
        self.decay = cfg.ema_decay
        self.eps = cfg.ema_eps
        self.beta = cfg.commit_beta
        self.dead_threshold = cfg.dead_code_threshold
        self.revival = cfg.dead_code_revival
        self._codebook = mx.random.normal((self.K, self.D)) * 0.1
        self._cluster_size = mx.zeros((self.K,))
        self._embed_avg = mx.array(self._codebook)

    def _nearest(self, flat):
        """argmin_e ||e||^2 - 2 x.e (the ||x||^2 term is constant in the argmin), lowest-index
        tie-break. float32 here (Metal has no float64); torch does the float64 authoritative
        search at inference."""
        dist = (self._codebook ** 2).sum(axis=1) - 2.0 * (flat @ self._codebook.T)  # [M,K]
        return mx.argmin(dist, axis=1)

    def __call__(self, z_e):
        N = z_e.shape[0]
        L = z_e.shape[1]
        flat = z_e.reshape(-1, self.D)                        # [M,D], C-order (z fastest)
        idx = self._nearest(flat)
        quant = mx.take(self._codebook, idx, axis=0)
        commit = self.beta * mx.mean((flat - mx.stop_gradient(quant)) ** 2)
        quant_st = flat + mx.stop_gradient(quant - flat)
        onehot = (idx[:, None] == mx.arange(self.K)).astype(flat.dtype)
        avg = onehot.mean(axis=0)
        perplexity = mx.exp(-(avg * mx.log(avg + 1e-10)).sum())
        codes = idx.reshape(N, -1)
        quant_grid = quant_st.reshape(N, L, L, L, self.D)
        return quant_grid, codes, commit, perplexity

    def ema_update(self, z_e):
        flat = z_e.reshape(-1, self.D)
        idx = self._nearest(flat)
        onehot = (idx[:, None] == mx.arange(self.K)).astype(flat.dtype)
        counts = onehot.sum(axis=0)
        self._cluster_size = self.decay * self._cluster_size + (1.0 - self.decay) * counts
        embed_sum = onehot.T @ flat
        self._embed_avg = self.decay * self._embed_avg + (1.0 - self.decay) * embed_sum
        n = self._cluster_size.sum()
        smoothed = (self._cluster_size + self.eps) / (n + self.K * self.eps) * n
        self._codebook = self._embed_avg / smoothed[:, None]
        if self.revival:
            self._revive_dead(flat)

    def _revive_dead(self, flat):
        dead = self._cluster_size < self.dead_threshold
        pick = mx.random.randint(0, flat.shape[0], (self.K,))
        seeds = mx.take(flat, pick, axis=0)
        dcol = dead[:, None]
        self._codebook = mx.where(dcol, seeds, self._codebook)
        self._embed_avg = mx.where(dcol, seeds, self._embed_avg)
        self._cluster_size = mx.where(dead, mx.ones_like(self._cluster_size), self._cluster_size)

    def quantize_indices(self, z_e):
        N = z_e.shape[0]
        return self._nearest(z_e.reshape(-1, self.D)).reshape(N, -1)

    def embed_codes(self, codes, latent_grid):
        N = codes.shape[0]
        q = mx.take(self._codebook, codes.reshape(-1), axis=0)
        L = latent_grid
        return q.reshape(N, L, L, L, self.D)


class VQVAEmlx(nn.Module):
    """MLX VQ-VAE v2. Data path is channels-last: residual [N,r,g,b,3]."""

    def __init__(self, cfg: TokenizerConfig = DEFAULT_CONFIG):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.vq = VectorQuantizerEMA(cfg)
        self.decoder = Decoder(cfg)

    def __call__(self, residual_input):
        z_e = self.encoder(residual_input)
        quant, codes, commit, perplexity = self.vq(z_e)
        recon = self.decoder(quant)
        return {"recon": recon, "codes": codes, "commit_loss": commit,
                "perplexity": perplexity, "z_e": z_e}

    def encode(self, residual: np.ndarray) -> list[int]:
        x = mx.array(np.asarray(residual, dtype=np.float32)[None])
        codes = self.vq.quantize_indices(self.encoder(x))
        mx.eval(codes)
        return [int(v) for v in np.array(codes)[0]]

    def decode(self, codes: Sequence[int]) -> np.ndarray:
        arr = np.asarray(list(codes), dtype=np.int32)
        if arr.shape != (self.cfg.token_count,):
            raise ValueError(f"expected {self.cfg.token_count} codes, got {arr.shape}")
        if arr.min() < 0 or arr.max() >= self.cfg.codebook_size:
            raise ValueError("code ids must be in [0, codebook_size)")
        q = self.vq.embed_codes(mx.array(arr)[None], self.cfg.latent_grid)
        recon = self.decoder(q)
        mx.eval(recon)
        return np.array(recon)[0].astype(np.float64)
