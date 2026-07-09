"""Single-stage VQ-VAE for canonical residual LUTs (model_architecture.md "LUT Tokenizer").

    17x17x17x3 residual  --enc-->  4x4x4 latent  --VQ-->  64 codes (0..255)
                                                --dec-->  17x17x17x3 residual

Exact geometry (verified) — Encoder Conv3d 17->9->5->4, Decoder resize(trilinear)+Conv3d
4->5->9->17 — is pinned in :mod:`tokenizer.config`. Convs use `replicate` boundary
padding (avoids the zero-padding bias at corner/edge LUT nodes) and the decoder upsamples
by trilinear resize then Conv3d (no stride-2 ConvTranspose, so no checkerboard artifact).
The VQ uses EMA codebook updates with dead-code revival and a float64 nearest-code search.
:meth:`VQVAE.encode` / :meth:`VQVAE.decode` provide the
numpy-in/numpy-out contract that the frozen encoder (data_pipeline/tokenize_targets.py)
and decoder (eval/lut_decoder.py) will call once Stage 8 wires them in.

Axis / flatten orders are the pinned constants in config: residual arrays are
``[r,g,b,c]`` and map to conv tensors ``[N,C,X=r,Y=g,Z=b]``; the 4x4x4 latent flattens
to 64 tokens as ``token = x*16 + y*4 + z`` (z fastest).

Importing this module imports torch but performs no compute and touches no files.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DEFAULT_CONFIG, TokenizerConfig


# --- axis helpers (residual [r,g,b,c] <-> conv [N,C,X,Y,Z]) -------------------------
def residual_to_input(residual: np.ndarray | torch.Tensor) -> torch.Tensor:
    """``[r,g,b,3]`` (or batched ``[N,r,g,b,3]``) -> conv tensor ``[N,3,X=r,Y=g,Z=b]``."""
    t = torch.as_tensor(np.asarray(residual)) if not torch.is_tensor(residual) else residual
    t = t.to(torch.float32)
    if t.dim() == 4:
        t = t.unsqueeze(0)                      # -> [1,r,g,b,c]
    if t.dim() != 5 or t.shape[-1] != 3:
        raise ValueError(f"expected [...,r,g,b,3], got {tuple(t.shape)}")
    return t.permute(0, 4, 1, 2, 3).contiguous()  # [N,C,X=r,Y=g,Z=b]


def output_to_residual(x: torch.Tensor) -> torch.Tensor:
    """conv tensor ``[N,3,X,Y,Z]`` -> residual ``[N,r,g,b,3]``."""
    return x.permute(0, 2, 3, 4, 1).contiguous()


class _ConvBlock(nn.Module):
    """Conv3d (down) + GroupNorm + SiLU. `replicate` padding avoids zero-boundary bias."""

    def __init__(self, cin: int, cout: int, k: int, s: int, p: int, groups: int,
                 act: bool = True, padding_mode: str = "replicate"):
        super().__init__()
        # padding_mode is a no-op when p == 0 (e.g. the linear 5->4 bottleneck conv).
        self.conv = nn.Conv3d(cin, cout, kernel_size=k, stride=s, padding=p, padding_mode=padding_mode)
        self.norm = nn.GroupNorm(groups, cout) if act else None
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.act:
            x = F.silu(self.norm(x))
        return x


class _ResizeConvBlock(nn.Module):
    """Trilinear resize to a fixed grid size, then Conv3d (k3,s1,p1, replicate) + norm/act.

    Replaces stride-2 ConvTranspose3d: fixed-kernel upsampling has no learned kernel to
    produce the uneven-overlap (Odena) checkerboard pattern that inflates the ΔE tail on
    the smoothness-gated LUT grid, and `replicate` padding avoids biasing the
    corner/edge nodes the max/p99 gates are strictest on.
    """

    def __init__(self, cin: int, cout: int, out_size: int, groups: int, act: bool = True):
        super().__init__()
        self.out_size = out_size
        self.conv = nn.Conv3d(cin, cout, kernel_size=3, stride=1, padding=1, padding_mode="replicate")
        self.norm = nn.GroupNorm(groups, cout) if act else None
        self.act = act

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.out_size, self.out_size, self.out_size),
                          mode="trilinear", align_corners=True)
        x = self.conv(x)
        if self.act:
            x = F.silu(self.norm(x))
        return x


class Encoder(nn.Module):
    """17 -> 9 -> 5 -> 4 (Conv3d). Output is the pre-quantization latent [N, code_dim, 4,4,4]."""

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        c1, c2 = cfg.enc_channels
        g = cfg.norm_groups
        self.b1 = _ConvBlock(3, c1, k=3, s=2, p=1, groups=g)       # 17 -> 9
        self.b2 = _ConvBlock(c1, c2, k=3, s=2, p=1, groups=g)      # 9 -> 5
        self.b3 = _ConvBlock(c2, cfg.code_dim, k=2, s=1, p=0, groups=g, act=False)  # 5 -> 4 (linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b3(self.b2(self.b1(x)))


class Decoder(nn.Module):
    """4 -> 5 -> 9 -> 17 via resize(trilinear)+Conv3d. Output residual [N, 3, 17,17,17] (linear).

    The intermediate sizes 5 and 9 mirror the encoder's 17->9->5->4 downsampling path.
    """

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        d1, d2 = cfg.dec_channels
        g = cfg.norm_groups
        self.b1 = _ResizeConvBlock(cfg.code_dim, d1, out_size=5, groups=g)            # 4 -> 5
        self.b2 = _ResizeConvBlock(d1, d2, out_size=9, groups=g)                      # 5 -> 9
        self.b3 = _ResizeConvBlock(d2, 3, out_size=cfg.grid, groups=g, act=False)     # 9 -> 17 (linear)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.b3(self.b2(self.b1(z)))


class VectorQuantizerEMA(nn.Module):
    """EMA vector quantizer with dead-code revival (van den Oord et al.; Sonnet-style EMA)."""

    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.K = cfg.codebook_size
        self.D = cfg.code_dim
        self.decay = cfg.ema_decay
        self.eps = cfg.ema_eps
        self.beta = cfg.commit_beta
        self.dead_threshold = cfg.dead_code_threshold
        self.revival = cfg.dead_code_revival

        codebook = torch.randn(self.K, self.D) * 0.1
        # buffers (not parameters): updated by EMA, saved in state_dict.
        self.register_buffer("codebook", codebook)
        self.register_buffer("cluster_size", torch.zeros(self.K))
        self.register_buffer("embed_avg", codebook.clone())

    def _flatten(self, z_e: torch.Tensor) -> torch.Tensor:
        # [N, D, X, Y, Z] -> [N, X, Y, Z, D] -> [M, D]; C-order over (X,Y,Z) => z fastest.
        return z_e.permute(0, 2, 3, 4, 1).reshape(-1, self.D)

    def _nearest(self, flat: torch.Tensor) -> torch.Tensor:
        """Nearest codebook index for each row of ``flat`` [M, D] -> [M].

        Computed in float64 for cross-hardware-stable assignment (the frozen tokenizer's
        byte-identical-token scope). The constant ``||x||^2`` term is dropped — it does
        not change the argmin — which also removes the catastrophic-cancellation-prone
        subtraction in the float32 expansion. ``torch.argmin`` breaks ties to the lowest
        index, pinning the token id when two codes are equidistant.
        """
        fd = flat.to(torch.float64)
        cb = self.codebook.to(torch.float64)
        # argmin_e ||x - e||^2  ==  argmin_e (||e||^2 - 2 x·e)
        dist = cb.pow(2).sum(1) - 2.0 * (fd @ cb.t())              # [M, K]
        return dist.argmin(1)

    def forward(self, z_e: torch.Tensor):
        N = z_e.shape[0]
        flat = self._flatten(z_e)                                  # [M, D], M = N*64

        idx = self._nearest(flat)                                  # [M]
        onehot = F.one_hot(idx, self.K).type_as(flat)             # [M, K]
        quant = self.codebook[idx]                                 # [M, D]

        if self.training:
            with torch.no_grad():
                self._ema_update(flat, onehot)

        # commitment loss (encoder is pulled toward its chosen code)
        commit = self.beta * F.mse_loss(flat, quant.detach())
        # straight-through estimator
        quant_st = flat + (quant - flat).detach()

        # reshape back to [N, D, X, Y, Z]
        L = z_e.shape[2]
        quant_grid = quant_st.view(N, L, L, L, self.D).permute(0, 4, 1, 2, 3).contiguous()

        # diagnostics
        avg_probs = onehot.mean(0)
        perplexity = torch.exp(-(avg_probs * torch.log(avg_probs + 1e-10)).sum())
        codes = idx.view(N, -1)                                    # [N, 64] in flatten order

        return quant_grid, codes, commit, perplexity

    def _ema_update(self, flat: torch.Tensor, onehot: torch.Tensor) -> None:
        counts = onehot.sum(0)                                     # [K]
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        embed_sum = onehot.t() @ flat                              # [K, D]
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (n + self.K * self.eps) * n
        self.codebook.copy_(self.embed_avg / smoothed.unsqueeze(1))

        if self.revival:
            self._revive_dead(flat)

    def _revive_dead(self, flat: torch.Tensor) -> None:
        dead = self.cluster_size < self.dead_threshold
        n_dead = int(dead.sum())
        if n_dead == 0 or flat.shape[0] == 0:
            return
        # reseed dead codes with random current-batch encodings
        pick = torch.randint(0, flat.shape[0], (n_dead,), device=flat.device)
        seeds = flat[pick]
        self.codebook[dead] = seeds
        self.embed_avg[dead] = seeds
        self.cluster_size[dead] = 1.0

    # -- inference helpers (no EMA / no grad) --
    def quantize_indices(self, z_e: torch.Tensor) -> torch.Tensor:
        flat = self._flatten(z_e)
        return self._nearest(flat).view(z_e.shape[0], -1)          # [N, 64]

    def embed_codes(self, codes: torch.Tensor, latent_grid: int) -> torch.Tensor:
        """[N, 64] code ids -> quantized latent [N, D, X, Y, Z] (inverse flatten order)."""
        N = codes.shape[0]
        q = self.codebook[codes.reshape(-1)]                       # [N*64, D]
        L = latent_grid
        return q.view(N, L, L, L, self.D).permute(0, 4, 1, 2, 3).contiguous()


class VQVAE(nn.Module):
    """Full residual-LUT tokenizer: encoder + EMA VQ + decoder."""

    def __init__(self, cfg: TokenizerConfig = DEFAULT_CONFIG):
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.vq = VectorQuantizerEMA(cfg)
        self.decoder = Decoder(cfg)

    # -- training forward: residual grid in, reconstruction + aux out --
    def forward(self, residual_input: torch.Tensor):
        """``residual_input`` is a conv tensor ``[N,3,17,17,17]`` (see residual_to_input)."""
        z_e = self.encoder(residual_input)
        quant, codes, commit, perplexity = self.vq(z_e)
        recon = self.decoder(quant)
        return {
            "recon": recon,               # [N,3,17,17,17]
            "codes": codes,               # [N,64]
            "commit_loss": commit,        # scalar
            "perplexity": perplexity,     # scalar
            "z_e": z_e,
        }

    # -- inference contract (numpy in / numpy out), matches the Stage-8 stubs --
    @torch.no_grad()
    def encode(self, residual: np.ndarray) -> list[int]:
        """Canonical 17^3 residual ``[r,g,b,3]`` -> 64 codebook ids (0..255)."""
        self.eval()
        x = residual_to_input(residual).to(self._device())
        z_e = self.encoder(x)
        codes = self.vq.quantize_indices(z_e)                      # [1,64]
        return codes[0].detach().cpu().tolist()

    @torch.no_grad()
    def decode(self, codes: Sequence[int]) -> np.ndarray:
        """64 codebook ids (0..255) -> canonical 17^3 residual ``[r,g,b,3]`` (float64).

        The decoder runs in float32 (matching training/gate exactly) and only widens to
        float64 at the numpy boundary. Byte-identical ``.cube`` export is therefore scoped
        to one hardware class (the manifest records ``decode_dtype``); the pinned 10-decimal
        ``.cube`` format is intentionally left unchanged (a canonical-contract value).
        """
        self.eval()
        arr = np.asarray(list(codes), dtype=np.int64)
        if arr.shape != (self.cfg.token_count,):
            raise ValueError(f"expected {self.cfg.token_count} codes, got {arr.shape}")
        if arr.min() < 0 or arr.max() >= self.cfg.codebook_size:
            raise ValueError("code ids must be in [0, codebook_size)")
        t = torch.as_tensor(arr, device=self._device()).view(1, -1)
        quant = self.vq.embed_codes(t, self.cfg.latent_grid)
        recon = self.decoder(quant)                                # [1,3,17,17,17]
        return output_to_residual(recon)[0].detach().cpu().numpy().astype(np.float64)

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
