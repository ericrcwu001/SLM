"""Differentiable (torch) sRGB->Lab(D65) + CIEDE2000, for the L_deltaE training loss.

A faithful torch port of the authoritative NumPy implementation in
``eval/color_pipeline.py`` (Sharma et al. 2005 CIEDE2000, kL=kC=kH=1; IEC 61966-2-1
sRGB; D65). Constants are copied verbatim so the training loss and the eval/gate
metric agree — ``tests/test_tokenizer_color_torch.py`` asserts parity vs NumPy.

Only the sRGB->Lab->deltaE forward path needed for the loss is ported here; the
authoritative measurement path stays in :mod:`eval.color_pipeline` (NumPy).

Inputs are encoded-sRGB in [0,1]; callers should clamp the absolute LUT to [0,1]
before conversion (the deterministic gamut clip), mirroring ``to_canonical_srgb``.
"""

from __future__ import annotations

import math

import torch

# --- constants (verbatim from eval/color_pipeline.py) -------------------------------
_D65_WHITE = (0.95047, 1.00000, 1.08883)

_SRGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)

_EPS = (6.0 / 29.0) ** 3           # ~0.008856
_KAPPA_LIN = 3.0 * (6.0 / 29.0) ** 2
_DEG = 180.0 / math.pi
_RAD = math.pi / 180.0
_POW25_7 = 25.0 ** 7


def _const(x, ref: torch.Tensor) -> torch.Tensor:
    """Materialize a constant tensor on ref's device/dtype."""
    return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)


def _safe_sqrt(x: torch.Tensor) -> torch.Tensor:
    """sqrt with a finite gradient at 0. ``torch.sqrt``/``hypot`` have inf/NaN grads at
    zero, which — for neutral-gray LUT nodes (a*=b*=0) and via ``torch.where`` masking
    (inf*0 -> NaN) — poisons training. Clamp to >=0 and add a tiny eps."""
    return torch.sqrt(torch.clamp(x, min=0.0) + 1.0e-12)


def _safe_atan2(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """atan2 with a finite gradient at the origin. ``atan2(0,0)`` has a NaN gradient
    (0/0); for neutral colors (a*=b*=0) this poisons training. Substitute (y=0, x=1)
    exactly at the origin -> hue 0 (the correct neutral hue) with a finite gradient."""
    both_zero = (x == 0.0) & (y == 0.0)
    xs = torch.where(both_zero, torch.ones_like(x), x)
    ys = torch.where(both_zero, torch.zeros_like(y), y)
    return torch.atan2(ys, xs)


# --- transfer function --------------------------------------------------------------
def srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    """Encoded sRGB [0,1] -> linear-light (IEC 61966-2-1 EOTF). Gradient-safe."""
    # clamp the power-branch base to >=0 so negative inputs cannot produce NaN grads;
    # on valid [0,1] inputs this is identical to the NumPy reference.
    hi_base = torch.clamp((rgb + 0.055) / 1.055, min=0.0)
    hi = hi_base ** 2.4
    lo = rgb / 12.92
    return torch.where(rgb <= 0.04045, lo, hi)


def srgb_to_xyz(rgb: torch.Tensor) -> torch.Tensor:
    """Encoded sRGB [0,1] -> CIE XYZ (D65). Input ``[...,3]`` -> ``[...,3]``."""
    lin = srgb_to_linear(rgb)
    mat = _const(_SRGB_TO_XYZ, rgb)          # (3,3)
    return lin @ mat.T


def _lab_f(t: torch.Tensor) -> torch.Tensor:
    # cube-root branch: clamp base to a small positive so the MASKED branch's local grad
    # ((1/3)x^-2/3) stays finite (x=0 -> inf grad -> inf*0 -> NaN under torch.where).
    # The clamp floor (1e-10) is far below _EPS, so the selected (t>_EPS) region is exact.
    cbrt = torch.clamp(t, min=1.0e-10) ** (1.0 / 3.0)
    lin = t / _KAPPA_LIN + 4.0 / 29.0
    return torch.where(t > _EPS, cbrt, lin)


def xyz_to_lab_d65(xyz: torch.Tensor) -> torch.Tensor:
    white = _const(_D65_WHITE, xyz)
    f = _lab_f(xyz / white)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return torch.stack((L, a, b), dim=-1)


def srgb_to_lab_d65(rgb: torch.Tensor) -> torch.Tensor:
    """Encoded sRGB [0,1] -> CIE Lab (D65). Shape ``[...,3]`` -> ``[...,3]``."""
    return xyz_to_lab_d65(srgb_to_xyz(rgb))


# --- CIEDE2000 ----------------------------------------------------------------------
def ciede2000(lab_a: torch.Tensor, lab_b: torch.Tensor) -> torch.Tensor:
    """CIEDE2000 (kL=kC=kH=1). Lab arrays ``[...,3]`` -> deltaE ``[...]``.

    Mirrors ``eval.color_pipeline.ciede2000`` term-for-term.
    """
    L1, a1, b1 = lab_a[..., 0], lab_a[..., 1], lab_a[..., 2]
    L2, a2, b2 = lab_b[..., 0], lab_b[..., 1], lab_b[..., 2]

    C1 = _safe_sqrt(a1 * a1 + b1 * b1)
    C2 = _safe_sqrt(a2 * a2 + b2 * b2)
    C_bar = 0.5 * (C1 + C2)
    C_bar7 = C_bar ** 7
    G = 0.5 * (1.0 - _safe_sqrt(C_bar7 / (C_bar7 + _POW25_7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = _safe_sqrt(a1p * a1p + b1 * b1)
    C2p = _safe_sqrt(a2p * a2p + b2 * b2)

    h1p = torch.remainder(_safe_atan2(b1, a1p) * _DEG, 360.0)
    h2p = torch.remainder(_safe_atan2(b2, a2p) * _DEG, 360.0)

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = torch.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = torch.where(dhp < -180.0, dhp + 360.0, dhp)
    zero_c = (C1p * C2p) == 0.0
    dhp = torch.where(zero_c, torch.zeros_like(dhp), dhp)
    dHp = 2.0 * _safe_sqrt(C1p * C2p) * torch.sin(dhp * _RAD / 2.0)

    Lp_bar = 0.5 * (L1 + L2)
    Cp_bar = 0.5 * (C1p + C2p)

    h_sum = h1p + h2p
    h_diff = torch.abs(h1p - h2p)
    hp_bar = torch.where(
        zero_c,
        h_sum,
        torch.where(
            h_diff <= 180.0,
            0.5 * h_sum,
            torch.where(h_sum < 360.0, 0.5 * (h_sum + 360.0), 0.5 * (h_sum - 360.0)),
        ),
    )

    T = (
        1.0
        - 0.17 * torch.cos((hp_bar - 30.0) * _RAD)
        + 0.24 * torch.cos((2.0 * hp_bar) * _RAD)
        + 0.32 * torch.cos((3.0 * hp_bar + 6.0) * _RAD)
        - 0.20 * torch.cos((4.0 * hp_bar - 63.0) * _RAD)
    )

    dtheta = 30.0 * torch.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    Cp_bar7 = Cp_bar ** 7
    Rc = 2.0 * _safe_sqrt(Cp_bar7 / (Cp_bar7 + _POW25_7))
    SL = 1.0 + (0.015 * (Lp_bar - 50.0) ** 2) / torch.sqrt(20.0 + (Lp_bar - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cp_bar
    SH = 1.0 + 0.015 * Cp_bar * T
    RT = -torch.sin((2.0 * dtheta) * _RAD) * Rc

    arg = (
        (dLp / SL) ** 2
        + (dCp / SC) ** 2
        + (dHp / SH) ** 2
        + RT * (dCp / SC) * (dHp / SH)
    )
    # The RT cross-term can push `arg` marginally below 0 numerically -> sqrt would be
    # NaN and poison training grads. _safe_sqrt clamps to >=0 (ΔE is real) and adds a
    # tiny eps so the gradient stays finite at exact matches (within parity tol).
    return _safe_sqrt(arg)


def deltae2000_srgb(rgb_a: torch.Tensor, rgb_b: torch.Tensor) -> torch.Tensor:
    """CIEDE2000 between two encoded-sRGB tensors ``[...,3]`` -> ``[...]``."""
    return ciede2000(srgb_to_lab_d65(rgb_a), srgb_to_lab_d65(rgb_b))
