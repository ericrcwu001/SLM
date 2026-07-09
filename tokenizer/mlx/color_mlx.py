"""Differentiable (MLX) sRGB->Lab(D65) + CIEDE2000 — the MLX mirror of tokenizer.color_torch.

Same math as the torch port (which itself mirrors the authoritative NumPy
eval.color_pipeline), including the gradient-safety tricks that keep neutral-gray LUT
nodes from producing NaN grads: safe sqrt (clamp>=0 + eps), safe atan2 (substitute the
origin), cube-root clamped to 1e-10.

Runs in MLX's default float32 on the Apple GPU, so this is a float32 TRAINING signal.
The authoritative gate ΔE stays NumPy float64 (tokenizer.metrics via eval.color_pipeline),
so parity here only needs to be close, not bit-exact.
"""

from __future__ import annotations

import math

import mlx.core as mx

# constants (verbatim from eval/color_pipeline.py)
_D65_WHITE = (0.95047, 1.00000, 1.08883)
_SRGB_TO_XYZ = (
    (0.4124564, 0.3575761, 0.1804375),
    (0.2126729, 0.7151522, 0.0721750),
    (0.0193339, 0.1191920, 0.9503041),
)
_EPS = (6.0 / 29.0) ** 3
_KAPPA_LIN = 3.0 * (6.0 / 29.0) ** 2
_DEG = 180.0 / math.pi
_RAD = math.pi / 180.0
_POW25_7 = 25.0 ** 7


def _safe_sqrt(x: mx.array) -> mx.array:
    return mx.sqrt(mx.maximum(x, 0.0) + 1.0e-12)


def _mod360(a: mx.array) -> mx.array:
    return a - 360.0 * mx.floor(a / 360.0)


def _safe_atan2(y: mx.array, x: mx.array) -> mx.array:
    both_zero = (x == 0.0) & (y == 0.0)
    xs = mx.where(both_zero, mx.ones_like(x), x)
    ys = mx.where(both_zero, mx.zeros_like(y), y)
    return mx.arctan2(ys, xs)


def srgb_to_linear(rgb: mx.array) -> mx.array:
    hi = mx.maximum((rgb + 0.055) / 1.055, 0.0) ** 2.4
    lo = rgb / 12.92
    return mx.where(rgb <= 0.04045, lo, hi)


def srgb_to_xyz(rgb: mx.array) -> mx.array:
    lin = srgb_to_linear(rgb)
    mat = mx.array(_SRGB_TO_XYZ, dtype=rgb.dtype)
    return lin @ mat.T


def _lab_f(t: mx.array) -> mx.array:
    cbrt = mx.maximum(t, 1.0e-10) ** (1.0 / 3.0)
    lin = t / _KAPPA_LIN + 4.0 / 29.0
    return mx.where(t > _EPS, cbrt, lin)


def xyz_to_lab_d65(xyz: mx.array) -> mx.array:
    white = mx.array(_D65_WHITE, dtype=xyz.dtype)
    f = _lab_f(xyz / white)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return mx.stack([L, a, b], axis=-1)


def srgb_to_lab_d65(rgb: mx.array) -> mx.array:
    return xyz_to_lab_d65(srgb_to_xyz(rgb))


def ciede2000(lab_a: mx.array, lab_b: mx.array) -> mx.array:
    """CIEDE2000 (kL=kC=kH=1). Lab ``[...,3]`` -> ΔE ``[...]``. Mirrors color_torch.ciede2000."""
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

    h1p = _mod360(_safe_atan2(b1, a1p) * _DEG)
    h2p = _mod360(_safe_atan2(b2, a2p) * _DEG)

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = mx.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = mx.where(dhp < -180.0, dhp + 360.0, dhp)
    zero_c = (C1p * C2p) == 0.0
    dhp = mx.where(zero_c, mx.zeros_like(dhp), dhp)
    dHp = 2.0 * _safe_sqrt(C1p * C2p) * mx.sin(dhp * _RAD / 2.0)

    Lp_bar = 0.5 * (L1 + L2)
    Cp_bar = 0.5 * (C1p + C2p)

    h_sum = h1p + h2p
    h_diff = mx.abs(h1p - h2p)
    hp_bar = mx.where(
        zero_c,
        h_sum,
        mx.where(
            h_diff <= 180.0,
            0.5 * h_sum,
            mx.where(h_sum < 360.0, 0.5 * (h_sum + 360.0), 0.5 * (h_sum - 360.0)),
        ),
    )

    T = (
        1.0
        - 0.17 * mx.cos((hp_bar - 30.0) * _RAD)
        + 0.24 * mx.cos((2.0 * hp_bar) * _RAD)
        + 0.32 * mx.cos((3.0 * hp_bar + 6.0) * _RAD)
        - 0.20 * mx.cos((4.0 * hp_bar - 63.0) * _RAD)
    )

    dtheta = 30.0 * mx.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    Cp_bar7 = Cp_bar ** 7
    Rc = 2.0 * _safe_sqrt(Cp_bar7 / (Cp_bar7 + _POW25_7))
    SL = 1.0 + (0.015 * (Lp_bar - 50.0) ** 2) / mx.sqrt(20.0 + (Lp_bar - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cp_bar
    SH = 1.0 + 0.015 * Cp_bar * T
    RT = -mx.sin((2.0 * dtheta) * _RAD) * Rc

    arg = (dLp / SL) ** 2 + (dCp / SC) ** 2 + (dHp / SH) ** 2 + RT * (dCp / SC) * (dHp / SH)
    return _safe_sqrt(arg)


def deltae2000_srgb(rgb_a: mx.array, rgb_b: mx.array) -> mx.array:
    return ciede2000(srgb_to_lab_d65(rgb_a), srgb_to_lab_d65(rgb_b))
