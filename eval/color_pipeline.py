"""ICC-aware color pipeline: canonical sRGB, CIE Lab (D65), and CIEDE2000.

Enabled implementation (cross-doc audit B2-B6). Responsibilities
(detailed_behavior_spec.md / model_architecture.md "Color Pipeline"):
  * conversion of an input image to canonical display-referred sRGB [0,1] under
    ``icc_conversion_config = srgb_relcol_bpc_float32_v1`` (relative-colorimetric intent,
    black-point compensation, deterministic gamut clip to [0,1], float32);
  * sRGB<->linear (IEC 61966-2-1 EOTF) and encoded-sRGB -> CIE Lab (D65);
  * CIEDE2000 color difference for reconstruction / target / representability reporting;
  * chroma / hue and highlight / shadow luminance masks used by the behavior vector.

Pure NumPy, deterministic, no external color deps. Embedded-ICC-profile transforms (real
LittleCMS CMM) are the one gated path: :func:`to_canonical_srgb` handles already-sRGB float
arrays now and raises for opaque ICC-profile bytes until a CMM backend is wired.

Consumed by :mod:`data_pipeline` (behavior vector, representability, quality) and, once the
decoder is enabled, by the eval color layers L4-L7. The identity grid + residual/absolute
conversion live in :mod:`eval.cube_io`.
"""

from __future__ import annotations

import numpy as np

ENABLED = True

COLOR_PIPELINE_VERSION = "color_v1_srgb_lab_d65_ciede2000"

# High/low L* region thresholds (audit B3): "highlights" / "shadows" masks.
HIGHLIGHT_L_MIN = 66.0
SHADOW_L_MAX = 33.0

# D65 reference white (2 deg observer), matching IEC 61966-2-1 sRGB.
_D65_WHITE = np.array([0.95047, 1.00000, 1.08883], dtype=np.float64)

# Linear sRGB -> XYZ (D65), IEC 61966-2-1.
_SRGB_TO_XYZ = np.array(
    [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ],
    dtype=np.float64,
)

_XYZ_TO_SRGB = np.linalg.inv(_SRGB_TO_XYZ)

# AdobeRGB (1998), D65 -- for color-managing AdobeRGB-authored LUTs (e.g. the ON1 pack) into the
# canonical sRGB domain. Symmetric power-law gamma (no linear toe) and the published D65 primaries.
_ADOBE_RGB_GAMMA = 563.0 / 256.0  # 2.19921875
_ADOBE_RGB_TO_XYZ = np.array(
    [
        [0.5767309, 0.1855540, 0.1881852],
        [0.2973769, 0.6273491, 0.0752741],
        [0.0270343, 0.0706872, 0.9911085],
    ],
    dtype=np.float64,
)
_XYZ_TO_ADOBE_RGB = np.linalg.inv(_ADOBE_RGB_TO_XYZ)

_EPS = (6.0 / 29.0) ** 3  # ~0.008856
_KAPPA_LIN = 3.0 * (6.0 / 29.0) ** 2  # linear-segment slope factor denominator helper
_DELTA = 6.0 / 29.0


class ColorLayerDisabled(RuntimeError):
    """Raised only for the still-gated embedded-ICC-profile transform path."""


def is_enabled() -> bool:
    return ENABLED


# --- transfer function ------------------------------------------------------------
def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Encoded sRGB [0,1] -> linear-light [0,1] (IEC 61966-2-1 EOTF)."""
    x = np.asarray(rgb, dtype=np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(lin: np.ndarray) -> np.ndarray:
    """Linear-light [0,1] -> encoded sRGB [0,1] (inverse EOTF)."""
    x = np.asarray(lin, dtype=np.float64)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(np.clip(x, 0.0, None), 1.0 / 2.4) - 0.055)


# --- AdobeRGB (1998) <-> sRGB color management ------------------------------------
def adobe_rgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    """Encoded AdobeRGB [0,1] -> linear-light AdobeRGB (symmetric power law)."""
    return np.power(np.clip(np.asarray(rgb, dtype=np.float64), 0.0, None), _ADOBE_RGB_GAMMA)


def linear_to_adobe_rgb(lin: np.ndarray) -> np.ndarray:
    """Linear-light AdobeRGB -> encoded AdobeRGB [0,1]."""
    return np.power(np.clip(np.asarray(lin, dtype=np.float64), 0.0, None), 1.0 / _ADOBE_RGB_GAMMA)


def adobe_rgb_to_srgb(rgb: np.ndarray) -> np.ndarray:
    """Encoded AdobeRGB [0,1] -> encoded sRGB [0,1] (D65->D65, clipped to sRGB gamut).

    AdobeRGB's wider gamut means some colours fall outside sRGB; those are gamut-clipped in
    linear light before re-encoding, matching our deterministic canonical clip.
    """
    lin_adobe = adobe_rgb_to_linear(rgb)
    xyz = lin_adobe @ _ADOBE_RGB_TO_XYZ.T
    lin_srgb = np.clip(xyz @ _XYZ_TO_SRGB.T, 0.0, 1.0)
    return linear_to_srgb(lin_srgb)


def srgb_to_adobe_rgb(rgb: np.ndarray) -> np.ndarray:
    """Encoded sRGB [0,1] -> encoded AdobeRGB [0,1] (D65->D65). sRGB is inside AdobeRGB's gamut."""
    lin_srgb = srgb_to_linear(rgb)
    xyz = lin_srgb @ _SRGB_TO_XYZ.T
    lin_adobe = np.clip(xyz @ _XYZ_TO_ADOBE_RGB.T, 0.0, 1.0)
    return linear_to_adobe_rgb(lin_adobe)


# --- sRGB -> Lab (D65) ------------------------------------------------------------
def srgb_to_xyz(rgb: np.ndarray) -> np.ndarray:
    """Encoded sRGB [0,1] -> CIE XYZ (D65), Y in [0,1]."""
    lin = srgb_to_linear(rgb)
    return lin @ _SRGB_TO_XYZ.T


def _lab_f(t: np.ndarray) -> np.ndarray:
    return np.where(t > _EPS, np.cbrt(t), t / _KAPPA_LIN + 4.0 / 29.0)


def xyz_to_lab_d65(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    scaled = xyz / _D65_WHITE
    f = _lab_f(scaled)
    fx, fy, fz = f[..., 0], f[..., 1], f[..., 2]
    lab = np.empty_like(xyz)
    lab[..., 0] = 116.0 * fy - 16.0
    lab[..., 1] = 500.0 * (fx - fy)
    lab[..., 2] = 200.0 * (fy - fz)
    return lab


def srgb_to_lab_d65(rgb: np.ndarray) -> np.ndarray:
    """Encoded sRGB [0,1] -> CIE Lab (D65). Input shape ``[...,3]`` -> same-shape Lab."""
    return xyz_to_lab_d65(srgb_to_xyz(rgb))


# --- Lab (D65) -> sRGB (inverse) --------------------------------------------------
def _lab_finv(t: np.ndarray) -> np.ndarray:
    return np.where(t > _DELTA, t ** 3, 3.0 * (_DELTA ** 2) * (t - 4.0 / 29.0))


def lab_d65_to_xyz(lab: np.ndarray) -> np.ndarray:
    lab = np.asarray(lab, dtype=np.float64)
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = fy + lab[..., 1] / 500.0
    fz = fy - lab[..., 2] / 200.0
    xyz = np.empty_like(lab)
    xyz[..., 0] = _D65_WHITE[0] * _lab_finv(fx)
    xyz[..., 1] = _D65_WHITE[1] * _lab_finv(fy)
    xyz[..., 2] = _D65_WHITE[2] * _lab_finv(fz)
    return xyz


def xyz_to_srgb(xyz: np.ndarray, clip: bool = True) -> np.ndarray:
    """CIE XYZ (D65) -> encoded sRGB. Clips to [0,1] by default (deterministic gamut clip)."""
    lin = np.asarray(xyz, dtype=np.float64) @ _XYZ_TO_SRGB.T
    lin = np.clip(lin, 0.0, None)
    srgb = linear_to_srgb(lin)
    return np.clip(srgb, 0.0, 1.0) if clip else srgb


def lab_d65_to_srgb(lab: np.ndarray, clip: bool = True) -> np.ndarray:
    """CIE Lab (D65) -> encoded sRGB [0,1]."""
    return xyz_to_srgb(lab_d65_to_xyz(lab), clip=clip)


# --- chroma / hue -----------------------------------------------------------------
def chroma(lab: np.ndarray) -> np.ndarray:
    """C* = sqrt(a*^2 + b*^2)."""
    lab = np.asarray(lab, dtype=np.float64)
    return np.sqrt(lab[..., 1] ** 2 + lab[..., 2] ** 2)


def hue_deg(lab: np.ndarray) -> np.ndarray:
    """Hue angle atan2(b*, a*) in degrees, [0,360)."""
    lab = np.asarray(lab, dtype=np.float64)
    h = np.degrees(np.arctan2(lab[..., 2], lab[..., 1]))
    return np.mod(h, 360.0)


def highlight_mask(lab: np.ndarray, l_min: float = HIGHLIGHT_L_MIN) -> np.ndarray:
    return np.asarray(lab, dtype=np.float64)[..., 0] >= l_min


def shadow_mask(lab: np.ndarray, l_max: float = SHADOW_L_MAX) -> np.ndarray:
    return np.asarray(lab, dtype=np.float64)[..., 0] <= l_max


# --- CIEDE2000 --------------------------------------------------------------------
def ciede2000(lab_a: np.ndarray, lab_b: np.ndarray) -> np.ndarray:
    """CIEDE2000 color difference (Sharma et al. 2005), kL=kC=kH=1.

    Accepts Lab arrays of shape ``[...,3]``; returns delta-E of shape ``[...]``.
    """
    lab1 = np.asarray(lab_a, dtype=np.float64)
    lab2 = np.asarray(lab_b, dtype=np.float64)
    L1, a1, b1 = lab1[..., 0], lab1[..., 1], lab1[..., 2]
    L2, a2, b2 = lab2[..., 0], lab2[..., 1], lab2[..., 2]

    C1 = np.hypot(a1, b1)
    C2 = np.hypot(a2, b2)
    C_bar = 0.5 * (C1 + C2)
    C_bar7 = C_bar ** 7
    G = 0.5 * (1.0 - np.sqrt(C_bar7 / (C_bar7 + 25.0 ** 7)))

    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = np.hypot(a1p, b1)
    C2p = np.hypot(a2p, b2)

    h1p = np.mod(np.degrees(np.arctan2(b1, a1p)), 360.0)
    h2p = np.mod(np.degrees(np.arctan2(b2, a2p)), 360.0)
    # atan2(0,0) -> 0 already; where C'==0 the hue terms are neutralized below.

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    dhp = np.where(dhp > 180.0, dhp - 360.0, dhp)
    dhp = np.where(dhp < -180.0, dhp + 360.0, dhp)
    dhp = np.where(C1p * C2p == 0.0, 0.0, dhp)
    dHp = 2.0 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp) / 2.0)

    Lp_bar = 0.5 * (L1 + L2)
    Cp_bar = 0.5 * (C1p + C2p)

    h_sum = h1p + h2p
    h_diff = np.abs(h1p - h2p)
    hp_bar = np.where(
        C1p * C2p == 0.0,
        h_sum,
        np.where(
            h_diff <= 180.0,
            0.5 * h_sum,
            np.where(h_sum < 360.0, 0.5 * (h_sum + 360.0), 0.5 * (h_sum - 360.0)),
        ),
    )

    T = (
        1.0
        - 0.17 * np.cos(np.radians(hp_bar - 30.0))
        + 0.24 * np.cos(np.radians(2.0 * hp_bar))
        + 0.32 * np.cos(np.radians(3.0 * hp_bar + 6.0))
        - 0.20 * np.cos(np.radians(4.0 * hp_bar - 63.0))
    )

    dtheta = 30.0 * np.exp(-(((hp_bar - 275.0) / 25.0) ** 2))
    Cp_bar7 = Cp_bar ** 7
    Rc = 2.0 * np.sqrt(Cp_bar7 / (Cp_bar7 + 25.0 ** 7))
    SL = 1.0 + (0.015 * (Lp_bar - 50.0) ** 2) / np.sqrt(20.0 + (Lp_bar - 50.0) ** 2)
    SC = 1.0 + 0.045 * Cp_bar
    SH = 1.0 + 0.015 * Cp_bar * T
    RT = -np.sin(np.radians(2.0 * dtheta)) * Rc

    dE = np.sqrt(
        (dLp / SL) ** 2
        + (dCp / SC) ** 2
        + (dHp / SH) ** 2
        + RT * (dCp / SC) * (dHp / SH)
    )
    return dE


def deltae2000_srgb(rgb_a: np.ndarray, rgb_b: np.ndarray) -> np.ndarray:
    """Convenience: CIEDE2000 between two encoded-sRGB arrays."""
    return ciede2000(srgb_to_lab_d65(rgb_a), srgb_to_lab_d65(rgb_b))


# --- canonical sRGB ---------------------------------------------------------------
def to_canonical_srgb(image, icc_profile: bytes | None = None) -> np.ndarray:  # noqa: ANN001
    """Convert an image to canonical display-referred sRGB [0,1] float32.

    Handles already-sRGB inputs (uint8 or float arrays, or PIL images decoded as sRGB):
    normalizes dtype/range to float32 and deterministically clips to [0,1]. Embedded ICC
    profiles (opaque bytes) require a CMM and remain gated -> ``ColorLayerDisabled``.
    """
    if icc_profile is not None:
        raise ColorLayerDisabled(
            "embedded-ICC-profile transform needs a CMM backend; convert to sRGB upstream "
            "or wire LittleCMS (icc_conversion_config=srgb_relcol_bpc_float32_v1)"
        )
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        arr = arr.astype(np.float32) / float(info.max)
    else:
        arr = arr.astype(np.float32)
    return np.clip(arr, 0.0, 1.0)


def icc_conversion_config_id() -> str:
    from .schemas import ICC_CONVERSION_CONFIG

    return ICC_CONVERSION_CONFIG
