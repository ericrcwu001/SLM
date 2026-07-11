"""Stage 4: canonicalize a raw LUT tensor to the v1 canonical domain.

canonical_domain_id = slm_lut_v1_srgb_display_encoded_17_trilinear:
display-referred IEC 61966-2-1 sRGB, encoded [0,1], D65, 17^3, trilinear;
residual = canonical absolute LUT - encoded-sRGB identity grid.

Raw LUTs must be color-managed into this domain before hashing/residual (ADR 0003,
model_architecture.md). Our decoded sources (HaldCLUT PNG, ``.cube``, XMP-render, pair-fit)
are already encoded-sRGB display-referred, so color-management here is: assume/verify sRGB,
resample to 17^3, deterministically clip to [0,1]. Unknown/camera-log domains are recorded
as assumed-sRGB with a warning, or rejected when explicitly a log/unknown domain.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from eval import color_pipeline as cp
from eval.cube_io import GRID_SIZE, absolute_to_residual, cube_bytes_hash, identity_grid

from .lut_ops import apply_lut_trilinear, resample_lut

KNOWN_SRGB_DOMAINS = {"srgb", "srgb_display", "srgb_8bit", "srgb_encoded", "rec709_srgb", None}
# AdobeRGB-authored LUTs (e.g. the ON1 pack) are color-managed into canonical sRGB, not assumed-sRGB.
ADOBE_RGB_DOMAINS = {"adobe_rgb", "adobergb", "adobe_rgb_1998", "argb"}
REJECT_DOMAINS = {"camera_log", "log", "acescg", "linear_unknown", "unknown_camera_log"}


def _canonicalize_adobe_rgb(arr: np.ndarray) -> np.ndarray:
    """Color-manage an AdobeRGB-authored LUT into a canonical sRGB 17^3 absolute LUT.

    The LUT maps AdobeRGB->AdobeRGB; the canonical LUT must map sRGB->sRGB. For each canonical
    sRGB node s: sRGB->AdobeRGB in, apply the native LUT, AdobeRGB->sRGB out.
    """
    nodes = identity_grid(GRID_SIZE).reshape(-1, 3)
    in_adobe = np.clip(cp.srgb_to_adobe_rgb(nodes), 0.0, 1.0)
    out_adobe = apply_lut_trilinear(arr, in_adobe)
    out_srgb = cp.adobe_rgb_to_srgb(out_adobe)
    return out_srgb.reshape(GRID_SIZE, GRID_SIZE, GRID_SIZE, 3)


@dataclass
class CanonicalResult:
    absolute: np.ndarray | None
    residual: np.ndarray | None
    canonical_absolute_lut_hash: str | None
    canonical_residual_lut_hash: str | None
    normalization_warnings: list = field(default_factory=list)
    rejected: bool = False
    reject_reason: str | None = None
    # Authored LUT values BEFORE the [0,1] clip — the out-of-gamut/pre-clamp quality gate needs these,
    # since after clipping the gate would only ever see in-gamut data and could never fire.
    pre_clamp_absolute: np.ndarray | None = None


def _residual_hash(residual: np.ndarray) -> str:
    buf = np.ascontiguousarray(np.round(residual, 10).astype("<f8")).tobytes()
    return hashlib.sha256(buf).hexdigest()


def canonicalize_lut(
    lut_tensor: np.ndarray,
    declared_domain: str | None = "srgb",
    *,
    assume_srgb_if_unknown: bool = True,
) -> CanonicalResult:
    """Canonicalize an absolute LUT tensor ``[M,M,M,3]`` to the v1 canonical domain."""
    warnings: list[str] = []
    dom = (declared_domain or "").lower() if declared_domain else None

    if dom in REJECT_DOMAINS:
        return CanonicalResult(None, None, None, None, [f"reject_domain:{dom}"],
                               rejected=True, reject_reason=f"camera_log_unknown_domain:{dom}")

    is_adobe = dom in ADOBE_RGB_DOMAINS
    if not is_adobe and dom not in KNOWN_SRGB_DOMAINS:
        if not assume_srgb_if_unknown:
            return CanonicalResult(None, None, None, None, [f"unknown_domain:{dom}"],
                                   rejected=True, reject_reason=f"unknown_domain:{dom}")
        warnings.append(f"assumed_srgb:{dom}")

    arr = np.asarray(lut_tensor, dtype=np.float64)
    pre_clamp = arr.copy()   # capture authored values before clipping (for the out-of-gamut gate)
    if arr.min() < 0.0 or arr.max() > 1.0:
        warnings.append("clipped_out_of_range")
    arr = np.clip(arr, 0.0, 1.0)

    if is_adobe:
        # color-manage AdobeRGB->sRGB (this also lands the LUT on the canonical 17^3 grid)
        absolute = _canonicalize_adobe_rgb(arr)
        warnings.append("color_managed_adobe_rgb_to_srgb")
    else:
        absolute = resample_lut(arr, GRID_SIZE)
        if absolute.shape[0] != GRID_SIZE and arr.shape[0] != GRID_SIZE:
            warnings.append(f"resampled_{arr.shape[0]}_to_{GRID_SIZE}")
    residual = absolute_to_residual(absolute)

    return CanonicalResult(
        absolute=absolute,
        residual=residual,
        canonical_absolute_lut_hash=cube_bytes_hash(absolute),
        canonical_residual_lut_hash=_residual_hash(residual),
        normalization_warnings=warnings,
        pre_clamp_absolute=pre_clamp,
    )


def is_identity(residual: np.ndarray, atol: float = 1e-9) -> bool:
    return bool(np.max(np.abs(residual)) <= atol)
