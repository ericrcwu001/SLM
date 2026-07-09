"""Core LUT tensor operations shared across derivation, canonicalization, and metrics.

All LUTs are canonical ``lut[r, g, b, channel]`` on an N-node grid, node ``i -> i/(N-1)``
(see :mod:`eval.cube_io`). Everything here is pure/deterministic NumPy + SciPy.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from eval.cube_io import GRID_SIZE, identity_grid


def _axes(n: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n, dtype=np.float64)


def apply_lut_trilinear(lut_abs: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Apply an absolute LUT ``[N,N,N,3]`` to encoded-sRGB ``rgb`` (``[...,3]`` in [0,1]).

    Trilinear interpolation on the canonical grid; inputs clipped to [0,1].
    """
    n = lut_abs.shape[0]
    axis = _axes(n)
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    flat = rgb.reshape(-1, 3)
    out = np.empty_like(flat)
    for ch in range(3):
        interp = RegularGridInterpolator(
            (axis, axis, axis), lut_abs[..., ch], method="linear", bounds_error=False, fill_value=None
        )
        out[:, ch] = interp(flat)
    return out.reshape(rgb.shape)


def resample_lut(lut_abs: np.ndarray, target_size: int = GRID_SIZE) -> np.ndarray:
    """Resample an absolute LUT to ``target_size`` nodes by sampling it at the new grid."""
    if lut_abs.shape[0] == target_size:
        return np.array(lut_abs, dtype=np.float64)
    target_inputs = identity_grid(target_size).reshape(-1, 3)
    sampled = apply_lut_trilinear(lut_abs, target_inputs)
    return sampled.reshape(target_size, target_size, target_size, 3)


def hald_level_and_edge(side: int) -> tuple[int, int]:
    """From a square HaldCLUT side length, return (level, cube_edge). side == level**3."""
    level = int(round(side ** (1.0 / 3.0)))
    if level ** 3 != side:
        raise ValueError(f"not a HaldCLUT side length: {side} (level {level} -> {level**3})")
    return level, level * level


def haldclut_to_lut(png: np.ndarray, target_size: int | None = GRID_SIZE) -> np.ndarray:
    """Convert a HaldCLUT image (H==W==level**3, cube edge = level**2) to a canonical LUT.

    Hald layout: flat pixel index ``i = b*E^2 + g*E + r`` (r fastest) holds the output color
    for input ``(r,g,b)/(E-1)``. Returns an absolute LUT resampled to ``target_size``, or the
    NATIVE-size LUT when ``target_size is None``.
    """
    arr = np.asarray(png)
    if arr.ndim != 3 or arr.shape[2] < 3 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"expected square RGB HaldCLUT image, got shape {arr.shape}")
    if arr.dtype == np.uint8:
        arr = arr.astype(np.float64) / 255.0
    elif np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float64) / float(np.iinfo(arr.dtype).max)
    else:
        arr = arr.astype(np.float64)
    side = arr.shape[0]
    _, edge = hald_level_and_edge(side)
    flat = arr[..., :3].reshape(edge * edge * edge, 3)
    # flat index i = b*E^2 + g*E + r  ->  reshape C-order gives [b, g, r, 3]
    cube_bgr = flat.reshape(edge, edge, edge, 3)
    lut = np.transpose(cube_bgr, (2, 1, 0, 3))  # -> lut[r, g, b, ch]
    return lut if target_size is None else resample_lut(lut, target_size)


def lut_to_hald(lut_abs: np.ndarray, level: int | None = None) -> np.ndarray:
    """Inverse of :func:`haldclut_to_lut` (used to synthesize HaldCLUT fixtures/tests)."""
    n = lut_abs.shape[0]
    if level is None:
        # smallest level whose cube edge >= n; for tests we resample to edge=level**2
        level = int(np.ceil(np.sqrt(n)))
    edge = level * level
    src = resample_lut(lut_abs, edge)
    cube_bgr = np.transpose(src, (2, 1, 0, 3))  # lut[r,g,b] -> [b,g,r]
    flat = cube_bgr.reshape(edge ** 3, 3)
    side = level ** 3
    img = flat.reshape(side, side, 3)
    return np.clip(img, 0.0, 1.0)


def residual_norm(residual: np.ndarray) -> float:
    """RMS magnitude of a residual LUT (0 for identity)."""
    return float(np.sqrt(np.mean(np.asarray(residual, dtype=np.float64) ** 2)))
