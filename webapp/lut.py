"""Thin webapp-facing wrappers around the repository's canonical LUT operations."""

from __future__ import annotations

from os import PathLike

import numpy as np
from PIL import Image

from data_pipeline.lut_ops import apply_lut_trilinear
from eval.behavioral_fidelity import decode_codes
from eval.cube_io import write_cube


def decode(codes, *, final_dir=None) -> np.ndarray:
    """Decode 64 VQ codes into an absolute ``[17, 17, 17, 3]`` LUT."""
    return decode_codes(codes, final_dir=final_dir)


def apply_lut(image_rgb: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """Apply an absolute LUT to an encoded-sRGB float image and clip to ``[0, 1]``."""
    out = apply_lut_trilinear(lut, image_rgb)
    return np.clip(out, 0.0, 1.0)


def export_cube(lut: np.ndarray, path: str | PathLike[str]) -> None:
    """Write a canonical, R-fastest ``.cube`` file from an absolute LUT."""
    lut_abs = np.clip(np.asarray(lut, dtype=np.float64), 0.0, 1.0)
    write_cube(path, lut_abs)


def load_image(path: str | PathLike[str]) -> Image.Image:
    """Load an image as an RGB PIL image."""
    return Image.open(path).convert("RGB")


def save_image(img: Image.Image | np.ndarray, path: str | PathLike[str]) -> None:
    """Save a PIL image or an RGB uint8 ndarray."""
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    img.save(path)
