"""Selection embeddings (data_collection_plan.md "Diversity And Usage-Aware Culling").

Three axes: image semantics (color-stats + pHash; optional CLIP under [ml]), LUT behavior
(residual PCA + measured behavior vector), prompt/tag semantics (structured multi-hot tag
vector; optional text embedding). Pure-NumPy defaults; ML axes are additive when available.
"""

from __future__ import annotations

import numpy as np

from .behavior_vector import measure_behavior
from .leakage import fit_lut_pca

# canonical tag vocabulary (attribute directions + style bundles)
TAG_VOCAB = [
    "warmer", "cooler", "tint_magenta", "tint_green", "brighter", "darker",
    "more_contrast", "less_contrast", "lifted_blacks", "crushed_blacks",
    "brighter_highlights", "lifted_shadows", "more_saturated", "muted",
    "matte", "faded", "filmic", "cinematic", "teal-orange", "sepia", "bleach bypass", "natural",
]
_BEHAVIOR_KEYS = [
    "temperature_delta_b", "tint_delta_a", "mean_l_delta", "contrast_l_spread_delta",
    "black_point_l_delta", "highlight_l_delta", "shadow_l_delta", "chroma_delta",
    "split_tone_strength",
]


def tag_embedding(tags: list[str]) -> np.ndarray:
    v = np.zeros(len(TAG_VOCAB), dtype=np.float64)
    idx = {t: i for i, t in enumerate(TAG_VOCAB)}
    for t in tags or []:
        if t in idx:
            v[idx[t]] = 1.0
    return v


def behavior_embedding(behavior: dict) -> np.ndarray:
    return np.array([float(behavior.get(k, 0.0)) for k in _BEHAVIOR_KEYS], dtype=np.float64)


def lut_behavior_embedding(residual: np.ndarray, behavior: dict | None = None,
                           pca=None) -> np.ndarray:
    """PCA projection of the residual (if a basis is provided) concatenated with the
    normalized measured behavior vector.
    """
    b = behavior if behavior is not None else None
    beh = behavior_embedding(b) if b is not None else np.zeros(len(_BEHAVIOR_KEYS))
    if pca is not None:
        proj = pca.project(np.asarray(residual, dtype=np.float64).reshape(-1))
        return np.concatenate([proj, beh])
    return beh


def image_color_stats(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.max() > 1.0:
        arr = arr / 255.0
    flat = arr.reshape(-1, arr.shape[-1])
    return np.concatenate([flat.mean(axis=0), flat.std(axis=0)])


def build_pca(residuals: list[np.ndarray], dim: int = 64):
    return fit_lut_pca(residuals, dim=dim) if len(residuals) >= 2 else None
