"""MLX batch iterator for tokenizer training.

Reuses the framework-agnostic record selection from ``tokenizer.data``
(`build_records_from_registry`, `dev_holdout`, `load_residual_arrays`, `load_train_manifest`)
and adds a channels-last MLX batcher with inverse-frequency family balancing. Residual
arrays are `[r,g,b,3]` == MLX's NDHWC, so no axis permute is needed.
"""

from __future__ import annotations

from collections import Counter

import mlx.core as mx
import numpy as np

from .. import data as D  # re-exported for callers

# convenience re-exports
build_records_from_registry = D.build_records_from_registry
dev_holdout = D.dev_holdout
load_residual_arrays = D.load_residual_arrays
load_train_manifest = D.load_train_manifest
LutRecord = D.LutRecord


class MlxBatcher:
    """Preloads all train residuals into one float32 array; yields family-balanced batches.

    Optional neutral-preserving scale-jitter augmentation (mirrors ResidualDataset): each
    sampled residual is multiplied by ``1 +/- U(0, scale_jitter)``. Train-only.
    """

    def __init__(self, records, batch_size: int, seed: int = 0,
                 augment: bool = False, scale_jitter: float = 0.0):
        if not records:
            raise ValueError("empty record set")
        self.records = records
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.augment = augment
        self.scale_jitter = scale_jitter
        self.arrays = np.stack([np.load(r.path).astype(np.float32) for r in records])  # [N,17,17,17,3]
        counts = Counter(r.source_family for r in records)
        w = np.array([1.0 / counts[r.source_family] for r in records], dtype=np.float64)
        self.p = w / w.sum()
        self.N = len(records)

    def batch(self) -> mx.array:
        idx = self.rng.choice(self.N, size=self.batch_size, p=self.p)
        arr = self.arrays[idx]
        if self.augment and self.scale_jitter > 0.0:
            g = (1.0 + self.rng.uniform(-self.scale_jitter, self.scale_jitter,
                                        size=(len(idx), 1, 1, 1, 1))).astype(np.float32)
            arr = arr * g
        return mx.array(arr)  # [B,17,17,17,3] channels-last
