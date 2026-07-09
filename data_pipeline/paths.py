"""Mandated artifact layout (training_plan_colab.md "Artifact Storage").

All long stages write under a single configurable artifact root so runs are resumable and
relocatable. Defaults to the current working directory (the repo), which matches the
``luts/raw/...`` + ``data/raw_registry/`` layout declared in configs/source_inventory.yaml.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path

    # luts/
    @property
    def luts_raw(self) -> Path:
        return self.root / "luts" / "raw"

    @property
    def canonical_absolute(self) -> Path:
        return self.root / "luts" / "canonical_absolute"

    @property
    def canonical_residual(self) -> Path:
        return self.root / "luts" / "canonical_residual"

    # data/
    @property
    def raw_registry(self) -> Path:
        return self.root / "data" / "raw_registry"

    @property
    def splits(self) -> Path:
        return self.root / "data" / "splits"

    @property
    def active_sft(self) -> Path:
        return self.root / "data" / "active_sft"

    @property
    def eval_sets(self) -> Path:
        return self.root / "data" / "eval"

    @property
    def warmup(self) -> Path:
        return self.root / "data" / "warmup"

    @property
    def support_maps(self) -> Path:
        return self.root / "luts" / "support_maps"

    def raw_family(self, family_subdir: str) -> Path:
        """Raw acquisition dir for a source, e.g. ``ppr10k`` -> ``luts/raw/ppr10k``."""
        return self.luts_raw / family_subdir

    def all_dirs(self) -> list[Path]:
        return [
            self.luts_raw,
            self.canonical_absolute,
            self.canonical_residual,
            self.support_maps,
            self.raw_registry,
            self.splits,
            self.active_sft,
            self.eval_sets,
            self.warmup,
        ]

    def ensure(self) -> "ArtifactPaths":
        for d in self.all_dirs():
            d.mkdir(parents=True, exist_ok=True)
        return self


def artifact_paths(root: str | os.PathLike | None = None) -> ArtifactPaths:
    """Build an :class:`ArtifactPaths` rooted at ``root`` (default: env or cwd)."""
    if root is None:
        root = os.environ.get("SLM_ARTIFACT_ROOT", os.getcwd())
    return ArtifactPaths(root=Path(root).resolve())
