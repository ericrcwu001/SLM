"""Colab/GCS corpus staging (``slm_stage``) — ADR-0018 + ADR-0019.

Moves the artifact corpus between a durable root (local path, Google Drive FUSE mount, or a
``gs://`` GCS bucket) and a fast local root, as bounded, sha256-verified, structure-preserving
tar shards. Three subcommands: ``pack`` (corpus -> shards in the durable root), ``stage``
(shards -> local root, verified/disk-aware/resumable, a drop-in ``SLM_ARTIFACT_ROOT``), and
``push`` (checkpoints/outputs -> durable root so they survive Colab session teardown).

Nothing runs on import; use ``python -m data_pipeline.staging.run_staging <cmd> ...``.
"""

from __future__ import annotations

from .backends import DurableBackend, GcsBackend, HfBackend, LocalBackend, backend_for
from .core import STAGING_VERSION, StageReport, run_pack, run_push, run_stage

__all__ = [
    "DurableBackend",
    "LocalBackend",
    "GcsBackend",
    "HfBackend",
    "backend_for",
    "StageReport",
    "STAGING_VERSION",
    "run_pack",
    "run_stage",
    "run_push",
]
