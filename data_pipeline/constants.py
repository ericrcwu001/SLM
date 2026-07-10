"""Pinned version keys for the data pipeline.

Canonical-domain / cube / ICC constants are reused verbatim from :mod:`eval.schemas`
so the whole system agrees on one identity.
"""

from __future__ import annotations

from eval.schemas import (  # re-exported for one-import convenience
    CANONICAL_DOMAIN_ID,
    CUBE_SERIALIZATION_VERSION,
    ICC_CONVERSION_CONFIG,
)

# Data-pipeline-owned version keys.
PIPELINE_VERSION = "datagen_v0"
REGISTRY_SCHEMA_VERSION = "provenance_v1"
ACQUISITION_POLICY_VERSION = "acq_v1_bounded_resumable"
PROCEDURAL_GENERATOR_VERSION = "proc_v1"
# v2: pair-fit LUTs completed by a Laplacian smooth fill (was identity fallback) + neutral-drift
# gate honours measured tint.
# v3: residual spatial-correlation gates (edge/xy/coord) demoted from hard-reject to
# gold-disqualifiers (within-tolerance fits with a minor local component -> diagnostic, not
# rejected); foldover safety bar relaxed 0.1%->0.5%. Bump invalidates cached tiers.
# v4: smoothness p99 bar relaxed 0.06->0.10 (admit creative/film LUTs whose sharper tonal
# transitions were marginal, not artefacts) + widened intended-tint detection (lower uniform-cast
# floor 1.5->1.0 and treat coherent split-tones as intended) so neutral-drift stops falsely
# rejecting deliberate colour casts. clip/foldover kept strict. Bump invalidates cached tiers.
# v5: smoothness is now resample-aware (measured on the LUT's NATIVE grid, normalized to 17^3-equiv,
# so our trilinear downsampling no longer inflates it) AND demote-don't-reject (moderate 0.10-0.30
# caps tier at diagnostic; only >0.30 hard-rejects). clip/foldover/neutral_drift unchanged.
# v6: gold bar loosened to raise gold yield (headline-eval slice was unbuildable at ~1.4% gold):
# pair-fit mean_gold 2.0->2.5 and support_gold 0.99->0.98; smoothness clean band DIAG 0.10->0.15.
# Accept/reject bars, structure/skin/cap disqualifiers unchanged. Bump invalidates cached tiers.
# v7: graded structure penalty -- a single marginal spatial-structure signal (below a wider gold
# ceiling) no longer blocks gold for pair-fits; >=2 signals or any past its ceiling still demote
# to diagnostic. Reclaims faithful global fits with a faint local component. Bump invalidates tiers.
# v8: behavior_v2 re-measurement (ADR 0022). The gates/tiers are UNCHANGED, but the pipeline's
# cache-currency check keys on QUALITY_FILTER_VERSION (run_pipeline.py:182), NOT on the behavior
# version, so bumping it here is what forces measure_behavior to re-run and write the new
# behavior_v2 fields. Bump invalidates cached tiers (they are recomputed identically).
QUALITY_FILTER_VERSION = "quality_v8_behavior_v2"
# behavior_v2 (ADR 0022): adds absolute/region hue, per-hue saturation, contrast-shape (toe/
# shoulder), and matte as a first-class axis; all 27 behavior_v1 fields retained.
BEHAVIOR_VECTOR_VERSION = "behavior_v2"

# Run-stamped placeholders (frozen at Stage 9 in a real run).
ACTIVE_SET_VERSION_PLACEHOLDER = "active_set_pending_freeze"
EVAL_SET_VERSION_PLACEHOLDER = "eval_set_pending_freeze"
WARMUP_SET_VERSION_PLACEHOLDER = "warmup_set_pending_freeze"

# Canonical grid geometry (matches eval.cube_io.GRID_SIZE / model_architecture.md).
GRID_SIZE = 17
TOKEN_COUNT = 64
CODEBOOK_SIZE = 256

# Status markers for gated materialization (honest, never fabricated).
TOKEN_STATUS_PENDING = "pending_tokenizer"
# Terminal state once the frozen tokenizer has actually encoded the row's residual -> 64 ids.
TOKEN_STATUS_MATERIALIZED = "materialized"
INSTRUCTION_STATUS_PENDING = "pending_teacher"
# Terminal states once the teacher has actually run over a row.
INSTRUCTION_STATUS_GENERATED = "teacher_generated"   # instruction written + validated
INSTRUCTION_STATUS_REJECTED = "rejected_teacher"     # generated but failed a quality gate
# Instruction supplied by the source dataset (e.g. MMArt-PPR10K user_want_*): authoritative,
# the teacher is skipped for these rows (deterministic + leakage gates still apply).
INSTRUCTION_STATUS_AUTHORED = "source_authored"

__all__ = [
    "CANONICAL_DOMAIN_ID",
    "CUBE_SERIALIZATION_VERSION",
    "ICC_CONVERSION_CONFIG",
    "PIPELINE_VERSION",
    "REGISTRY_SCHEMA_VERSION",
    "ACQUISITION_POLICY_VERSION",
    "PROCEDURAL_GENERATOR_VERSION",
    "QUALITY_FILTER_VERSION",
    "BEHAVIOR_VECTOR_VERSION",
    "ACTIVE_SET_VERSION_PLACEHOLDER",
    "EVAL_SET_VERSION_PLACEHOLDER",
    "WARMUP_SET_VERSION_PLACEHOLDER",
    "GRID_SIZE",
    "TOKEN_COUNT",
    "CODEBOOK_SIZE",
    "TOKEN_STATUS_PENDING",
    "TOKEN_STATUS_MATERIALIZED",
    "INSTRUCTION_STATUS_PENDING",
    "INSTRUCTION_STATUS_GENERATED",
    "INSTRUCTION_STATUS_REJECTED",
    "INSTRUCTION_STATUS_AUTHORED",
]
