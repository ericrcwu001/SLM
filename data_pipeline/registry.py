"""Provenance registry (master-plan Stage 3; data_collection_plan.md "Provenance Registry").

One traceable, removable row per candidate. The field set is the verbatim list from the
spec (lines 84-215); the two ``fit_(train|validation)_deltaE00_*`` globs are represented as
nested dicts. ``RegistryStore`` persists rows as JSONL (+ optional parquet) under
``data/raw_registry/`` and computes the ADR-0016 family-removal invalidation scope.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

from .constants import CANONICAL_DOMAIN_ID, REGISTRY_SCHEMA_VERSION

# Minimal traceable + removable contract: enough to find origin and to remove downstream.
REQUIRED_FIELDS = (
    "source_family",
    "source_pack_id",
    "file_hash",
    "canonical_domain_id",
)


@dataclass
class ProvenanceRow:
    # --- identity / source ---
    source_family: Optional[str] = None
    source_url_or_dataset: Optional[str] = None
    download_timestamp: Optional[str] = None
    author_uploader_pack_id: Optional[str] = None
    image_id: Optional[str] = None
    input_image_id: Optional[str] = None
    canonical_input_image_hash: Optional[str] = None
    input_phash: Optional[str] = None
    input_embedding_id: Optional[str] = None
    image_split_unit_id: Optional[str] = None
    original_image_id: Optional[str] = None
    source_photo_id: Optional[str] = None
    ppr_group_id: Optional[str] = None
    group_id: Optional[str] = None
    expert_id: Optional[str] = None
    source_pack_id: Optional[str] = None
    lut_id: Optional[str] = None
    target_id: Optional[str] = None
    file_hash: Optional[str] = None
    perceptual_hash: Optional[str] = None
    normalized_lut_hash: Optional[str] = None
    # --- canonical domain metadata ---
    canonical_domain_id: Optional[str] = None
    canonical_color_space: Optional[str] = None
    canonical_transfer: Optional[str] = None
    canonical_white_point: Optional[str] = None
    canonical_range: Optional[str] = None
    lut_grid_size: Optional[str] = None
    lut_representation: Optional[str] = None
    interpolation_method: Optional[str] = None
    axis_order: Optional[str] = None
    cube_table_order: Optional[str] = None
    token_flatten_order: Optional[str] = None
    # --- color management / ICC ---
    raw_color_space: Optional[str] = None
    raw_transfer: Optional[str] = None
    raw_icc_profile_description: Optional[str] = None
    raw_icc_profile_sha256: Optional[str] = None
    profile_source: Optional[str] = None
    conversion_engine: Optional[str] = None
    conversion_intent: Optional[str] = None
    black_point_compensation: Optional[bool] = None
    canonical_absolute_lut_hash: Optional[str] = None
    canonical_residual_lut_hash: Optional[str] = None
    # --- tokenizer ---
    tokenizer_version: Optional[str] = None
    vq_codebook_sha256: Optional[str] = None
    vq_decoder_sha256: Optional[str] = None
    normalization_warnings: list = field(default_factory=list)
    out_of_gamut_rate_before_canonical_clip: Optional[float] = None
    canonical_clip_rate_from_conversion: Optional[float] = None
    bit_depth_pipeline: Optional[str] = None
    # --- derivation ---
    derivation_method: Optional[str] = None
    derivation_path: Optional[str] = None
    renderer_version: Optional[str] = None
    raw_processor_version: Optional[str] = None
    color_pipeline_id: Optional[str] = None
    raw_edit_metadata_path: Optional[str] = None
    # --- XMP ---
    xmp_hash: Optional[str] = None
    xmp_parser_version: Optional[str] = None
    xmp_global_fields_present: list = field(default_factory=list)
    xmp_rejected_fields: list = field(default_factory=list)
    xmp_local_tool_count: Optional[int] = None
    xmp_parse_status: Optional[str] = None
    # --- representability / quality ---
    representability_status: Optional[str] = None
    representability_tier: Optional[str] = None
    reject_reason_codes: list = field(default_factory=list)
    fit_deltaE00_mean: Optional[float] = None
    fit_deltaE00_median: Optional[float] = None
    fit_deltaE00_p95: Optional[float] = None
    fit_deltaE00_p99: Optional[float] = None
    fit_deltaE00_max: Optional[float] = None
    fit_train_deltaE00: dict = field(default_factory=dict)       # *_mean/median/p95/p99/max
    fit_validation_deltaE00: dict = field(default_factory=dict)  # *_mean/median/p95/p99/max
    residual_tile_p95: Optional[float] = None
    residual_tile_max: Optional[float] = None
    residual_xy_r2: Optional[float] = None
    residual_moran_i: Optional[float] = None
    residual_edge_corr: Optional[float] = None
    largest_high_residual_component_pct: Optional[float] = None
    support_map_path: Optional[str] = None
    support_map_hash: Optional[str] = None
    supported_cell_rate: Optional[float] = None
    input_pixel_supported_rate: Optional[float] = None
    generic_input_supported_rate: Optional[float] = None
    quality_scores: dict = field(default_factory=dict)
    quality_filter_version: Optional[str] = None
    # --- behavior / tags ---
    behavior_vector_version: Optional[str] = None
    behavior_probe_id: Optional[str] = None
    structured_tags: list = field(default_factory=list)
    style_bundle: Optional[str] = None
    unsupported_category: Optional[str] = None
    mixed_boundary_case: Optional[bool] = None
    measured_behavior: dict = field(default_factory=dict)  # the ~29-field behavior vector
    # --- prompt / teacher / judge ---
    prompt_id: Optional[str] = None
    prompt_template_family: Optional[str] = None
    prompt_template_hash: Optional[str] = None
    # Source-authored instruction (e.g. MMArt-PPR10K user_want_*): authoritative, so the
    # teacher is skipped for these rows. Only env-var names / model ids are ever recorded.
    authored_instruction: Optional[str] = None
    authored_instruction_natural: Optional[str] = None
    authored_instruction_source: Optional[str] = None
    teacher_provider: Optional[str] = None
    teacher_model_id: Optional[str] = None
    teacher_endpoint_env: Optional[str] = None
    teacher_api_key_env: Optional[str] = None
    teacher_model_version: Optional[str] = None
    teacher_prompt_version: Optional[str] = None
    prompt_generation_batch_id: Optional[str] = None
    prompt_seed: Optional[int] = None
    judge_provider: Optional[str] = None
    judge_model_id: Optional[str] = None
    judge_endpoint_env: Optional[str] = None
    judge_api_key_env: Optional[str] = None
    judge_prompt_version: Optional[str] = None
    judge_batch_id: Optional[str] = None
    credential_profile: Optional[str] = None
    # --- selection / usage ---
    selection_bucket: Optional[str] = None
    usage_prior_bucket: Optional[str] = None
    usage_weight: Optional[float] = None
    selection_reason: Optional[str] = None
    procedural_filler: bool = False
    headline_eligible: bool = False
    # --- usage flags (drive removal invalidation) ---
    used_for_tokenizer: bool = False
    used_for_warmup: bool = False
    used_for_sft: bool = False
    used_for_eval: bool = False
    eval_reserved: bool = False
    diagnostic_only: bool = False
    # --- pairing / split / versioning ---
    source_pair_id: Optional[str] = None
    paired_input_image_hash: Optional[str] = None
    split_unit_id: Optional[str] = None
    split_id: Optional[str] = None
    active_set_version: Optional[str] = None
    eval_set_version: Optional[str] = None
    warmup_set_version: Optional[str] = None
    leakage_report_hash: Optional[str] = None
    leakage_policy_version: Optional[str] = None
    rights_notes: Optional[str] = None
    # --- operational (working paths for derivation; not part of the spec record set) ---
    source_image_path: Optional[str] = None
    target_image_path: Optional[str] = None
    # --- registry bookkeeping ---
    registry_schema_version: str = REGISTRY_SCHEMA_VERSION

    @classmethod
    def field_names(cls) -> set[str]:
        return {f.name for f in fields(cls)}

    @classmethod
    def from_dict(cls, d: dict) -> "ProvenanceRow":
        known = cls.field_names()
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


def validate_row(row: ProvenanceRow) -> list[str]:
    """Structural validation of a registry row. Returns error strings ([] == valid)."""
    errors: list[str] = []
    for f in REQUIRED_FIELDS:
        if getattr(row, f) in (None, ""):
            errors.append(f"missing_required:{f}")
    if row.canonical_domain_id and row.canonical_domain_id != CANONICAL_DOMAIN_ID:
        errors.append(f"bad_canonical_domain_id:{row.canonical_domain_id}")
    if row.headline_eligible and row.representability_tier not in (None, "gold"):
        errors.append(f"headline_row_tier_{row.representability_tier}_not_gold")
    if not row.is_removable():
        errors.append("not_removable:no_usage_flags_trackable")
    return errors


def _is_removable(row: ProvenanceRow) -> bool:
    # Removable == we can identify what a removal invalidates. Always true structurally,
    # because usage flags default to False; kept as a hook so a subclass can tighten it.
    return isinstance(row.used_for_tokenizer, bool)


# attach as method (keeps dataclass body clean)
ProvenanceRow.is_removable = lambda self: _is_removable(self)  # type: ignore[attr-defined]


class RegistryStore:
    """Append/load provenance rows as JSONL under ``data/raw_registry/``."""

    def __init__(self, registry_dir: str | Path, filename: str = "provenance.jsonl"):
        self.dir = Path(registry_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / filename

    def add(self, row: ProvenanceRow) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")

    def write_all(self, rows: list[ProvenanceRow]) -> None:
        with open(self.path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")

    def load(self) -> list[ProvenanceRow]:
        if not self.path.exists():
            return []
        rows: list[ProvenanceRow] = []
        with open(self.path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(ProvenanceRow.from_dict(json.loads(line)))
        return rows

    def to_parquet(self, path: str | Path) -> None:
        import pandas as pd

        df = pd.DataFrame([r.to_dict() for r in self.load()])
        df.to_parquet(path)


def removal_manifest(rows: list[ProvenanceRow], removed_family: str) -> dict:
    """Compute the ADR-0016 invalidation scope for removing a source family."""
    affected = [r for r in rows if r.source_family == removed_family]
    invalidates = {
        "tokenizer": any(r.used_for_tokenizer for r in affected),
        "warmup": any(r.used_for_warmup for r in affected),
        "sft": any(r.used_for_sft for r in affected),
        "eval": any(r.used_for_eval for r in affected),
    }
    actions: list[str] = []
    if invalidates["tokenizer"]:
        actions.append("retrain_tokenizer;regen_vq_hashes;retokenize;rebuild_warmup_active_eval")
    if invalidates["warmup"]:
        actions.append("rebuild_warmup_set_version;rebuild_warmup_adapters")
    if invalidates["sft"]:
        actions.append("rebuild_active_sft;rebuild_downstream_adapters")
    if invalidates["eval"]:
        actions.append("freeze_new_eval_set_version;do_not_compare_across_versions")
    return {
        "removed_family": removed_family,
        "affected_row_count": len(affected),
        "affected_file_hashes": sorted({r.file_hash for r in affected if r.file_hash}),
        "invalidates": invalidates,
        "required_actions": actions,
    }
