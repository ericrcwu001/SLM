"""Row / model-output / metric / manifest schemas + the layer status model.

Mirrors docs/eval_harness_implementation.md "Eval Unit", model_architecture.md
"Dataset Interface" / "Version Manifest", and detailed_behavior_spec.md metrics.json.

Version keys are pinned to the doc defaults (cross-doc audit item D3). The
`gating_slice_registry.yaml` strata fields that the Eval Unit omits are added here as
nullable columns (audit item E1) so `stats.py` can stratify without a schema mismatch.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

# --- pinned version / domain constants (audit D3) -------------------------------
CANONICAL_DOMAIN_ID = "slm_lut_v1_srgb_display_encoded_17_trilinear"
CUBE_SERIALIZATION_VERSION = "cube_v1_size17_domain01_rgb_rfast_f10_lf"
ICC_CONVERSION_CONFIG = "srgb_relcol_bpc_float32_v1"
SCHEMA_VERSION = "1.0"
PARSER_VERSION = "1.0"
FSM_VERSION = "1.0"
EVAL_CONFIG_VERSION = "eval_v0_spine"  # this decode-disabled spine build
SAFETY_THRESHOLD_VERSION = "TODO_set_at_color_layer_enable"

AcceptanceMode = Literal[
    "exact_target", "multi_reference", "behavior_window", "multi_reference|behavior_window"
]

# --- layer status model ----------------------------------------------------------
STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_NOT_EVALUATED = "not_evaluated"
STATUS_BLOCKED = "blocked"

DECODER_DISABLED_REASON = "decoder_disabled"


@dataclass
class LayerResult:
    """Result of one evaluation layer (L0..L8) for one row+output."""

    layer: str  # e.g. "L0_boundary", "L1_syntax", "L4_direction"
    status: str  # STATUS_*
    reason: Optional[str] = None
    details: dict = field(default_factory=dict)

    @property
    def is_pass(self) -> bool:
        return self.status == STATUS_PASS

    @classmethod
    def disabled(cls, layer: str, reason: str = DECODER_DISABLED_REASON) -> "LayerResult":
        return cls(layer=layer, status=STATUS_NOT_EVALUATED, reason=reason)


@dataclass
class ParsedOutput:
    """Output of the strict parser (output_parsers.parse_output)."""

    kind: Literal["lut_tokens", "unsupported", "invalid"]
    token_ids: list[int] = field(default_factory=list)  # codebook indices 0..255
    token_count: int = 0
    syntax_pass: bool = False
    parser_errors: list[str] = field(default_factory=list)


@dataclass
class RawModelOutput:
    """One raw output for one row from one adapter/seed/mode."""

    row_id: str
    adapter_id: str
    seed: int
    mode: str  # "free_generation" | "runtime_constrained"
    text: Optional[str] = None
    token_ids: Optional[list[int]] = None  # optional model-vocab ids (constrained path)
    provenance: dict = field(default_factory=dict)


# --- eval row --------------------------------------------------------------------
_EVALROW_FIELDS = {
    "id", "image_path", "image_sha256", "instruction", "is_supported", "support_label",
    "gold_tags", "style_bundle", "style_primary", "target_lut_path", "target_tokens",
    "acceptance_mode", "reference_tokens", "reference_lut_paths", "behavior_window",
    "canonical_domain_id", "representability_tier", "headline_eligible",
    "procedural_filler", "usage_weight", "split", "measured_behavior",
    "derived_lut_quality", "unsupported_category", "unsupported_components",
    "supported_components", "mixed_prompt", "boundary_pair_id", "boundary_pair_role",
    "route", "refuse_kind",
    "source_family", "source_lut_id",
    # strata fields referenced by gating_slice_registry.yaml (audit E1)
    "style_bucket", "usage_prior_bucket", "magnitude_bucket", "behavior_bucket",
    "boundary_type", "attribute", "input_file_type", "compression_bucket",
    "color_profile_bucket", "size_bucket",
    "metadata",
}


@dataclass
class EvalRow:
    id: str
    instruction: str
    is_supported: bool
    image_path: Optional[str] = None
    image_sha256: Optional[str] = None
    support_label: Optional[str] = None
    gold_tags: list[str] = field(default_factory=list)
    style_bundle: Optional[str] = None
    style_primary: Optional[str] = None
    target_lut_path: Optional[str] = None
    target_tokens: list[int] = field(default_factory=list)
    acceptance_mode: str = "exact_target"
    reference_tokens: list[list[int]] = field(default_factory=list)
    reference_lut_paths: list[str] = field(default_factory=list)
    behavior_window: Optional[dict] = None
    canonical_domain_id: Optional[str] = None
    representability_tier: Optional[str] = None
    headline_eligible: bool = False
    procedural_filler: bool = False
    usage_weight: float = 1.0
    split: Optional[str] = None
    measured_behavior: dict = field(default_factory=dict)
    derived_lut_quality: dict = field(default_factory=dict)
    # unsupported / mixed
    unsupported_category: Optional[str] = None
    unsupported_components: list[str] = field(default_factory=list)
    supported_components: list[str] = field(default_factory=list)
    mixed_prompt: bool = False
    # route taxonomy (ADR 0021/0023): {grade, clarify, refuse} + refuse subtype.
    route: Optional[str] = None
    refuse_kind: Optional[str] = None
    boundary_pair_id: Optional[str] = None
    boundary_pair_role: Optional[str] = None
    source_family: Optional[str] = None
    source_lut_id: Optional[str] = None
    # strata (nullable)
    style_bucket: Optional[str] = None
    usage_prior_bucket: Optional[str] = None
    magnitude_bucket: Optional[str] = None
    behavior_bucket: Optional[str] = None
    boundary_type: Optional[str] = None
    attribute: Optional[str] = None
    input_file_type: Optional[str] = None
    compression_bucket: Optional[str] = None
    color_profile_bucket: Optional[str] = None
    size_bucket: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalRow":
        known = {k: v for k, v in d.items() if k in _EVALROW_FIELDS}
        extra = {k: v for k, v in d.items() if k not in _EVALROW_FIELDS}
        row = cls(**known)  # type: ignore[arg-type]
        if extra:
            row.metadata = {**extra, **(row.metadata or {})}
        return row

    def to_dict(self) -> dict:
        return asdict(self)


def validate_row(row: EvalRow) -> list[str]:
    """Structural validation. Returns a list of error strings ([] == valid).

    Note: in this decode-disabled build, supported rows may legitimately carry an
    empty ``target_tokens`` (there is no frozen tokenizer to produce 64 ids yet).
    If ``target_tokens`` is non-empty it must be exactly 64 ids in 0..255.
    """

    errors: list[str] = []
    if not row.id:
        errors.append("missing_id")
    if not row.instruction:
        errors.append("missing_instruction")
    if not isinstance(row.is_supported, bool):
        errors.append("is_supported_not_bool")

    if row.canonical_domain_id and row.canonical_domain_id != CANONICAL_DOMAIN_ID:
        errors.append(f"bad_canonical_domain_id:{row.canonical_domain_id}")

    if row.target_tokens:
        if len(row.target_tokens) != 64:
            errors.append(f"target_tokens_count_{len(row.target_tokens)}_not_64")
        if any((not isinstance(t, int)) or t < 0 or t > 255 for t in row.target_tokens):
            errors.append("target_token_out_of_range")

    if row.is_supported:
        if row.support_label not in (None, "supported"):
            errors.append(f"supported_row_bad_label:{row.support_label}")
    else:
        if row.target_tokens:
            errors.append("unsupported_row_has_target_tokens")
        if not row.unsupported_category:
            errors.append("unsupported_row_missing_category")

    if row.headline_eligible and row.representability_tier not in (None, "gold"):
        # headline rows require gold tier (Target Fidelity). None tolerated in the
        # decode-disabled spine where representability has not been computed.
        errors.append(f"headline_row_tier_{row.representability_tier}_not_gold")

    return errors


def load_rows(path: str) -> list[EvalRow]:
    """Load eval rows from a JSONL file."""
    rows: list[EvalRow] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(EvalRow.from_dict(json.loads(line)))
    return rows


def write_rows(path: str, rows: list[EvalRow]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")


# --- version manifest ------------------------------------------------------------
def build_version_manifest(
    *,
    vocab_added_special_token_ids: dict[str, int],
    vocab_size_after_resize: Optional[int] = None,
    base_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
    decoder_enabled: bool = False,
    extra: Optional[dict] = None,
) -> dict:
    """Assemble version_manifest.json (model_architecture.md "Version Manifest").

    Decode-dependent fields are recorded as ``"pending:decoder_disabled"`` in this
    build; startup self-check treats them as pending rather than hard-fail.
    """

    pending = "pending:decoder_disabled"
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "base_model_id": base_model_id,
        "base_model_revision": pending,
        "adapter_id": pending,
        "adapter_sha256": pending,
        "adapter_step": pending,
        "added_special_token_ids": vocab_added_special_token_ids,
        "vocab_size_after_resize": vocab_size_after_resize
        if vocab_size_after_resize is not None
        else pending,
        "tied_embedding_status": pending,
        "codebook_size": 256,
        "token_count": 64,
        "latent_shape": [4, 4, 4],
        "token_suffix_to_codebook_index": "identity",
        "flatten_order": pending,
        "vq_codebook_sha256": pending,
        "vq_decoder_sha256": pending,
        "lut_grid": [17, 17, 17],
        "canonical_domain_id": CANONICAL_DOMAIN_ID,
        "color_pipeline_version": pending,
        "icc_conversion_config": ICC_CONVERSION_CONFIG,
        "cube_serialization_version": CUBE_SERIALIZATION_VERSION,
        "interpolation": "trilinear",
        "parser_version": PARSER_VERSION,
        "fsm_version": FSM_VERSION,
        "safety_threshold_version": SAFETY_THRESHOLD_VERSION,
        "eval_config_version": EVAL_CONFIG_VERSION,
        "decoder_enabled": decoder_enabled,
    }
    if extra:
        manifest.update(extra)
    return manifest
