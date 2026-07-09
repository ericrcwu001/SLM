"""Shared acquisition types: raw artifacts, limits, reports, connector protocol.

A connector fetches raw assets into ``luts/raw/...`` and yields :class:`RawArtifact`
descriptors. The orchestrator turns those into raw provenance rows (Stage 3). Derivation +
canonicalization (Stage 4) happen later, reading the raw registry.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Protocol, runtime_checkable

from ..registry import ProvenanceRow


@dataclass
class AcquireLimits:
    """Bounded acquisition. ``max_items=None`` means uncapped (e.g. FreshLUTs per user)."""
    max_items: Optional[int] = 200
    max_bytes: Optional[int] = None
    rate_limit_s: float = 0.0  # min seconds between remote requests


@dataclass
class RawArtifact:
    kind: str                       # "lut_file" | "image_pair"
    source_pack_id: str
    family: str
    declared_domain: Optional[str] = "srgb"
    license: Optional[str] = None
    source_url: Optional[str] = None
    file_hash: Optional[str] = None
    download_timestamp: Optional[str] = None
    # lut_file
    lut_id: Optional[str] = None
    file_path: Optional[str] = None
    derivation_method: Optional[str] = None   # "haldclut" | "cube" | "pair_fit" | "xmp_grid" | "generated"
    author_uploader_pack_id: Optional[str] = None
    # image_pair (FiveK / PPR10K)
    source_pair_id: Optional[str] = None
    source_image_path: Optional[str] = None
    target_image_path: Optional[str] = None
    xmp_path: Optional[str] = None
    source_photo_id: Optional[str] = None
    group_id: Optional[str] = None
    expert_id: Optional[str] = None
    # provisional tags / selection hints (procedural + tagged sources)
    gold_tags: list = field(default_factory=list)
    style_bundle: Optional[str] = None
    attribute: Optional[str] = None
    usage_prior_bucket: Optional[str] = None
    procedural_filler: bool = False
    # source-authored instruction (e.g. MMArt-PPR10K user_want_*)
    authored_instruction: Optional[str] = None
    authored_instruction_natural: Optional[str] = None
    authored_instruction_source: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def to_registry_row(self) -> ProvenanceRow:
        return ProvenanceRow(
            source_family=self.family,
            source_pack_id=self.source_pack_id,
            source_url_or_dataset=self.source_url,
            download_timestamp=self.download_timestamp or utcnow_iso(),
            file_hash=self.file_hash or "",
            lut_id=self.lut_id,
            source_pair_id=self.source_pair_id,
            source_photo_id=self.source_photo_id,
            group_id=self.group_id,
            expert_id=self.expert_id,
            author_uploader_pack_id=self.author_uploader_pack_id,
            derivation_method=self.derivation_method,
            derivation_path=self.file_path or self.source_image_path,
            raw_edit_metadata_path=self.xmp_path,
            source_image_path=self.source_image_path,
            target_image_path=self.target_image_path,
            raw_color_space=self.declared_domain,
            rights_notes=self.license,
            structured_tags=list(self.gold_tags),
            style_bundle=self.style_bundle,
            usage_prior_bucket=self.usage_prior_bucket,
            procedural_filler=self.procedural_filler,
            authored_instruction=self.authored_instruction,
            authored_instruction_natural=self.authored_instruction_natural,
            authored_instruction_source=self.authored_instruction_source,
            canonical_domain_id=None,  # filled at Stage 4
        )


@dataclass
class AcquireReport:
    source_pack_id: str
    status: str = "ok"               # ok | failed | skipped | partial
    attempted: int = 0
    acquired: int = 0
    failed: int = 0
    skipped: int = 0
    note: Optional[str] = None
    artifacts: list = field(default_factory=list)  # list[RawArtifact]

    def summary(self) -> dict:
        return {
            "source_pack_id": self.source_pack_id,
            "status": self.status,
            "attempted": self.attempted,
            "acquired": self.acquired,
            "failed": self.failed,
            "skipped": self.skipped,
            "note": self.note,
        }


@runtime_checkable
class SourceConnector(Protocol):
    source_pack_id: str
    family: str

    def verify(self) -> tuple[bool, str]:
        """Cheap reachability/credential check. Returns (ok, note)."""

    def acquire(self, raw_root, limits: AcquireLimits) -> AcquireReport:  # noqa: ANN001
        """Fetch bounded raw assets under ``raw_root`` and return a report."""


class RateLimiter:
    def __init__(self, min_interval_s: float = 0.0):
        self.min_interval = max(0.0, min_interval_s)
        self._last = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        now = time.monotonic()
        delta = now - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.monotonic()


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
