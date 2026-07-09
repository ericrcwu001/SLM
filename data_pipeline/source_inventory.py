"""Load + validate configs/source_inventory.yaml (Stage 2 acquisition manifest).

This file is the single source of truth for WHAT to acquire. ``access_method`` is advisory
for HOW; the acquisition orchestrator maps ``source_pack_id`` -> a concrete connector (some
sources declared ``direct_download`` are actually fetched via a HuggingFace mirror).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

ACCESS_METHODS = {"direct_download", "hf_dataset", "scrape", "manual", "generated"}

_DEFAULT_INVENTORY_PATH = Path("configs/source_inventory.yaml")


@dataclass
class SourcePack:
    source_pack_id: str
    priority: int
    family: str
    dataset_name: Optional[str] = None
    url_or_dataset_id: Optional[str] = None
    url_status: Optional[str] = None
    access_method: Optional[str] = None
    expected_layout: Optional[str] = None
    approx_item_count: Optional[int] = None
    notes: Optional[str] = None
    handling_ref: Optional[str] = None
    enabled: bool = True  # inventory-level default; pipeline config may override


@dataclass
class ExcludedSource:
    family: str
    reason: Optional[str] = None


@dataclass
class SourceInventory:
    version: str
    sources: list[SourcePack] = field(default_factory=list)
    excluded: list[ExcludedSource] = field(default_factory=list)

    def get(self, source_pack_id: str) -> SourcePack:
        for s in self.sources:
            if s.source_pack_id == source_pack_id:
                return s
        raise KeyError(f"unknown source_pack_id: {source_pack_id}")

    def by_priority(self) -> list[SourcePack]:
        return sorted(self.sources, key=lambda s: s.priority)

    def excluded_families(self) -> set[str]:
        return {e.family for e in self.excluded}

    def is_excluded(self, family: str) -> bool:
        return family in self.excluded_families()

    def assert_not_excluded(self, family: str) -> None:
        if self.is_excluded(family):
            raise ValueError(f"excluded source family rejected at Stage 2: {family}")

    def validate(self) -> list[str]:
        errors: list[str] = []
        seen: set[str] = set()
        for s in self.sources:
            if s.source_pack_id in seen:
                errors.append(f"duplicate_source_pack_id:{s.source_pack_id}")
            seen.add(s.source_pack_id)
            if s.access_method not in ACCESS_METHODS:
                errors.append(f"bad_access_method:{s.source_pack_id}:{s.access_method}")
            if self.is_excluded(s.family):
                errors.append(f"source_in_excluded_family:{s.source_pack_id}:{s.family}")
        return errors


def _coerce_int(v) -> Optional[int]:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def load_source_inventory(path: str | Path = _DEFAULT_INVENTORY_PATH) -> SourceInventory:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    sources = [
        SourcePack(
            source_pack_id=s["source_pack_id"],
            priority=int(s.get("priority", 999)),
            family=s.get("family", ""),
            dataset_name=s.get("dataset_name"),
            url_or_dataset_id=s.get("url_or_dataset_id"),
            url_status=s.get("url_status"),
            access_method=s.get("access_method"),
            expected_layout=s.get("expected_layout"),
            approx_item_count=_coerce_int(s.get("approx_item_count")),
            notes=s.get("notes"),
            handling_ref=s.get("handling_ref"),
            enabled=bool(s.get("enabled", True)),
        )
        for s in (data.get("sources") or [])
    ]
    excluded = [
        ExcludedSource(family=e.get("family", ""), reason=e.get("reason"))
        for e in (data.get("excluded_sources") or [])
    ]
    return SourceInventory(version=data.get("version", "unknown"), sources=sources, excluded=excluded)
