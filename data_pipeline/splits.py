"""Stage 6 split manifest + eval reservations (data_collection_plan.md "Splits And Leakage").

Deterministic split units: rows sharing a base identity (PPR10K group / FiveK photo / pack
LUT) OR flagged as near-duplicates by :mod:`data_pipeline.leakage` are unioned into one split
unit, so no near-neighbor can cross a split boundary. Units are then assigned to
train/eval/diagnostic/qualitative by a stable hash; procedural fillers are forced train-only
(headline-ineligible). The resulting assignment is verified leakage-clean.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Optional

from .leakage import LeakageChecker, LeakageItem, leakage_report

DEFAULT_RATIOS = {"train": 0.80, "eval": 0.10, "diagnostic": 0.07, "qualitative": 0.03}


@dataclass
class SplitCandidate:
    id: str
    base_key: Optional[str] = None
    procedural: bool = False
    # leakage features (mirror LeakageItem)
    lut_hash: Optional[str] = None
    image_hash: Optional[str] = None
    prompt_template_hash: Optional[str] = None
    phash: Optional[int] = None
    residual_vec: object = None
    prompt_text: Optional[str] = None

    def base(self) -> str:
        return self.base_key or self.lut_hash or self.image_hash or self.id


class _UnionFind:
    def __init__(self, ids):
        self.parent = {i: i for i in ids}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _as_item(c: SplitCandidate) -> LeakageItem:
    return LeakageItem(id=c.id, split="train", lut_hash=c.lut_hash, image_hash=c.image_hash,
                       prompt_template_hash=c.prompt_template_hash, phash=c.phash,
                       residual_vec=c.residual_vec, prompt_text=c.prompt_text)


def assign_split_units(candidates: list[SplitCandidate],
                       checker: Optional[LeakageChecker] = None) -> dict[str, str]:
    """Union rows by shared base identity and near-duplicate leakage; return id -> unit id."""
    uf = _UnionFind([c.id for c in candidates])
    # union by base key
    by_base: dict[str, list[str]] = {}
    for c in candidates:
        by_base.setdefault(c.base(), []).append(c.id)
    for ids in by_base.values():
        for other in ids[1:]:
            uf.union(ids[0], other)
    # union by near-duplicate leakage (any axis)
    items = [_as_item(c) for c in candidates]
    if checker is None:
        checker = LeakageChecker.from_config(items)
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if checker.pair_axes(items[i], items[j]):
                uf.union(items[i].id, items[j].id)
    # component -> stable unit id
    comp_members: dict[str, list[str]] = {}
    for c in candidates:
        comp_members.setdefault(uf.find(c.id), []).append(c.id)
    id_to_unit: dict[str, str] = {}
    for root, members in comp_members.items():
        unit_id = "unit_" + hashlib.sha256("|".join(sorted(members)).encode()).hexdigest()[:16]
        for m in members:
            id_to_unit[m] = unit_id
    return id_to_unit


def _bucket(unit_id: str, seed: int, ratios: dict[str, float]) -> str:
    h = int(hashlib.sha256(f"{unit_id}:{seed}".encode()).hexdigest(), 16)
    x = (h % 10_000_000) / 10_000_000.0
    cum = 0.0
    for name, frac in ratios.items():
        cum += frac
        if x < cum:
            return name
    return list(ratios.keys())[-1]


@dataclass
class SplitManifest:
    split_id: str
    leakage_policy_version: str
    leakage_report_hash: str
    leakage_status: str
    ratios: dict = field(default_factory=dict)
    assignments: dict = field(default_factory=dict)   # row id -> {"split_unit_id","split"}
    unit_count: int = 0

    def split_of(self, row_id: str) -> Optional[str]:
        a = self.assignments.get(row_id)
        return a["split"] if a else None


def build_split_manifest(candidates: list[SplitCandidate], seed: int = 1234,
                         ratios: Optional[dict] = None,
                         thresholds_path: Optional[str] = None) -> SplitManifest:
    ratios = ratios or DEFAULT_RATIOS
    items = [_as_item(c) for c in candidates]
    checker = (LeakageChecker.from_config(items, thresholds_path) if thresholds_path
               else LeakageChecker.from_config(items))
    id_to_unit = assign_split_units(candidates, checker=checker)

    # procedural fillers are train-only -> force their whole unit to train
    procedural_units = {id_to_unit[c.id] for c in candidates if c.procedural}

    assignments: dict[str, dict] = {}
    unit_split: dict[str, str] = {}
    for c in candidates:
        unit = id_to_unit[c.id]
        if unit not in unit_split:
            unit_split[unit] = "train" if unit in procedural_units else _bucket(unit, seed, ratios)
        assignments[c.id] = {"split_unit_id": unit, "split": unit_split[unit]}

    # verify leakage-clean under the assignment
    verify_items = [
        LeakageItem(id=c.id, split=assignments[c.id]["split"], lut_hash=c.lut_hash,
                    image_hash=c.image_hash, prompt_template_hash=c.prompt_template_hash,
                    phash=c.phash, residual_vec=c.residual_vec, prompt_text=c.prompt_text)
        for c in candidates
    ]
    report = leakage_report(verify_items, checker=checker)

    split_id = "split_" + hashlib.sha256(
        ("|".join(f"{k}:{v['split']}" for k, v in sorted(assignments.items()))).encode()
    ).hexdigest()[:16]
    return SplitManifest(
        split_id=split_id,
        leakage_policy_version=report.leakage_policy_version,
        leakage_report_hash=report.leakage_report_hash,
        leakage_status=report.status,
        ratios=ratios,
        assignments=assignments,
        unit_count=len(set(id_to_unit.values())),
    )
