"""Train-split residual LUT dataset for tokenizer training (leakage-safe).

The tokenizer trains on **train-split accepted** canonical residual LUTs only; eval /
diagnostic / qualitative reserved identities must never be seen (master-plan Stage 7;
data_collection_plan.md "Splits And Leakage Rules"). A held-out *tokenizer-dev* subset
is carved deterministically from the train split for the reconstruction gate — it does
NOT touch the downstream-reserved identities.

Source of truth is a **train manifest** (jsonl): one row per train residual with its
``residual_key``, ``.npy`` path, ``source_family`` and ``representability_tier``. Prefer
a manifest emitted by the data pipeline. :func:`build_records_from_registry` can also
reconstruct records from ``data/raw_registry/provenance.jsonl`` +
``data/splits/split_manifest.json`` + the residual dir — but it EXCLUDES any residual it
cannot positively confirm is train-split (fail-closed, never leak).

NAMING NOTE: the pipeline names each residual ``<lut_id or source_pair_id or file_hash>.npy``
(run_pipeline.py). :func:`build_records_from_registry` mirrors that precedence, so the full
train set resolves from the current registry. That precedence is a pipeline internal,
though — ``scripts/patches/persist_residual_key.py`` (staged, apply after the live run)
persists an explicit ``residual_key`` field so this consumer joins on one authoritative
key instead of replicating the derivation. The builder still fails closed and logs any
residual it cannot confirm is train-split.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from .model import residual_to_input

ACCEPTED_TIERS = ("gold", "diagnostic_only")
TRAIN_MANIFEST_DEFAULT = "data/splits/tokenizer_train_manifest.jsonl"


@dataclass(frozen=True)
class LutRecord:
    residual_key: str
    path: str
    source_family: str
    representability_tier: str
    split: str = "train"

    def to_dict(self) -> dict:
        return asdict(self)


# --- manifest IO --------------------------------------------------------------------
def write_train_manifest(records: list[LutRecord], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")


def load_train_manifest(path: str, allowed_tiers=ACCEPTED_TIERS) -> list[LutRecord]:
    records: list[LutRecord] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("split") != "train":
                continue
            if allowed_tiers and d.get("representability_tier") not in allowed_tiers:
                continue
            if not os.path.exists(d["path"]):
                continue
            records.append(LutRecord(**{k: d[k] for k in ("residual_key", "path", "source_family",
                                                           "representability_tier", "split")}))
    return records


# --- fail-closed reconstruction from the registry -----------------------------------
def build_records_from_registry(
    root: str = ".",
    allowed_tiers=ACCEPTED_TIERS,
    split: str = "train",
) -> tuple[list[LutRecord], dict]:
    """Reconstruct train records from provenance + split manifest + residual dir.

    Fail-closed: a residual is kept ONLY if it resolves to the requested split via the
    persisted registry. Anything unresolved is excluded and counted. Returns
    (records, coverage) where coverage explains what was dropped and why.
    """
    residual_dir = os.path.join(root, "luts", "canonical_residual")
    prov_path = os.path.join(root, "data", "raw_registry", "provenance.jsonl")
    split_path = os.path.join(root, "data", "splits", "split_manifest.json")

    rows = [json.loads(l) for l in open(prov_path, encoding="utf-8")] if os.path.exists(prov_path) else []
    assignments = (
        json.load(open(split_path, encoding="utf-8")).get("assignments", {})
        if os.path.exists(split_path) else {}
    )
    # The residual .npy stem is the pipeline's residual key. run_pipeline.py names it
    # `lut_id or source_pair_id or file_hash`; once the staged patch lands it is also
    # persisted verbatim as `residual_key`. Mirror that precedence to index rows by stem.
    def _key_of(r: dict):
        return (r.get("residual_key") or r.get("_residual_key") or r.get("lut_id")
                or r.get("source_pair_id") or r.get("file_hash"))

    by_key: dict[str, dict] = {}
    for r in rows:
        k = _key_of(r)
        if k is not None:
            by_key.setdefault(str(k), r)

    stems = [os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(residual_dir, "*.npy"))]
    cov = {"residuals_on_disk": len(stems), "unresolved_no_row": 0,
           "unresolved_no_split": 0, "wrong_split": 0, "wrong_tier": 0, "kept": 0}
    records: list[LutRecord] = []
    for stem in sorted(stems):
        row = by_key.get(stem)
        if row is None:
            cov["unresolved_no_row"] += 1
            continue
        rid = row.get("file_hash") or stem
        sp = (assignments.get(rid) or {}).get("split")
        if sp is None:
            cov["unresolved_no_split"] += 1
            continue
        if sp != split:
            cov["wrong_split"] += 1
            continue
        tier = row.get("representability_tier")
        if allowed_tiers and tier not in allowed_tiers:
            cov["wrong_tier"] += 1
            continue
        records.append(LutRecord(
            residual_key=stem,
            path=os.path.join(residual_dir, f"{stem}.npy"),
            source_family=row.get("source_family") or "unknown",
            representability_tier=tier or "unknown",
            split=sp,
        ))
        cov["kept"] += 1
    return records, cov


# --- deterministic tokenizer-dev holdout (carved from train only) -------------------
def _hash_frac(key: str) -> float:
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def dev_holdout(records: list[LutRecord], frac: float = 0.05) -> tuple[list[LutRecord], list[LutRecord]]:
    """Split train records into (train, dev) deterministically by residual_key hash.

    Both subsets are within the train split — the dev subset is the tokenizer's
    reconstruction-gate holdout, distinct from the downstream eval-reserved identities.
    """
    dev = [r for r in records if _hash_frac(r.residual_key) < frac]
    train = [r for r in records if _hash_frac(r.residual_key) >= frac]
    return train, dev


# --- torch Dataset ------------------------------------------------------------------
class ResidualDataset(Dataset):
    """Yields (conv-tensor ``[3,17,17,17]`` float32, family_index) from residual .npy files."""

    def __init__(self, records: list[LutRecord], augment: bool = False, scale_jitter: float = 0.0):
        if not records:
            raise ValueError("empty record set")
        self.records = records
        self.augment = augment
        self.scale_jitter = scale_jitter
        fams = sorted({r.source_family for r in records})
        self.family_to_idx = {f: i for i, f in enumerate(fams)}
        self.families = fams

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        rec = self.records[i]
        res = np.load(rec.path).astype(np.float32)          # [17,17,17,3]
        if self.augment and self.scale_jitter > 0.0:
            # neutral-preserving residual scaling (a light LUT augmentation)
            g = 1.0 + float(np.random.uniform(-self.scale_jitter, self.scale_jitter))
            res = res * g
        x = residual_to_input(res)[0]                        # [3,17,17,17]
        return x, self.family_to_idx[rec.source_family]


def family_balanced_sampler(records: list[LutRecord], num_samples: int | None = None) -> WeightedRandomSampler:
    """Inverse-frequency family weighting so no source family dominates a batch."""
    from collections import Counter

    counts = Counter(r.source_family for r in records)
    weights = torch.tensor([1.0 / counts[r.source_family] for r in records], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=num_samples or len(records), replacement=True)


def load_residual_arrays(records: list[LutRecord]) -> list[np.ndarray]:
    """Load raw ``[17,17,17,3]`` residual arrays (float64) for numpy-side gate metrics."""
    return [np.load(r.path).astype(np.float64) for r in records]
