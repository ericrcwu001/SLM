"""Usage-aware, quota-constrained diversity selection (ADR 0015; data_collection_plan.md).

Pipeline: hard-gate-passing candidates -> usage-prior bucket assignment -> per-family hard
caps -> facility-location/MMR diversity within buckets -> bounded coverage-tail budget.
Deterministic under a seed. "kNN finds what is too close; usage buckets decide what matters;
facility-location/MMR decides what survives inside each bucket."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# --- Deliberate divergence from data_collection_plan.md "Diversity And Usage-Aware Culling" ---
# That doc's recipe (steps 4-5) calls for unsupervised clustering (k-means / HDBSCAN / Leiden)
# with per-cluster seed quotas. This module intentionally does NOT implement it. ADR 0015
# (Accepted; supersedes 0005/0013) settled on usage-prior buckets + facility-location/MMR +
# source/family quotas as the coverage mechanism, and names HDBSCAN only as a negative guardrail
# ("noise is not used for seeding unless manually approved"). Farthest-point MMR (`mmr_select`)
# already spreads selections across the embedding, so cluster quotas would be largely redundant
# while adding hyperparameters (min_cluster_size / resolution / k) and heavy deps (hdbscan,
# leidenalg) that break this pipeline's deterministic, pure-NumPy contract. Do NOT add clustering
# as a selection driver without first revising ADR 0015 — a "TODO: implement clustering" here
# would contradict the accepted decision.
#
# Two pieces the accepted spec DOES require but that are still unimplemented (do these — but keep
# them lightweight; neither needs full clustering):
#   1. Bounded coverage-tail budget (ADR 0015). `coverage_tail` is currently just a 10% usage
#      bucket in DEFAULT_BUCKET_ALLOC; there is no hard outlier cap or manual-approval gate, so
#      "without letting outliers dominate" is not actually enforced. Add a cap on tail/outlier
#      rows (see the coverage_tail branch in `select_active` and SelectionReport.coverage_tail_used).
#   2. LUT-behavior cluster-dominance metric (acceptance criterion #2, data_collection_plan.md
#      L931). active_dataset.AcceptanceChecker.no_dominance checks source_family only; it should
#      also assert no usage bucket / LUT-behavior cluster dominates. A cheap report-only k-means
#      over the selection embedding would satisfy this WITHOUT feeding back into seeding. That
#      check lives in active_dataset.py, not here.

# ADR 0015 source-mix hard caps (fraction of the active supported set).
SOURCE_CAPS = {
    "ppr10k_derived": 0.25,
    "fivek_derived": 0.25,
    "fresh_luts": 0.25,
    "gmic_rawtherapee": 0.30,
    "smaller_public_packs": 0.20,
    "controlled_procedural": 0.10,
}
USAGE_BUCKETS = ("common_head", "common_style", "subtle_control", "boundary_refusal", "coverage_tail")
DEFAULT_BUCKET_ALLOC = {
    "common_head": 0.40, "common_style": 0.30, "subtle_control": 0.15,
    "boundary_refusal": 0.05, "coverage_tail": 0.10,
}


@dataclass
class SelectionCandidate:
    id: str
    family: str
    usage_prior_bucket: str = "common_head"
    embedding: Optional[np.ndarray] = None
    procedural: bool = False


@dataclass
class SelectionReport:
    selected_ids: list = field(default_factory=list)
    per_family: dict = field(default_factory=dict)
    per_bucket: dict = field(default_factory=dict)
    family_caps: dict = field(default_factory=dict)
    target_size: int = 0
    effective_size: int = 0
    target_met: bool = False
    coverage_tail_used: int = 0
    notes: list = field(default_factory=list)


def mmr_select(embeddings: np.ndarray, k: int, seed: int = 0) -> list[int]:
    """Greedy facility-location / max-min-distance selection of ``k`` diverse rows."""
    n = embeddings.shape[0]
    if k >= n:
        return list(range(n))
    if k <= 0:
        return []
    # deterministic seed pick: row with largest norm (ties -> lowest index)
    norms = np.linalg.norm(embeddings, axis=1)
    first = int(np.lexsort((np.arange(n), -norms))[0])
    selected = [first]
    min_dist = np.linalg.norm(embeddings - embeddings[first], axis=1)
    while len(selected) < k:
        min_dist[selected] = -1.0
        nxt = int(np.lexsort((np.arange(n), -min_dist))[0])
        selected.append(nxt)
        d = np.linalg.norm(embeddings - embeddings[nxt], axis=1)
        min_dist = np.where(min_dist >= 0, np.minimum(min_dist, d), min_dist)
    return selected


def select_active(candidates: list[SelectionCandidate], target_size: int,
                  source_caps: Optional[dict] = None, seed: int = 1234) -> SelectionReport:
    """Select up to ``target_size`` candidates honoring per-family caps + bucket ordering.

    Per-family caps bind against the *realized* pool size (``min(target_size, len(pool))``),
    not the aspirational ``target_size``. Below target this keeps the anti-dominance guarantee:
    a 25% cap on a ~2.9k pool holds a family to ~725 rows instead of the non-binding
    ``floor(0.25 * 12000) = 3000``. At or above target the denominator is ``target_size``, so
    the surplus / cull-down behavior is unchanged (ADR 0015).

    NOTE: this only makes the *per-family* caps bind. The combined expert-source guarantee
    (ppr10k + fivek <= 50% of the shipped set) can still be violated when the gate-passing
    supply is itself dominated by expert sources — that is a supply-mix problem, not a
    selection one, and resolves once non-expert supply is added.
    """
    caps = source_caps or SOURCE_CAPS
    report = SelectionReport(target_size=target_size, family_caps=dict(caps))

    # Denominator for the fraction caps: the size we can actually ship. In deficit (pool below
    # target) this is the gate-passing pool; in surplus it is target_size, i.e. identical to the
    # original behavior. This is what makes a "25% cap" mean 25% of the real dataset.
    effective = min(int(target_size), len(candidates))
    report.effective_size = effective

    # per-family hard caps (count), against the realized size rather than the target.
    family_max = {fam: int(np.floor(frac * effective)) for fam, frac in caps.items()}

    # group by (bucket) then run MMR, but enforce family caps globally
    chosen: list[str] = []
    family_count: dict[str, int] = {}
    bucket_count: dict[str, int] = {}

    buckets: dict[str, list] = {b: [] for b in USAGE_BUCKETS}
    for c in candidates:
        key = c.usage_prior_bucket if c.usage_prior_bucket in buckets else "coverage_tail"
        buckets[key].append(c)

    for bucket in USAGE_BUCKETS:
        members = buckets.get(bucket, [])
        if not members:
            continue
        embs = _stack([m.embedding for m in members])
        # Rank the whole bucket by facility-location/MMR diversity; the per-family caps and the
        # realized-size ceiling do the culling. (Bucket *share* quotas are deferred until
        # usage_prior_bucket is populated upstream; today real rows default to common_head, so a
        # per-bucket allocation would be a no-op. See DEFAULT_BUCKET_ALLOC.)
        order = mmr_select(embs, len(members), seed=seed)  # full diversity ranking
        for idx in order:
            if len(chosen) >= effective:
                break
            c = members[idx]
            cap = family_max.get(c.family)
            if cap is not None and family_count.get(c.family, 0) >= cap:
                continue
            chosen.append(c.id)
            family_count[c.family] = family_count.get(c.family, 0) + 1
            bucket_count[bucket] = bucket_count.get(bucket, 0) + 1
        if len(chosen) >= effective:
            break

    report.selected_ids = chosen
    report.per_family = family_count
    report.per_bucket = bucket_count
    report.coverage_tail_used = bucket_count.get("coverage_tail", 0)
    report.target_met = len(chosen) >= target_size
    if not report.target_met:
        report.notes.append(
            f"selected {len(chosen)}/{target_size} (caps bound against realized pool "
            f"size={effective}); family caps or pool size limited coverage"
        )
    return report


def _stack(embs: list) -> np.ndarray:
    dim = next((e.shape[0] for e in embs if e is not None), 1)
    return np.stack([e if e is not None else np.zeros(dim) for e in embs], axis=0)
