"""Stage 9 frozen eval-set structure (eval_harness_implementation.md "Eval Splits").

Builds the eval-slice manifest from split-reserved rows: usage-weighted headline (faithful
global fits -- gold, or diagnostic with mean fit <= HEADLINE_FIT_MAX -- non-procedural), plus
diagnostic and qualitative. Procedural rows are headline-ineligible
(ADR 0016) and land in the diagnostic slice. On a procedural-only pool the headline slice is
empty and the manifest records it as diagnostic-only. Sizes are targets from config; not
enforced on a small smoke pool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .constants import EVAL_SET_VERSION_PLACEHOLDER

# master-plan / eval-harness targets (floors in data_collection_plan are lower).
DEFAULT_EVAL_SIZES = {"headline_supported": 800, "headline_unsupported": 200, "qualitative": 100}

# Headline eligibility is a *fidelity* bar, not a tier label: a row is headline-worthy if it is a
# faithful global fit, whether or not it reached gold. gold rows qualify automatically; diagnostic
# rows demoted for a reason orthogonal to global fidelity (mild structure / moderate smoothness)
# qualify too when their held-out mean fit is at or under this bar (= pair-fit mean_gold). Direct
# LUTs have mean 0.0 (global by construction) and so clear it. Keeps genuinely poor fits out.
HEADLINE_FIT_MAX = 2.5


@dataclass
class EvalCandidate:
    id: str
    split: str                       # eval | diagnostic | qualitative | train | ...
    is_supported: bool = True
    representability_tier: Optional[str] = None
    procedural_filler: bool = False
    unsupported_category: Optional[str] = None
    fit_deltaE00_mean: Optional[float] = None


@dataclass
class EvalSetManifest:
    eval_set_version: str
    sizes_target: dict = field(default_factory=dict)
    slices: dict = field(default_factory=dict)         # slice -> [row ids]
    headline_eligible_count: int = 0
    diagnostic_only: bool = False
    notes: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "eval_set_version": self.eval_set_version,
            "sizes_target": self.sizes_target,
            "slice_counts": {k: len(v) for k, v in self.slices.items()},
            "headline_eligible_count": self.headline_eligible_count,
            "diagnostic_only": self.diagnostic_only,
            "notes": self.notes,
        }


def build_eval_sets(candidates: list[EvalCandidate], sizes: Optional[dict] = None,
                    version: str = EVAL_SET_VERSION_PLACEHOLDER) -> EvalSetManifest:
    sizes = sizes or DEFAULT_EVAL_SIZES
    slices: dict[str, list[str]] = {
        "usage_weighted_headline_supported": [],
        "usage_weighted_headline_unsupported": [],
        "diagnostic": [],
        "qualitative": [],
    }
    for c in candidates:
        if c.split not in ("eval", "diagnostic", "qualitative"):
            continue
        headline_ok = (not c.procedural_filler) and (
            c.representability_tier == "gold"
            or (c.representability_tier == "diagnostic_only"
                and c.fit_deltaE00_mean is not None
                and c.fit_deltaE00_mean <= HEADLINE_FIT_MAX))
        if c.split == "qualitative":
            slices["qualitative"].append(c.id)
        elif c.split == "diagnostic" or not headline_ok:
            slices["diagnostic"].append(c.id)
        elif c.is_supported:
            slices["usage_weighted_headline_supported"].append(c.id)
        else:
            slices["usage_weighted_headline_unsupported"].append(c.id)

    headline_n = (len(slices["usage_weighted_headline_supported"])
                  + len(slices["usage_weighted_headline_unsupported"]))
    manifest = EvalSetManifest(
        eval_set_version=version, sizes_target=sizes, slices=slices,
        headline_eligible_count=headline_n, diagnostic_only=(headline_n == 0),
    )
    if manifest.diagnostic_only:
        manifest.notes.append("no headline-eligible rows (procedural-only / no gold eval rows): "
                               "eval set is diagnostic-only")
    return manifest
