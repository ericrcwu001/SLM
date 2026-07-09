"""Statistics: Wilson CIs, stratified paired bootstrap, McNemar / exact paired
permutation, Holm-Bonferroni multiplicity, seed summaries, and min_N gating.

Implements docs/eval_harness_implementation.md "Statistics" + "Pass Criteria" (the
CI machinery) and honors the gating-slice registry's `min_N`/`underpowered_behavior`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
from scipy import stats as _sps

DEFAULT_BOOTSTRAP_B = 10_000
NOT_EVALUABLE = "not-evaluable-below-min_N"


# --- Wilson interval -------------------------------------------------------------
@dataclass
class WilsonResult:
    k: int
    n: int
    point: Optional[float]
    low: Optional[float]
    high: Optional[float]
    confidence: float = 0.95


def wilson_ci(k: int, n: int, confidence: float = 0.95) -> WilsonResult:
    if n <= 0:
        return WilsonResult(k, n, None, None, None, confidence)
    z = float(_sps.norm.ppf(1 - (1 - confidence) / 2))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return WilsonResult(k, n, p, max(0.0, center - half), min(1.0, center + half), confidence)


# --- paired bootstrap ------------------------------------------------------------
@dataclass
class PairedDeltaResult:
    delta: Optional[float]  # mean(a - b) over shared units, as a rate difference
    ci_low: Optional[float]
    ci_high: Optional[float]
    n: int
    B: int
    seed: int
    confidence: float = 0.95


def paired_delta_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    strata: Optional[Sequence] = None,
    B: int = DEFAULT_BOOTSTRAP_B,
    seed: int = 0,
    confidence: float = 0.95,
) -> PairedDeltaResult:
    """Stratified paired bootstrap of ``mean(a_i - b_i)`` over shared units.

    ``a``/``b`` are aligned per-unit values (0/1 pass indicators for rate deltas, or
    continuous). ``strata`` (optional) resamples within each stratum, preserving
    per-stratum unit counts (spec: "stratified paired bootstrap over row ids").
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must be aligned (same length)")
    n = a.size
    if n == 0:
        return PairedDeltaResult(None, None, None, 0, B, seed, confidence)
    delta = a - b
    point = float(delta.mean())

    if strata is None:
        groups = [np.arange(n)]
    else:
        strata = np.asarray(list(strata))
        groups = [np.where(strata == s)[0] for s in np.unique(strata)]

    rng = np.random.default_rng(seed)
    boots = np.empty(B, dtype=float)
    for i in range(B):
        picks = [g[rng.integers(0, g.size, size=g.size)] for g in groups if g.size]
        idx = np.concatenate(picks)
        boots[i] = delta[idx].mean()
    lo = float(np.percentile(boots, (1 - confidence) / 2 * 100))
    hi = float(np.percentile(boots, (1 + confidence) / 2 * 100))
    return PairedDeltaResult(point, lo, hi, n, B, seed, confidence)


# --- McNemar / exact paired test -------------------------------------------------
@dataclass
class McNemarResult:
    b: int  # a passed, b failed (discordant)
    c: int  # a failed, b passed (discordant)
    p_value: float
    statistic: Optional[float]


def mcnemar(a: Sequence[float], b: Sequence[float]) -> McNemarResult:
    """Exact McNemar on paired binary outcomes (two-sided binomial on discordants)."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    disc_b = int(np.sum((a == 1) & (b == 0)))
    disc_c = int(np.sum((a == 0) & (b == 1)))
    nd = disc_b + disc_c
    if nd == 0:
        return McNemarResult(disc_b, disc_c, 1.0, 0.0)
    p = float(_sps.binomtest(disc_b, nd, 0.5, alternative="two-sided").pvalue)
    stat = (abs(disc_b - disc_c) - 1) ** 2 / nd  # continuity-corrected chi-square
    return McNemarResult(disc_b, disc_c, p, stat)


def exact_paired_permutation(
    a: Sequence[float], b: Sequence[float], B: int = DEFAULT_BOOTSTRAP_B, seed: int = 0
) -> float:
    """Two-sided paired sign-flip permutation p-value for mean(a-b) (continuous ok)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    n = d.size
    if n == 0:
        return 1.0
    obs = abs(d.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(B, n))
    stats_ = np.abs((signs * d).mean(axis=1))
    return float((np.sum(stats_ >= obs - 1e-12) + 1) / (B + 1))


# --- Holm-Bonferroni -------------------------------------------------------------
@dataclass
class HolmResult:
    name: str
    p_value: float
    reject: bool
    adjusted_threshold: float


def holm_bonferroni(pvalues: dict[str, float], alpha: float = 0.05) -> list[HolmResult]:
    """Holm-Bonferroni step-down over a family of tests.

    Once a test fails to reject, all higher-p tests also fail (step-down rule).
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: list[HolmResult] = []
    still_rejecting = True
    for i, (name, p) in enumerate(items):
        thr = alpha / (m - i)
        rej = still_rejecting and (p <= thr)
        if not rej:
            still_rejecting = False
        out.append(HolmResult(name, p, rej, thr))
    return out


# --- OR groups -------------------------------------------------------------------
def or_group(member_pass: dict[str, bool]) -> bool:
    """An OR-group passes iff at least one member test passes (spec 'OR groups')."""
    return any(member_pass.values())


# --- seed summary ----------------------------------------------------------------
@dataclass
class SeedSummary:
    metric: str
    seed_count: int
    mean: Optional[float]
    std: Optional[float]
    min: Optional[float]
    median: Optional[float]
    max: Optional[float]
    seed_mean_ci_low: Optional[float] = None
    seed_mean_ci_high: Optional[float] = None


def seed_summary(metric: str, values: Sequence[float], confidence: float = 0.95) -> SeedSummary:
    vals = np.asarray([v for v in values if v is not None], dtype=float)
    if vals.size == 0:
        return SeedSummary(metric, 0, None, None, None, None, None)
    mean = float(vals.mean())
    std = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    lo = hi = None
    if vals.size > 1:
        se = std / np.sqrt(vals.size)
        t = float(_sps.t.ppf(1 - (1 - confidence) / 2, df=vals.size - 1))
        lo, hi = mean - t * se, mean + t * se
    return SeedSummary(
        metric, vals.size, mean, std, float(vals.min()), float(np.median(vals)),
        float(vals.max()), lo, hi,
    )


# --- gating ----------------------------------------------------------------------
@dataclass
class GateResult:
    metric: str
    bound: str  # "lower" | "upper"
    threshold: float
    observed: Optional[float]  # the relevant CI bound
    n: int
    min_n: Optional[int]
    status: str  # "pass" | "fail" | NOT_EVALUABLE
    detail: dict = field(default_factory=dict)


def evaluable(n: int, min_n: Optional[int]) -> bool:
    return min_n is None or n >= min_n


def wilson_gate(
    k: int, n: int, *, bound: str, threshold: float, min_n: Optional[int], name: str,
    confidence: float = 0.95,
) -> GateResult:
    """Gate a rate on a Wilson CI bound with min_N gating.

    bound="lower": pass iff CI lower bound >= threshold.
    bound="upper": pass iff CI upper bound <= threshold.
    """
    if not evaluable(n, min_n):
        return GateResult(name, bound, threshold, None, n, min_n, NOT_EVALUABLE)
    w = wilson_ci(k, n, confidence)
    observed = w.low if bound == "lower" else w.high
    if observed is None:
        return GateResult(name, bound, threshold, None, n, min_n, NOT_EVALUABLE)
    if bound == "lower":
        status = "pass" if observed >= threshold else "fail"
    else:
        status = "pass" if observed <= threshold else "fail"
    return GateResult(name, bound, threshold, observed, n, min_n, status,
                      {"point": w.point, "ci_low": w.low, "ci_high": w.high})
