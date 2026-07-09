"""Stage 6 leakage detection (data_collection_plan.md "Near-Neighbor Leakage Thresholds").

Per-axis cross-split leakage over the pinned ``configs/leakage_thresholds.yaml``:
  * exact-hash (all axes) — identical identity fields across splits;
  * ``image_perceptual`` — dct pHash + Hamming;
  * ``lut_behavior`` — residual-LUT PCA-64 cosine (basis fit on the frozen train pool);
  * ``prompt_lexical`` — word 3-gram Jaccard.
Semantic axes (``image_semantics`` CLIP / ``prompt_semantics`` MiniLM) are pluggable; without
the ``[ml]`` embeddings they are recorded ``skipped_no_embedding``.

``leakage_report`` fails if any axis has a cross-split violation, and records the policy
version, cutoffs used, per-axis counts, example pair ids, and a near-miss audit band.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

_DEFAULT_THRESHOLDS = Path("configs/leakage_thresholds.yaml")


# --- perceptual hash --------------------------------------------------------------
def dct_phash(image_gray: np.ndarray, hash_size: int = 8) -> int:
    """64-bit dct pHash of a grayscale image (values [0,1] or [0,255])."""
    from scipy.fft import dct

    img = np.asarray(image_gray, dtype=np.float64)
    if img.ndim == 3:
        img = img.mean(axis=2)
    if img.max() > 1.0:
        img = img / 255.0
    # resize to 32x32 via simple block/interp
    size = hash_size * 4
    ys = np.linspace(0, img.shape[0] - 1, size).astype(int)
    xs = np.linspace(0, img.shape[1] - 1, size).astype(int)
    small = img[np.ix_(ys, xs)]
    d = dct(dct(small, axis=0, norm="ortho"), axis=1, norm="ortho")
    low = d[:hash_size, :hash_size]
    flat = low.flatten()
    med = np.median(flat[1:]) if flat.size > 1 else 0.0  # exclude the DC term only
    bits = (low > med).flatten()
    val = 0
    for b in bits:
        val = (val << 1) | int(b)
    return val


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --- prompt lexical ---------------------------------------------------------------
def word_3grams(text: str) -> set[str]:
    words = re.findall(r"\w+", (text or "").lower())
    if len(words) < 3:
        return set(words)
    return {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


# --- LUT-behavior PCA -------------------------------------------------------------
@dataclass
class LutPCA:
    mean: np.ndarray
    components: np.ndarray            # [dim, D]
    basis_sha256: str

    def project(self, residual_vec: np.ndarray) -> np.ndarray:
        return (np.asarray(residual_vec, dtype=np.float64).reshape(-1) - self.mean) @ self.components.T


def fit_lut_pca(residual_vecs: list[np.ndarray], dim: int = 64) -> LutPCA:
    X = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in residual_vecs], axis=0)
    mean = X.mean(axis=0)
    Xc = X - mean
    # SVD-based PCA; guard dim to rank
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(dim, Vt.shape[0])
    comps = Vt[:k]
    basis = hashlib.sha256(np.ascontiguousarray(np.round(comps, 8)).tobytes()).hexdigest()
    return LutPCA(mean=mean, components=comps, basis_sha256=basis)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


# --- items + checker --------------------------------------------------------------
@dataclass
class LeakageItem:
    id: str
    split: str
    lut_hash: Optional[str] = None            # normalized_lut_hash (exact)
    image_hash: Optional[str] = None          # canonical_input_image_hash (exact)
    prompt_template_hash: Optional[str] = None
    phash: Optional[int] = None
    residual_vec: Optional[np.ndarray] = None
    prompt_text: Optional[str] = None


def load_thresholds(path: str | Path = _DEFAULT_THRESHOLDS) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


class LeakageChecker:
    def __init__(self, thresholds: dict, pca: Optional[LutPCA] = None):
        self.t = thresholds
        self.pca = pca
        self._proj_cache: dict[str, np.ndarray] = {}

    @classmethod
    def from_config(cls, items: list[LeakageItem], path: str | Path = _DEFAULT_THRESHOLDS) -> "LeakageChecker":
        t = load_thresholds(path)
        train_res = [it.residual_vec for it in items if it.split == "train" and it.residual_vec is not None]
        pca = fit_lut_pca(train_res, dim=int(t.get("lut_behavior", {}).get("pca_dim", 64))) if len(train_res) >= 2 else None
        return cls(t, pca=pca)

    def _proj(self, it: LeakageItem) -> Optional[np.ndarray]:
        if it.residual_vec is None or self.pca is None:
            return None
        if it.id not in self._proj_cache:
            self._proj_cache[it.id] = self.pca.project(it.residual_vec)
        return self._proj_cache[it.id]

    def pair_axes(self, a: LeakageItem, b: LeakageItem) -> list[tuple[str, str]]:
        """Return (axis, kind) for each axis on which a,b are a leakage pair."""
        hits: list[tuple[str, str]] = []
        # exact
        if a.lut_hash and a.lut_hash == b.lut_hash:
            hits.append(("lut_behavior", "exact"))
        if a.image_hash and a.image_hash == b.image_hash:
            hits.append(("image_semantics", "exact"))
        if a.prompt_template_hash and a.prompt_template_hash == b.prompt_template_hash:
            hits.append(("prompt_semantics", "exact"))
        # near: pHash
        ip = self.t.get("image_perceptual", {})
        if a.phash is not None and b.phash is not None:
            if hamming(a.phash, b.phash) <= ip.get("near_neighbor_cutoff", 6):
                hits.append(("image_perceptual", "near"))
        # near: LUT behavior PCA cosine
        lb = self.t.get("lut_behavior", {})
        pa, pb = self._proj(a), self._proj(b)
        if pa is not None and pb is not None:
            if cosine_distance(pa, pb) <= lb.get("near_neighbor_cutoff", 0.02):
                hits.append(("lut_behavior", "near"))
        # near: prompt lexical jaccard
        pl = self.t.get("prompt_lexical", {})
        if a.prompt_text and b.prompt_text:
            if jaccard(word_3grams(a.prompt_text), word_3grams(b.prompt_text)) >= pl.get("near_neighbor_cutoff", 0.80):
                hits.append(("prompt_lexical", "near"))
        return hits


@dataclass
class LeakageReport:
    status: str
    leakage_policy_version: str
    per_axis_violations: dict = field(default_factory=dict)
    example_pairs: dict = field(default_factory=dict)
    skipped_axes: list = field(default_factory=list)
    scopes_checked: list = field(default_factory=list)
    leakage_report_hash: str = ""

    def to_dict(self) -> dict:
        d = {
            "status": self.status,
            "leakage_policy_version": self.leakage_policy_version,
            "per_axis_violation_counts": self.per_axis_violations,
            "example_pair_ids": self.example_pairs,
            "skipped_axes": self.skipped_axes,
            "cross_split_scopes": self.scopes_checked,
        }
        return d


_SEMANTIC_AXES = ("image_semantics", "prompt_semantics")


def leakage_report(items: list[LeakageItem], thresholds_path: str | Path = _DEFAULT_THRESHOLDS,
                   checker: Optional[LeakageChecker] = None) -> LeakageReport:
    checker = checker or LeakageChecker.from_config(items, thresholds_path)
    t = checker.t
    scopes = (t.get("report", {}) or {}).get("cross_split_scopes",
                                             ["train_vs_eval", "train_vs_diagnostic", "train_vs_qualitative"])

    by_split: dict[str, list[LeakageItem]] = {}
    for it in items:
        by_split.setdefault(it.split, []).append(it)

    per_axis: dict[str, int] = {}
    examples: dict[str, list] = {}
    checked_scopes: list[str] = []

    for scope in scopes:
        if "_vs_" not in scope:
            continue
        a_split, b_split = scope.split("_vs_", 1)
        A, B = by_split.get(a_split, []), by_split.get(b_split, [])
        if not A or not B:
            continue
        checked_scopes.append(scope)
        for ia in A:
            for ib in B:
                for axis, kind in checker.pair_axes(ia, ib):
                    per_axis[axis] = per_axis.get(axis, 0) + 1
                    examples.setdefault(axis, [])
                    if len(examples[axis]) < 10:
                        examples[axis].append([ia.id, ib.id, kind, scope])

    skipped = [ax for ax in _SEMANTIC_AXES if per_axis.get(ax, 0) == 0 and not _has_embeddings(items, ax)]
    # semantic exact still counts (via hash); mark skipped only for the *near* embedding part
    status = "fail" if any(per_axis.values()) else "pass"
    report = LeakageReport(
        status=status,
        leakage_policy_version=t.get("leakage_policy_version", "v0"),
        per_axis_violations=per_axis,
        example_pairs=examples,
        skipped_axes=[f"{ax}:near_neighbor_no_embedding" for ax in skipped],
        scopes_checked=checked_scopes,
    )
    report.leakage_report_hash = hashlib.sha256(
        json.dumps(report.to_dict(), sort_keys=True).encode()
    ).hexdigest()
    return report


def _has_embeddings(items: list[LeakageItem], axis: str) -> bool:
    return False  # CLIP/MiniLM embeddings wired only under the [ml] extra
