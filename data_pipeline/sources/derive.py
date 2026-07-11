"""Stage 4 derivation: raw asset -> canonical absolute LUT tensor.

Paths (ADR 0003, data_collection_plan.md "PPR10K/FiveK Plan"):
  * ``.cube`` / HaldCLUT PNG -> LUT tensor (direct decode; see :mod:`data_pipeline.lut_ops`).
  * PPR10K: parse expert XMP (allowlist global fields, hard-reject local/non-LUT tools),
    then fit a global LUT from the before->processed image pair (faithful + validates).
  * FiveK: fit a global LUT from source -> expert-retouched target.

The pair-fit is a scattered-data LUT estimator on the canonical grid: each source pixel is
assigned to its nearest node; a node's raw fitted output is the mean target color of pixels
landing in it. A single photo only touches a sliver of the 17^3 cube, so most nodes are
unobserved. Rather than snap those to identity -- which leaves cliffs between fitted and
identity nodes that read as (spurious) roughness/non-monotonicity to the quality gate -- the
residual is completed by a Laplacian-regularized smooth fill: observed nodes are pulled toward
their measured residual with weight proportional to support, and unobserved nodes are diffused
smoothly from their neighbours. This makes the gate measure the *edit's* representability
rather than the estimator's sparsity. The raw (identity-fallback) estimator is still available
via ``smooth=False`` for exact-recovery diagnostics. Fitting/evaluation are in encoded sRGB;
:mod:`data_pipeline.representability` computes held-out CIEDE2000 + spatial gates on the result.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from eval.cube_io import GRID_SIZE, identity_grid, parse_cube

from ..lut_ops import haldclut_to_lut, resample_lut

# --- decode -----------------------------------------------------------------------
def cube_bytes_to_lut(data: bytes | str, target_size: int | None = GRID_SIZE) -> np.ndarray:
    """Parse a ``.cube`` to an absolute LUT. ``target_size=None`` returns the NATIVE-size LUT
    (no resample) so callers can measure resample-aware smoothness / color-manage from native."""
    lut, _ = parse_cube(data)
    return lut if target_size is None else resample_lut(lut, target_size)


def haldclut_png_to_lut(png: np.ndarray, target_size: int | None = GRID_SIZE) -> np.ndarray:
    return haldclut_to_lut(png, target_size)


# --- XMP (PPR10K / Lightroom crs) -------------------------------------------------
# TRUE local-region editors (masks / brush / gradient / retouch): a global RGB LUT cannot represent
# these, so a pair carrying an ACTIVE one is hard-rejected (data_collection_plan.md "XMP hard-reject").
# Restricted to genuine local-region CONTAINERS. Global/default fields (Sharpness, LuminanceSmoothing,
# ColorNoiseReduction, Texture, Clarity, Dehaze, LensProfileEnable, AutoLateralCA,
# PostCropVignetteAmount) and geometry (Crop*, Perspective*) are deliberately NOT here: Lightroom
# writes them as defaults in ~every XMP, so substring-matching them tripped the reject on ~100% of
# PPR10K. Their residual is handled by the representability spatial gate / pair-fit magnitude gate.
_XMP_LOCAL_TOOL_MARKERS = (
    "crs:MaskGroup", "crs:Masks", "crs:PaintBasedCorrections", "crs:PaintCorrection",
    "crs:CircularGradientBasedCorrections", "crs:GradientBasedCorrections",
    "crs:RetouchAreas", "crs:RetouchInfo", "crs:RedEyeInfo", "crs:DustSpots",
)
# Accepted global tone/color fields (allowlist).
_XMP_GLOBAL_FIELDS = (
    "Temperature", "IncrementalTemperature", "Tint", "IncrementalTint",
    "Exposure2012", "Contrast2012", "Highlights2012", "Shadows2012",
    "Whites2012", "Blacks2012", "Saturation", "Vibrance",
)


@dataclass
class XmpResult:
    parse_status: str                       # "parsed" | "unparsed" | "unknown_schema"
    global_fields_present: list = field(default_factory=list)
    rejected_fields: list = field(default_factory=list)
    local_tool_count: int = 0
    values: dict = field(default_factory=dict)

    @property
    def accepted(self) -> bool:
        return self.parse_status == "parsed" and self.local_tool_count == 0


def _find_field(text: str, name: str) -> str | None:
    # crs:Name="value"  or  <crs:Name>value</crs:Name>
    m = re.search(rf'crs:{name}="([^"]*)"', text)
    if m:
        return m.group(1)
    m = re.search(rf"<crs:{name}>([^<]*)</crs:{name}>", text)
    return m.group(1) if m else None


def _local_edit_active(text: str, marker: str) -> bool:
    """A local-region editor counts only if its bag is actually POPULATED. An empty/self-closed
    placeholder (``<crs:Masks/>``, written by default) does not — matching the intended "active list
    bag" semantics and avoiding the false-positive where merely declaring the schema tripped a reject.
    """
    if marker not in text:
        return False
    return re.search(rf"{re.escape(marker)}\s*/>", text) is None   # skip empty self-closed tag


def parse_xmp(text: str) -> XmpResult:
    if not text or "crs:" not in text and "<x:xmpmeta" not in text:
        return XmpResult(parse_status="unknown_schema")
    local = 0
    rejected: list[str] = []
    for marker in _XMP_LOCAL_TOOL_MARKERS:
        if _local_edit_active(text, marker):
            local += 1
            rejected.append(marker.split(":", 1)[-1])
    values: dict[str, float] = {}
    present: list[str] = []
    for name in _XMP_GLOBAL_FIELDS:
        raw = _find_field(text, name)
        if raw is None:
            continue
        try:
            v = float(raw)
        except ValueError:
            continue
        present.append(name)
        values[name] = v
    return XmpResult(parse_status="parsed", global_fields_present=present,
                     rejected_fields=rejected, local_tool_count=local, values=values)


# --- pair fit ---------------------------------------------------------------------
@dataclass
class FitResult:
    lut_abs: np.ndarray
    support_counts: np.ndarray          # [N,N,N] pixel counts per node
    supported_mask: np.ndarray          # count >= min_support
    low_support_mask: np.ndarray        # 0 < count < min_support
    empty_mask: np.ndarray              # count == 0 (filled with identity)


# Laplacian smooth-fill hyperparameters. ``lambda`` sets how strongly a node trusts its own
# measured mean vs its neighbours: trust = count / (count + lambda), so a node needs ~lambda
# pixels to be believed half as much as the smooth neighbourhood. ``iters`` of Jacobi diffusion
# propagate observed nodes across the whole grid (17 nodes/axis -> a few hundred iters converge).
SMOOTH_FILL_LAMBDA = 8.0
SMOOTH_FILL_ITERS = 200


def _smooth_fill_residual(lut_abs: np.ndarray, counts: np.ndarray, size: int,
                          lam: float = SMOOTH_FILL_LAMBDA,
                          iters: int = SMOOTH_FILL_ITERS) -> np.ndarray:
    """Complete a scattered-node fit into a smooth global LUT (see module docstring).

    Minimizes ``sum_observed count*(r-r_meas)^2 + lam*||grad r||^2`` over the residual grid by
    Jacobi iteration: observed nodes are anchored to their measured residual with weight
    ``count/(count+lam)``; every node relaxes toward its 6-neighbour mean (Neumann boundary).
    Deterministic (fixed iteration count) so canonical hashes stay reproducible.
    """
    identity = identity_grid(size)
    data = lut_abs - identity                       # measured residual (0 at unobserved nodes)
    w = np.where(counts > 0, counts / (counts + lam), 0.0).astype(np.float64)[..., None]
    r = data.copy()
    for _ in range(iters):
        pad = np.pad(r, ((1, 1), (1, 1), (1, 1), (0, 0)), mode="edge")
        neighbour_mean = (
            pad[:-2, 1:-1, 1:-1] + pad[2:, 1:-1, 1:-1]
            + pad[1:-1, :-2, 1:-1] + pad[1:-1, 2:, 1:-1]
            + pad[1:-1, 1:-1, :-2] + pad[1:-1, 1:-1, 2:]
        ) / 6.0
        r = w * data + (1.0 - w) * neighbour_mean
    return np.clip(identity + r, 0.0, 1.0)


def fit_global_lut(
    source_rgb: np.ndarray,
    target_rgb: np.ndarray,
    size: int = GRID_SIZE,
    min_support: int = 32,
    smooth: bool = True,
) -> FitResult:
    """Fit a global absolute LUT mapping ``source_rgb`` -> ``target_rgb`` (encoded sRGB).

    With ``smooth=True`` (default) the sparse per-node means are completed by a
    Laplacian-regularized fill (:func:`_smooth_fill_residual`); with ``smooth=False`` unobserved
    nodes fall back to identity (the raw estimator, kept for exact-recovery diagnostics).
    ``support_counts``/masks always reflect the true observed pixel coverage.
    """
    src = np.clip(np.asarray(source_rgb, dtype=np.float64).reshape(-1, 3), 0.0, 1.0)
    tgt = np.clip(np.asarray(target_rgb, dtype=np.float64).reshape(-1, 3), 0.0, 1.0)
    if src.shape != tgt.shape:
        raise ValueError(f"source/target pixel count mismatch: {src.shape} vs {tgt.shape}")

    idx = np.rint(src * (size - 1)).astype(np.int64)  # nearest node per pixel [P,3]
    flat_node = (idx[:, 0] * size + idx[:, 1]) * size + idx[:, 2]

    n_nodes = size ** 3
    sums = np.zeros((n_nodes, 3), dtype=np.float64)
    counts = np.zeros(n_nodes, dtype=np.int64)
    np.add.at(sums, flat_node, tgt)
    np.add.at(counts, flat_node, 1)

    identity = identity_grid(size).reshape(n_nodes, 3)
    lut = identity.copy()
    nonzero = counts > 0
    lut[nonzero] = sums[nonzero] / counts[nonzero][:, None]

    lut = lut.reshape(size, size, size, 3)
    counts3 = counts.reshape(size, size, size)
    if smooth:
        lut = _smooth_fill_residual(lut, counts3, size)
    supported = counts3 >= min_support
    low = (counts3 > 0) & (counts3 < min_support)
    empty = counts3 == 0
    return FitResult(
        lut_abs=np.clip(lut, 0.0, 1.0),
        support_counts=counts3,
        supported_mask=supported,
        low_support_mask=low,
        empty_mask=empty,
    )
