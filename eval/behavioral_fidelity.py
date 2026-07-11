"""Free-running BEHAVIORAL-FIDELITY metric for the prompt->LUT generator.

Teacher-forced token accuracy (:mod:`sft.score_tokens`) is BLIND to the free-running
collapse: it feeds the model the gold code prefix at every step, so it only measures
one-step-ahead prediction and never sees the model commit to its own 64-code trajectory.
Free-running (greedy) the model sinks to a near-neutral code — decoding to a ~identity
residual (RMS ~0.001 vs real corpus LUTs ~0.08 median) that "does nothing" to the image.
That is textbook exposure bias, and only a metric that scores what the model actually
PRODUCES can catch it.

This module scores a *generated* 64-code sequence against the *requested* AttributeSpec:

  * decode codes -> residual -> absolute LUT (:func:`tokenizer.frozen.load_frozen_vqvae`
    + :func:`eval.cube_io.residual_to_absolute`) — the frozen decoder ships on disk under
    ``tokenizer/final/`` (``VQVAE.decode`` is the live path the notebooks already use; the
    ``eval.lut_decoder`` stub is a *separate*, unrelated decode-disabled interface);
  * re-measure ``behavior_v2`` from the decoded LUT
    (:func:`data_pipeline.behavior_vector.measure_behavior`);
  * score DIRECTION+MAGNITUDE agreement with the requested spec
    (:func:`data_pipeline.attribute_spec.is_backed`) — the fraction of asserted axes that
    move the right way by roughly the right amount. A collapsed output moves nothing, so
    every asserted axis is unbacked and fidelity -> 0;
  * flag the collapse directly (residual RMS vs the ``degenerate_identity`` floor, plus
    code-histogram entropy + dominant-code share);
  * report decoded ΔE00 vs the target's OWN decoded LUT when target codes are supplied.

Pure/CPU and dependency-light: only :func:`decode_codes` imports torch (lazily, via
``tokenizer.frozen``); the scoring/aggregation helpers use just numpy + the decoder-free
color machinery, so they import and unit-test cleanly without the ``sft`` extra or a GPU.
"""

from __future__ import annotations

import math

import numpy as np

from data_pipeline.attribute_spec import AttributeSpec, is_backed
from data_pipeline.attribute_spec import parse as parse_spec
from data_pipeline.behavior_vector import measure_behavior
from data_pipeline.lut_ops import residual_norm
from eval.color_pipeline import deltae2000_srgb
from eval.cube_io import absolute_to_residual, residual_to_absolute
from eval.refuse_taxonomy import ROUTE_GRADE

TOKEN_COUNT = 64
CODEBOOK_SIZE = 256

# Below this RMS the decoded LUT is *effectively the identity* — the exact
# ``degenerate_identity`` floor from eval.frontier_scoring._MIN_RESIDUAL_NORM.
DEGENERATE_RESIDUAL_NORM = 5e-4
# A looser "did ~nothing relative to a requested edit" floor: real corpus residuals are
# ~0.08 median (weak ones ~0.02, strong ~0.27), the neutral-code collapse is ~0.001. This
# sits an order of magnitude above the degenerate floor and ~1/8 of the median. Tunable.
COLLAPSE_RESIDUAL_NORM = 0.01
# Greedy exposure-bias collapse does not always land on the identity code: it can over-commit to a
# dominant NON-neutral code (e.g. one code filling 48/64 positions -> RMS ~0.05, not ~0.001), which
# the RMS floor alone misses. Real corpus rows use a dominant code ~0.1-0.2 of the time, so a single
# code owning >= half the 64 positions is degenerate. Tunable.
DOMINANT_SHARE_MAX = 0.5
DEFAULT_TOL = 1.0  # attribute_spec.is_backed default (Lab units)


def decode_codes(codes, *, final_dir: str | None = None) -> np.ndarray:
    """Decode 64 codebook indices to an ABSOLUTE LUT ``[17,17,17,3]`` via the frozen decoder.

    Lazily imports :func:`tokenizer.frozen.load_frozen_vqvae` (which imports torch and is
    ``lru_cache``d, so repeated calls reuse the loaded VQ-VAE). ``final_dir`` overrides the
    frozen-tokenizer directory (defaults to ``$SLM_ARTIFACT_ROOT/tokenizer/final`` else the
    repo-relative ``tokenizer/final``).
    """
    from tokenizer.frozen import load_frozen_vqvae  # lazy: imports torch

    model, _ = load_frozen_vqvae(final_dir)
    residual = model.decode([int(c) for c in codes])  # VQVAE.decode -> [17,17,17,3] residual
    # Clip to [0,1] to match the REAL apply path (both notebooks clip the absolute LUT); LUT
    # nodes are encoded-sRGB in [0,1] and out-of-gamut nodes otherwise NaN in srgb->Lab.
    return np.clip(residual_to_absolute(residual), 0.0, 1.0)


def code_histogram_stats(codes) -> dict:
    """Histogram diagnostics over a code sequence — low entropy / high dominant share = collapse."""
    arr = np.asarray([int(c) for c in codes], dtype=np.int64)
    n = int(arr.size)
    if n == 0:
        return {"n_codes": 0, "unique_codes": 0, "dominant_code": None,
                "dominant_share": 0.0, "entropy_bits": 0.0, "entropy_norm": 0.0}
    vals, counts = np.unique(arr, return_counts=True)
    p = counts / n
    entropy_bits = float(-(p * np.log2(p)).sum())
    max_bits = math.log2(n) if n > 1 else 1.0
    dom = int(counts.argmax())
    return {
        "n_codes": n,
        "unique_codes": int(vals.size),
        "dominant_code": int(vals[dom]),
        "dominant_share": float(counts[dom] / n),
        "entropy_bits": entropy_bits,
        "entropy_norm": float(entropy_bits / max_bits) if max_bits > 0 else 0.0,
    }


def behavioral_agreement(spec: AttributeSpec, mb: dict, *, tol: float = DEFAULT_TOL) -> dict:
    """Fraction of the spec's asserted axes that the measured behavior backs (sign + magnitude).

    Reuses :func:`data_pipeline.attribute_spec.is_backed` (the interpreter↔LUT backing rule):
    an axis is *backed* iff the re-measured behavior has the same sign and is within ``tol``
    of the requested value. Hue-angle axes (``*_hue_deg``) are not magnitude-backed by
    ``is_backed`` and are excluded from the count (mirrors its ``continue``). Returns the
    fidelity fraction in ``[0,1]`` (``None`` when the spec asserts no measurable axis).
    """
    ok, issues = is_backed(spec, mb, tol=tol)
    issue_fields = {iss.split(":", 1)[1] for iss in issues if ":" in iss}

    per_axis: dict[str, dict] = {}
    n_axes = n_backed = 0
    for fld, v in spec.axes.items():
        if fld.endswith("_hue_deg"):
            continue  # is_backed skips hue-angle backing (checked by proximity elsewhere)
        n_axes += 1
        backed = fld not in issue_fields
        n_backed += int(backed)
        per_axis[fld] = {"requested": float(v), "measured": float(mb.get(fld, 0.0) or 0.0),
                         "backed": backed}
    phs = mb.get("per_hue_saturation") or {}
    for sector, v in spec.sat.items():
        n_axes += 1
        backed = sector not in issue_fields
        n_backed += int(backed)
        per_axis[f"sat_{sector}"] = {"requested": float(v),
                                     "measured": float(phs.get(sector, 0.0) or 0.0),
                                     "backed": backed}

    fidelity = (n_backed / n_axes) if n_axes else None
    return {"fidelity": fidelity, "axes_total": n_axes, "axes_backed": n_backed,
            "all_backed": bool(ok), "issues": issues, "per_axis": per_axis}


def rerank_key(rec: dict) -> tuple:
    """Canonical best-of-N reranker order (docs/collapse_fix) — HIGHER tuple wins under ``max``.

    Primary: ``behavioral_fidelity``. Tie-breaks: not ``collapsed``; higher code ``entropy_norm``;
    lower decoded ΔE (``decoded_delta_e.mean``) — the ΔE term is only meaningful when a target was
    scored (eval); it defaults to a neutral 0.0 when the key is absent (deploy), so the pick never
    depends on a target LUT that doesn't exist at inference time.
    """
    cs = rec.get("code_stats") or {}
    de = rec.get("decoded_delta_e") or {}
    return (
        rec.get("behavioral_fidelity") or 0.0,
        0 if rec.get("collapsed") else 1,
        cs.get("entropy_norm", 0.0),
        -float(de.get("mean", 0.0)),   # lower ΔE -> higher key; 0.0 (neutral) when no target scored
    )


def decoded_delta_e(pred_lut: np.ndarray, target_lut: np.ndarray) -> dict:
    """Node-wise CIEDE2000 between two absolute LUTs (the LUT nodes are sRGB in [0,1])."""
    a = np.clip(np.asarray(pred_lut, dtype=np.float64), 0.0, 1.0).reshape(-1, 3)
    b = np.clip(np.asarray(target_lut, dtype=np.float64), 0.0, 1.0).reshape(-1, 3)
    dE = deltae2000_srgb(a, b)
    return {"mean": float(np.mean(dE)), "p95": float(np.percentile(dE, 95)), "max": float(np.max(dE))}


def score_from_lut(pred_lut: np.ndarray, spec, *, target_lut: np.ndarray | None = None,
                   codes=None, tol: float = DEFAULT_TOL,
                   collapse_floor: float = COLLAPSE_RESIDUAL_NORM,
                   dominant_share_max: float = DOMINANT_SHARE_MAX) -> dict:
    """Score an already-decoded absolute LUT against a requested spec (torch-free).

    ``spec`` is an ``attribute_spec_text`` string or a parsed :class:`AttributeSpec`. Returns
    a record with ``behavioral_fidelity`` (the headline, ``None`` for non-grade/axis-less
    specs), collapse flags, and optional code/ΔE diagnostics. A row is ``collapsed`` if it did
    ~nothing (residual below ``collapse_floor``) OR over-committed to a single dominant code
    (``dominant_share >= dominant_share_max``) — the greedy exposure-bias failure mode.
    """
    spec = parse_spec(spec) if isinstance(spec, str) else spec
    mb = measure_behavior(pred_lut)
    resid_rms = float(mb.get("residual_norm") or residual_norm(absolute_to_residual(pred_lut)))
    code_stats = code_histogram_stats(codes) if codes is not None else None
    dom_share = code_stats["dominant_share"] if code_stats else 0.0

    rec: dict = {
        "route": spec.route,
        "residual_norm": resid_rms,
        "degenerate_identity": resid_rms < DEGENERATE_RESIDUAL_NORM,
        "collapsed": (resid_rms < collapse_floor) or (dom_share >= dominant_share_max),
    }
    if spec.route == ROUTE_GRADE:
        agree = behavioral_agreement(spec, mb, tol=tol)
        rec["agreement"] = agree
        rec["behavioral_fidelity"] = agree["fidelity"]
    else:
        rec["behavioral_fidelity"] = None
    if code_stats is not None:
        rec["code_stats"] = code_stats
    if target_lut is not None:
        rec["decoded_delta_e"] = decoded_delta_e(pred_lut, target_lut)
    return rec


def score_generation(codes, spec, *, target_codes=None, final_dir: str | None = None,
                     tol: float = DEFAULT_TOL,
                     collapse_floor: float = COLLAPSE_RESIDUAL_NORM) -> dict:
    """Decode generated ``codes`` and score them against ``spec`` (the full free-running path).

    ``target_codes`` (the row's corpus ``target_tokens``) enables the decoded-ΔE column by
    decoding the target's own LUT. Requires the frozen decoder (imports torch).
    """
    pred_lut = decode_codes(codes, final_dir=final_dir)
    target_lut = decode_codes(target_codes, final_dir=final_dir) if target_codes is not None else None
    return score_from_lut(pred_lut, spec, target_lut=target_lut, codes=codes,
                          tol=tol, collapse_floor=collapse_floor)


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summarize_fidelity(records: list[dict]) -> dict:
    """Aggregate per-row records into a behavioral-fidelity summary (pure numpy).

    The headline ``behavioral_fidelity_mean`` is over grade rows that assert ≥1 axis;
    ``collapse_rate`` / ``degenerate_rate`` are over ALL scored rows.
    """
    if not records:
        return {"rows": 0}
    graded = [r for r in records if r.get("behavioral_fidelity") is not None]
    rms = [r["residual_norm"] for r in records if "residual_norm" in r]
    dE = [r["decoded_delta_e"]["mean"] for r in records if r.get("decoded_delta_e")]
    ent = [r["code_stats"]["entropy_norm"] for r in records if r.get("code_stats")]
    dom = [r["code_stats"]["dominant_share"] for r in records if r.get("code_stats")]
    return {
        "rows": len(records),
        "grade_rows": len(graded),
        "behavioral_fidelity_mean": _mean(r["behavioral_fidelity"] for r in graded),
        "collapse_rate": _mean(float(bool(r.get("collapsed"))) for r in records),
        "degenerate_rate": _mean(float(bool(r.get("degenerate_identity"))) for r in records),
        "residual_norm_mean": _mean(rms),
        "residual_norm_median": float(np.median(rms)) if rms else None,
        "decoded_delta_e_mean": _mean(dE),
        "code_entropy_norm_mean": _mean(ent),
        "dominant_share_mean": _mean(dom),
    }
