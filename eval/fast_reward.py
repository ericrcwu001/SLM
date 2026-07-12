"""Batched, device-aware FAST PATH for the behavioral-fidelity reward (opt-in, additive).

The canonical reward (:mod:`eval.behavioral_fidelity`) scores one generated 64-code sample at a
time: it decodes each sample through the frozen VQ decoder on the CPU and runs the full
``behavior_v2`` measurement (``data_pipeline.behavior_vector.measure_behavior``) in numpy/scipy.
Best-of-N harvesting and the RL loop score ``N=16`` samples per row, so the per-sample CPU decode
dominates and the Colab GPU sits idle.

This module keeps the *numbers* identical but does the work in bulk:

  * :func:`decode_batch` decodes ``B`` code sequences in ONE batched forward on ``device`` (CUDA on
    Colab, CPU locally). Conv3d / GroupNorm / trilinear-resize all act per-sample, so a batched
    decode is bit-identical to stacking :func:`eval.behavioral_fidelity.decode_codes` (verified in
    ``tests/test_fast_reward.py``: max abs diff 0.0 on CPU).
  * :func:`score_batch` scores the whole batch, computing ONLY the ``behavior_v2`` axes the spec
    actually asserts (the neutral-ramp + chromatic-chart probes), skipping the skin / clip /
    foldover / smoothness / neutral-drift probes that never feed ``behavioral_fidelity`` or the
    collapse flags. The retained axes are computed with a *vectorized* trilinear apply that is
    numerically identical to scipy's ``RegularGridInterpolator`` (both are float64 multilinear on
    the same grid; verified max abs diff 0.0), and the agreement / collapse / rerank logic is the
    SAME canonical code — so the fidelity, ``collapsed`` flags, and reranker ordering match.

Nothing here changes the canonical semantics: the canonical path stays the default everywhere and
is the validation oracle. This module is imported only when a caller opts in (``--fast-reward`` /
``fast=True``).
"""

from __future__ import annotations

import copy
from functools import lru_cache

import numpy as np

from data_pipeline.attribute_spec import parse as parse_spec
from data_pipeline.behavior_vector import (
    _HUE_SECTOR_CENTERS,
    _assign_hue_sector,
    _color_chart,
    _neutral_ramp,
    _pct,
)
from data_pipeline.lut_ops import residual_norm
from eval import color_pipeline as cp
from eval.behavioral_fidelity import (
    DEGENERATE_RESIDUAL_NORM,
    COLLAPSE_RESIDUAL_NORM,
    DEFAULT_TOL,
    DOMINANT_SHARE_MAX,
    behavioral_agreement,
    code_histogram_stats,
    decoded_delta_e,
)
from eval.cube_io import GRID_SIZE, absolute_to_residual, identity_grid, residual_to_absolute
from eval.refuse_taxonomy import ROUTE_GRADE


def _norm_device(device) -> str:
    """Normalize a device arg (None / str / torch.device) to a hashable cache key string."""
    if device is None:
        return "cpu"
    return str(device)


@lru_cache(maxsize=None)
def _decoder_on_device(final_dir: str | None, device_str: str):
    """Return a frozen VQVAE placed on ``device_str`` (cached per (dir, device)).

    For CPU we reuse the ``load_frozen_vqvae`` lru_cached instance directly and only ever READ from
    it (never mutate its device in place). For a non-CPU device we deep-copy that instance and move
    the COPY, so the shared CPU cache is left untouched (per the frozen-tokenizer contract).
    """
    from tokenizer.frozen import load_frozen_vqvae  # lazy: imports torch

    model, _ = load_frozen_vqvae(final_dir)
    if device_str == "cpu":
        return model
    import torch

    m = copy.deepcopy(model).to(torch.device(device_str))
    m.eval()
    return m


def _as_codes_batch(codes_batch) -> np.ndarray:
    """Coerce ``codes_batch`` to an int64 ``[B, TOKEN_COUNT]`` array (accepts a single ``[TOKEN]``)."""
    arr = np.asarray(codes_batch, dtype=np.int64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"expected [B, token_count] codes, got shape {arr.shape}")
    return arr


def decode_batch(codes_batch, *, device=None, final_dir: str | None = None) -> np.ndarray:
    """Decode ``B`` code sequences ``[B, 64]`` to absolute LUTs ``[B,17,17,17,3]`` in ONE forward.

    Mirrors :func:`eval.behavioral_fidelity.decode_codes` (frozen VQ decode -> residual_to_absolute
    -> clip to [0,1]) but batched on ``device`` (``None`` => CPU). Returns float64, bit-identical to
    stacking ``decode_codes`` per sample.
    """
    import torch

    from tokenizer.model import output_to_residual

    arr = _as_codes_batch(codes_batch)
    model = _decoder_on_device(final_dir, _norm_device(device))
    cfg = model.cfg
    if arr.shape[1] != cfg.token_count:
        raise ValueError(f"expected {cfg.token_count} codes per sample, got {arr.shape[1]}")
    if arr.size and (arr.min() < 0 or arr.max() >= cfg.codebook_size):
        raise ValueError("code ids must be in [0, codebook_size)")

    dev = next(model.parameters()).device
    with torch.no_grad():
        t = torch.as_tensor(arr, device=dev)                       # [B, 64]
        quant = model.vq.embed_codes(t, cfg.latent_grid)           # [B, D, 4,4,4]
        recon = model.decoder(quant)                               # [B, 3, 17,17,17]
        resid = output_to_residual(recon).detach().cpu().numpy().astype(np.float64)  # [B,17,17,17,3]
    # residual_to_absolute infers the identity from axis 0, which is the BATCH dim here -> pass the
    # canonical 17^3 identity explicitly so it broadcasts over the batch.
    return np.clip(residual_to_absolute(resid, identity_grid(GRID_SIZE)), 0.0, 1.0)


def _apply_lut_trilinear_batch(luts: np.ndarray, rgb: np.ndarray) -> np.ndarray:
    """Trilinear apply of a BATCH of LUTs ``[B,N,N,N,3]`` to fixed probe points ``rgb`` ``[P,3]``.

    Numerically identical to stacking :func:`data_pipeline.lut_ops.apply_lut_trilinear` per LUT:
    both are float64 multilinear interpolation on the canonical ``i/(N-1)`` grid with inputs clipped
    to [0,1] (so no out-of-bounds). The probe points and their corner weights are shared across the
    batch, so the eight-corner blend is one vectorized gather -> ``[B,P,3]``.
    """
    B, N = luts.shape[0], luts.shape[1]
    rgb = np.clip(np.asarray(rgb, dtype=np.float64), 0.0, 1.0)
    coord = rgb * (N - 1)                                    # [P,3] in node units
    i0 = np.clip(np.floor(coord).astype(np.int64), 0, N - 2)  # lower corner (keep last cell at x==1)
    frac = coord - i0                                        # [P,3] fractional offset
    out = np.zeros((B, rgb.shape[0], 3), dtype=np.float64)
    for dr in (0, 1):
        for dg in (0, 1):
            for db in (0, 1):
                w = ((frac[:, 0] if dr else 1.0 - frac[:, 0])
                     * (frac[:, 1] if dg else 1.0 - frac[:, 1])
                     * (frac[:, 2] if db else 1.0 - frac[:, 2]))          # [P]
                vals = luts[:, i0[:, 0] + dr, i0[:, 1] + dg, i0[:, 2] + db, :]  # [B,P,3]
                out += w[None, :, None] * vals
    return out


# Probe geometry is fixed, so compute the input-side Lab / masks / hue sectors once.
@lru_cache(maxsize=1)
def _ramp_probe():
    ramp = _neutral_ramp()
    r_before = cp.srgb_to_lab_d65(ramp)
    low = ramp[:, 0] <= 0.25
    high = ramp[:, 0] >= 0.75
    spread_before = _pct(r_before[:, 0], 95) - _pct(r_before[:, 0], 5)
    return ramp, r_before, low, high, spread_before


@lru_cache(maxsize=1)
def _chart_probe():
    chart = _color_chart()
    c_before = cp.srgb_to_lab_d65(chart)
    chroma_before = cp.chroma(c_before)
    chart_L = c_before[:, 0]
    hi = chart_L >= 66.0
    lo = chart_L <= 33.0
    sectors = _assign_hue_sector(cp.hue_deg(c_before))
    return chart, c_before, chroma_before, hi, lo, sectors


def _measure_reduced(lut_abs: np.ndarray, r_after: np.ndarray, c_after: np.ndarray) -> dict:
    """Compute ONLY the ``behavior_v2`` fields the agreement/collapse logic reads, for one LUT.

    ``r_after`` / ``c_after`` are the already-applied+Lab-converted neutral-ramp / chromatic-chart
    probes for this LUT. Every line below is copied verbatim from
    ``data_pipeline.behavior_vector.measure_behavior`` for the retained fields, so the values are
    bit-identical; the skin / clip / foldover / smoothness / neutral-drift probes and the
    ``*_hue_deg`` vector fields (which ``behavioral_agreement`` skips) are simply not computed.
    """
    _ramp, r_before, low, high, spread_before = _ramp_probe()
    _chart, c_before, chroma_before, hi, lo, sectors = _chart_probe()

    dL = r_after[:, 0] - r_before[:, 0]
    da = r_after[:, 1] - r_before[:, 1]
    db = r_after[:, 2] - r_before[:, 2]
    spread_after = _pct(r_after[:, 0], 95) - _pct(r_after[:, 0], 5)

    chroma_after = cp.chroma(c_after)
    ab_shift = c_after[:, 1:] - c_before[:, 1:]
    shadow_ab = ab_shift[lo].mean(axis=0) if lo.any() else np.zeros(2)
    highlight_ab = ab_shift[hi].mean(axis=0) if hi.any() else np.zeros(2)
    chroma_shift = chroma_after - chroma_before
    per_hue_saturation = {
        name: (float(chroma_shift[sectors == name].mean()) if (sectors == name).any() else 0.0)
        for name in _HUE_SECTOR_CENTERS
    }

    _black_point = float(np.mean(dL[low])) if low.any() else 0.0
    _contrast_spread = float(spread_after - spread_before)
    _chroma_d = float(np.mean(chroma_after - chroma_before))
    matte_strength = (max(0.0, _black_point)
                      + max(0.0, -_contrast_spread)
                      + 0.5 * max(0.0, -_chroma_d))

    residual = absolute_to_residual(lut_abs)
    return {
        "temperature_delta_b": float(np.mean(db)),
        "tint_delta_a": float(np.mean(da)),
        "mean_l_delta": float(np.mean(dL)),
        "contrast_l_spread_delta": float(spread_after - spread_before),
        "black_point_l_delta": float(np.mean(dL[low])) if low.any() else 0.0,
        "highlight_l_delta": float(np.mean((r_after[:, 0] - r_before[:, 0])[high])) if high.any() else 0.0,
        "shadow_l_delta": float(np.mean((r_after[:, 0] - r_before[:, 0])[low])) if low.any() else 0.0,
        "chroma_delta": float(np.mean(chroma_after - chroma_before)),
        "split_tone_strength": float(np.hypot(*shadow_ab) + np.hypot(*highlight_ab)),
        "per_hue_saturation": per_hue_saturation,
        "matte_strength": matte_strength,
        "residual_norm": residual_norm(residual),
    }


def measure_reduced_batch(luts: np.ndarray) -> list[dict]:
    """Reduced ``behavior_v2`` measurement over a batch of absolute LUTs ``[B,17,17,17,3]``.

    Batches the (expensive) trilinear apply + Lab conversion across the batch, then computes the
    per-sample reductions with the exact ``measure_behavior`` arithmetic. Returns ``B`` dicts, each
    carrying exactly the fields :func:`eval.behavioral_fidelity.behavioral_agreement` /
    ``score_from_lut`` read (the asserted-axis magnitudes + ``per_hue_saturation`` + ``residual_norm``).
    """
    luts = np.asarray(luts, dtype=np.float64)
    if luts.ndim == 4:
        luts = luts[None, ...]
    ramp, _rb, _lo, _hi, _sb = _ramp_probe()
    chart, _cb, _crb, _chi, _clo, _sec = _chart_probe()
    ramp_after = cp.srgb_to_lab_d65(_apply_lut_trilinear_batch(luts, ramp))     # [B,33,3]
    chart_after = cp.srgb_to_lab_d65(_apply_lut_trilinear_batch(luts, chart))   # [B,P,3]
    return [_measure_reduced(luts[b], ramp_after[b], chart_after[b]) for b in range(luts.shape[0])]


def score_batch(codes_batch, spec, *, device=None, target_codes=None,
                tol: float = DEFAULT_TOL, collapse_floor: float = COLLAPSE_RESIDUAL_NORM,
                dominant_share_max: float = DOMINANT_SHARE_MAX,
                final_dir: str | None = None) -> list[dict]:
    """Decode + score ``B`` code sequences against ``spec`` — the batched analogue of
    :func:`eval.behavioral_fidelity.score_generation` (per sample).

    Returns one record per sample with the SAME keys the reranker / harvest use — at least
    ``behavioral_fidelity``, ``collapsed``, ``residual_norm``, ``code_stats``, and ``agreement`` —
    so records are drop-in for :func:`eval.behavioral_fidelity.rerank_key`. ``target_codes`` (the
    row's ``target_tokens``) enables the ``decoded_delta_e`` column (decoded once, shared by the
    batch). The ``behavioral_fidelity`` / ``collapsed`` values equal the canonical path (validated
    in ``tests/test_fast_reward.py``).
    """
    spec = parse_spec(spec) if isinstance(spec, str) else spec
    codes_arr = _as_codes_batch(codes_batch)
    luts = decode_batch(codes_arr, device=device, final_dir=final_dir)          # [B,17,17,17,3]
    target_lut = None
    if target_codes is not None:
        target_lut = decode_batch(target_codes, device=device, final_dir=final_dir)[0]

    mbs = measure_reduced_batch(luts)
    records: list[dict] = []
    for b in range(codes_arr.shape[0]):
        mb = mbs[b]
        codes = codes_arr[b]
        resid_rms = float(mb["residual_norm"])
        code_stats = code_histogram_stats(codes)
        dom_share = code_stats["dominant_share"]

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
        rec["code_stats"] = code_stats
        if target_lut is not None:
            rec["decoded_delta_e"] = decoded_delta_e(luts[b], target_lut)
        records.append(rec)
    return records
