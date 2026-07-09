"""Tokenizer reconstruction diagnostics + acceptance gate (NumPy, authoritative ΔE).

Uses the authoritative :mod:`eval.color_pipeline` CIEDE2000 (not the torch port) so gate
decisions match the eval harness. Thresholds are from training_plan_colab.md "Stage 1"
and model_architecture.md "LUT Tokenizer".

Nothing here touches the model directly except :func:`reconstruct`, which runs a trained
VQVAE in eval mode over a set of residuals (torch imported lazily inside it).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from eval.color_pipeline import ciede2000, srgb_to_lab_d65
from eval.cube_io import identity_grid


# --- acceptance thresholds (pinned) -------------------------------------------------
@dataclass(frozen=True)
class GateThresholds:
    mean_deltae: float = 2.0
    p95_deltae: float = 4.0
    p99_deltae: float = 6.0
    max_deltae: float = 10.0          # or reviewed exception (flagged, not auto-fail)
    mean_psnr: float = 35.0
    p5_psnr: float = 30.0
    per_family_mean_deltae: float = 2.5
    per_family_p95_deltae: float = 5.0
    min_family_rows: int = 30
    active_code_frac_alert: float = 0.70   # alert (health), not a hard blocker
    perplexity_alert: float = 64.0
    # per-target SFT admission (Stage 8/9)
    admission_mean_deltae: float = 3.0
    admission_p95_deltae: float = 6.0


GATE = GateThresholds()


# --- per-LUT primitives -------------------------------------------------------------
def _absolute(residual: np.ndarray) -> np.ndarray:
    """residual [17,17,17,3] -> clamped absolute LUT in [0,1] (deterministic gamut clip)."""
    return np.clip(residual + identity_grid(residual.shape[0]), 0.0, 1.0)


def lut_deltae_nodes(target_res: np.ndarray, recon_res: np.ndarray) -> np.ndarray:
    """CIEDE2000 per LUT node -> flat array of node ΔE for one LUT."""
    ta = _absolute(target_res)
    ra = _absolute(recon_res)
    return ciede2000(srgb_to_lab_d65(ta), srgb_to_lab_d65(ra)).reshape(-1)


def lut_psnr(target_res: np.ndarray, recon_res: np.ndarray) -> float:
    """PSNR (dB) on the [0,1] absolute LUT; data range 1.0."""
    ta = _absolute(target_res)
    ra = _absolute(recon_res)
    mse = float(np.mean((ta - ra) ** 2))
    if mse <= 1e-20:
        return 120.0
    return float(10.0 * np.log10(1.0 / mse))


# --- dataset-level aggregation ------------------------------------------------------
def aggregate_reconstruction(
    targets: list[np.ndarray],
    recons: list[np.ndarray],
    families: list[str] | None = None,
    thr: GateThresholds = GATE,
) -> dict:
    """Per-LUT ΔE (mean over nodes) + PSNR, aggregated overall and per source family."""
    if len(targets) != len(recons):
        raise ValueError("targets/recons length mismatch")
    if not targets:
        raise ValueError("empty reconstruction set")

    per_lut_deltae = np.array([lut_deltae_nodes(t, r).mean() for t, r in zip(targets, recons)])
    per_lut_psnr = np.array([lut_psnr(t, r) for t, r in zip(targets, recons)])
    finite = bool(np.all(np.isfinite(per_lut_deltae)) and np.all(np.isfinite(per_lut_psnr)))

    overall = {
        "n_luts": len(targets),
        "mean_deltae": float(per_lut_deltae.mean()),
        "p95_deltae": float(np.percentile(per_lut_deltae, 95)),
        "p99_deltae": float(np.percentile(per_lut_deltae, 99)),
        "max_deltae": float(per_lut_deltae.max()),
        "mean_psnr": float(per_lut_psnr.mean()),
        "p5_psnr": float(np.percentile(per_lut_psnr, 5)),
        "finite": finite,
    }

    per_family: dict[str, dict] = {}
    if families is not None:
        fam = np.array(families)
        for f in sorted(set(families)):
            mask = fam == f
            if int(mask.sum()) < thr.min_family_rows:
                continue
            d = per_lut_deltae[mask]
            per_family[f] = {
                "n": int(mask.sum()),
                "mean_deltae": float(d.mean()),
                "p95_deltae": float(np.percentile(d, 95)),
            }

    return {"overall": overall, "per_family": per_family,
            "per_lut_deltae": per_lut_deltae, "per_lut_psnr": per_lut_psnr}


def codebook_stats(all_codes: np.ndarray, codebook_size: int) -> dict:
    """Codebook usage health from all emitted code ids (any shape, flattened)."""
    codes = np.asarray(all_codes).reshape(-1)
    counts = np.bincount(codes, minlength=codebook_size).astype(np.float64)
    total = counts.sum()
    probs = counts / total if total > 0 else counts
    nz = probs[probs > 0]
    perplexity = float(np.exp(-(nz * np.log(nz)).sum())) if nz.size else 0.0
    active = int((counts > 0).sum())
    return {
        "active_codes": active,
        "active_frac": active / codebook_size,
        "perplexity": perplexity,
        "top_code_share": float(counts.max() / total) if total > 0 else 0.0,
        "dead_code_count": int((counts == 0).sum()),
    }


# --- gate ---------------------------------------------------------------------------
def evaluate_gate(recon_agg: dict, cb_stats: dict, thr: GateThresholds = GATE) -> dict:
    """Return {'pass': bool, 'checks': {...}, 'alerts': [...]}.

    Hard checks: reconstruction mean/tail ΔE, PSNR, per-family, finite. Codebook
    active%/perplexity are alerts (tokenizer-health), not hard blockers on their own
    (training_plan_colab.md Stage 1).
    """
    o = recon_agg["overall"]
    checks = {
        "mean_deltae": o["mean_deltae"] <= thr.mean_deltae,
        "p95_deltae": o["p95_deltae"] <= thr.p95_deltae,
        "p99_deltae": o["p99_deltae"] <= thr.p99_deltae,
        "max_deltae": o["max_deltae"] <= thr.max_deltae,   # reviewed-exception handled by caller
        "mean_psnr": o["mean_psnr"] >= thr.mean_psnr,
        "p5_psnr": o["p5_psnr"] >= thr.p5_psnr,
        "finite": o["finite"],
    }
    for f, s in recon_agg["per_family"].items():
        checks[f"family[{f}].mean_deltae"] = s["mean_deltae"] <= thr.per_family_mean_deltae
        checks[f"family[{f}].p95_deltae"] = s["p95_deltae"] <= thr.per_family_p95_deltae

    alerts = []
    if cb_stats["active_frac"] < thr.active_code_frac_alert:
        alerts.append(f"active_codes {cb_stats['active_frac']:.1%} < {thr.active_code_frac_alert:.0%}")
    if cb_stats["perplexity"] < thr.perplexity_alert:
        alerts.append(f"perplexity {cb_stats['perplexity']:.1f} < {thr.perplexity_alert:.0f}")

    return {"pass": all(checks.values()), "checks": checks, "alerts": alerts}


# --- structural roundtrip contracts (training-quality-independent) ------------------
def roundtrip_contracts(model) -> dict:  # noqa: ANN001 (torch model)
    """Decode output is finite + in range; .cube serialize/parse round-trips exactly.

    These verify the token<->LUT plumbing regardless of reconstruction quality; the
    'nearly identical' requirement is the ΔE gate above.
    """
    from eval.cube_io import parse_cube, residual_to_absolute, serialize_cube

    checks: dict[str, bool] = {}
    # decode a fixed code sequence -> correct-shape, finite residual (plumbing only;
    # [0,1] range is a trained-quality/gate property, enforced by the export clamp).
    res = model.decode(list(range(model.cfg.token_count)))
    checks["decode_shape"] = res.shape == (model.cfg.grid, model.cfg.grid, model.cfg.grid, 3)
    checks["decode_finite"] = bool(np.all(np.isfinite(res)))
    # .cube serialize -> parse roundtrip is exact on the clamped absolute LUT
    clamped = np.clip(residual_to_absolute(res), 0.0, 1.0)
    parsed, _ = parse_cube(serialize_cube(clamped))
    checks["cube_roundtrip"] = bool(np.allclose(parsed, clamped, atol=1e-9))
    return {"pass": all(checks.values()), "checks": checks}


# --- helper: run a trained model over residuals (torch, eval mode) ------------------
def reconstruct(model, residuals: list[np.ndarray]):  # noqa: ANN001
    """Encode+decode each residual; return (recon_residuals, codes[N,64]). No grad."""
    import torch  # local import keeps this module importable without torch

    recons, codes = [], []
    with torch.no_grad():
        for r in residuals:
            c = model.encode(r)
            codes.append(c)
            recons.append(model.decode(c))
    return recons, np.asarray(codes, dtype=np.int64)
