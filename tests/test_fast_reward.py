"""Parity gate for the batched fast-reward path (eval.fast_reward) vs the canonical reward.

The fast path (batched device-aware decode + a reduced axis-subset measurement) MUST produce the
SAME numbers as :mod:`eval.behavioral_fidelity` — this reward defines the experiment's baselines, so
a fast-but-wrong reward is worse than the slow one. This machine is CPU-only (darwin), so parity is
validated on CPU; the GPU speedup is a Colab runtime property and is not asserted here.

Two tiers:
  * torch-free / weight-free: the vectorized batched trilinear equals scipy's
    ``RegularGridInterpolator``, and the reduced measurement equals ``measure_behavior`` on the
    fields the agreement/collapse logic reads. These run everywhere.
  * decode-path (needs the frozen VQ weights, gitignored -> skipped when absent): batched decode
    vs per-sample ``decode_codes``; and on >=30 REAL corpus rows, ``score_batch`` fidelity /
    collapse / reranker order vs canonical ``score_generation``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest

from data_pipeline.attribute_spec import from_measured_behavior, ground_truth_attribute_spec_text
from data_pipeline.behavior_vector import measure_behavior
from data_pipeline.lut_ops import apply_lut_trilinear
from eval import cube_io
from eval.behavioral_fidelity import behavioral_agreement, rerank_key, score_from_lut
from eval.fast_reward import _apply_lut_trilinear_batch, measure_reduced_batch

_WEIGHTS = Path("tokenizer/final/model.pt").is_file()
_ROWS_PATH = Path("data/active_sft/active_rows.jsonl")


# --- helpers ----------------------------------------------------------------------
def _synthetic_luts(n: int = 6) -> np.ndarray:
    """A spread of clearly-non-trivial absolute LUTs (warm/cool/dark/bright/desat...)."""
    rng = np.random.default_rng(0)
    base = cube_io.identity_grid(17)
    luts = []
    for _ in range(n):
        lut = base.copy()
        lut[..., 0] = np.clip(lut[..., 0] + rng.uniform(-0.1, 0.1), 0, 1)
        lut[..., 1] = np.clip(lut[..., 1] + rng.uniform(-0.1, 0.1), 0, 1)
        lut[..., 2] = np.clip(lut[..., 2] + rng.uniform(-0.12, 0.12), 0, 1)
        lut = np.clip(lut + rng.uniform(-0.05, 0.05), 0, 1)
        luts.append(lut)
    return np.stack(luts)


def _load_score_rows(limit: int = 40) -> list[dict]:
    rows = [json.loads(line) for line in _ROWS_PATH.read_text().splitlines() if line.strip()]
    return [r for r in rows
            if r.get("is_supported") and r.get("target_tokens") and r.get("measured_behavior")][:limit]


# --- tier 1: torch-free numeric parity (runs everywhere) --------------------------
def test_batched_trilinear_matches_scipy_regulargrid():
    """The vectorized batch apply equals scipy's RegularGridInterpolator (float64 multilinear)."""
    from data_pipeline.behavior_vector import _clip_probe, _color_chart, _neutral_ramp

    luts = _synthetic_luts(5)
    for probe in (_neutral_ramp(), _color_chart(), _clip_probe()):
        ref = np.stack([apply_lut_trilinear(luts[b], probe) for b in range(luts.shape[0])])
        got = _apply_lut_trilinear_batch(luts, probe)
        assert np.max(np.abs(ref - got)) == 0.0


def test_reduced_measurement_matches_measure_behavior_on_asserted_axes():
    """The reduced measurement equals measure_behavior for every field agreement/collapse reads."""
    luts = _synthetic_luts(6)
    fast = measure_reduced_batch(luts)
    for b in range(luts.shape[0]):
        full = measure_behavior(luts[b])
        red = fast[b]
        for fld, v in red.items():
            if fld == "per_hue_saturation":
                for sector, sv in v.items():
                    assert sv == pytest.approx(full["per_hue_saturation"][sector], abs=1e-12)
            else:
                assert v == pytest.approx(full[fld], abs=1e-12), fld


def test_reduced_measurement_yields_identical_agreement():
    """behavioral_agreement is identical under the reduced mb vs the full mb, for a real spec."""
    luts = _synthetic_luts(6)
    fast = measure_reduced_batch(luts)
    for b in range(luts.shape[0]):
        full = measure_behavior(luts[b])
        spec = from_measured_behavior(full)
        a_full = behavioral_agreement(spec, full)
        a_red = behavioral_agreement(spec, fast[b])
        assert a_full["fidelity"] == a_red["fidelity"]
        assert a_full["axes_total"] == a_red["axes_total"]
        assert a_full["axes_backed"] == a_red["axes_backed"]


# --- tier 2: decode + end-to-end parity (needs the frozen VQ weights) -------------
@pytest.mark.skipif(not _WEIGHTS, reason="frozen VQ weights absent (staged-corpus only)")
def test_decode_batch_matches_per_sample_decode_codes():
    pytest.importorskip("torch")
    from eval.behavioral_fidelity import decode_codes
    from eval.fast_reward import decode_batch

    rows = _load_score_rows(12)
    assert rows, "no supported rows with target_tokens found"
    codes_batch = np.array([r["target_tokens"] for r in rows], dtype=np.int64)
    ref = np.stack([decode_codes(r["target_tokens"]) for r in rows])
    got = decode_batch(codes_batch)                       # device=None -> CPU
    assert got.shape == ref.shape
    assert np.max(np.abs(got - ref)) <= 1e-5


@pytest.mark.skipif(not _WEIGHTS or not _ROWS_PATH.is_file(),
                    reason="frozen VQ weights or active_rows corpus absent (staged-corpus only)")
def test_score_batch_fidelity_parity_on_real_rows():
    """>=30 real corpus rows: |Δfidelity| <= 0.02 and identical collapsed flags vs canonical."""
    pytest.importorskip("torch")
    from eval.behavioral_fidelity import score_generation
    from eval.fast_reward import score_batch

    rows = _load_score_rows(40)
    assert len(rows) >= 30, f"need >=30 scoreable rows, got {len(rows)}"
    max_dfid = 0.0
    for r in rows:
        spec = ground_truth_attribute_spec_text(r)
        can = score_generation(r["target_tokens"], spec, target_codes=r["target_tokens"])
        fast = score_batch(r["target_tokens"], spec, target_codes=r["target_tokens"])[0]
        cf, ff = can["behavioral_fidelity"], fast["behavioral_fidelity"]
        if cf is None or ff is None:
            assert cf is ff
        else:
            max_dfid = max(max_dfid, abs(cf - ff))
            assert abs(cf - ff) <= 0.02, f"row {r.get('id')}: |Δfidelity|={abs(cf - ff)}"
        assert bool(can["collapsed"]) == bool(fast["collapsed"]), f"row {r.get('id')} collapsed flag"
    assert max_dfid <= 0.02


@pytest.mark.skipif(not _WEIGHTS or not _ROWS_PATH.is_file(),
                    reason="frozen VQ weights or active_rows corpus absent (staged-corpus only)")
def test_score_batch_reranker_agreement_on_real_rows():
    """Harvest/RL must pick the SAME winner: argmax rerank_key agrees for >=95% of rows."""
    pytest.importorskip("torch")
    from eval.behavioral_fidelity import score_generation
    from eval.fast_reward import score_batch

    rng = random.Random(0)

    def perturb(codes: list[int], k: int) -> list[int]:
        c = list(codes)
        for _ in range(k):
            c[rng.randrange(64)] = rng.randrange(256)
        return c

    rows = _load_score_rows(40)
    assert len(rows) >= 30
    agree = 0
    for r in rows:
        tgt = r["target_tokens"]
        cands = [tgt, perturb(tgt, 4), perturb(tgt, 12), perturb(tgt, 40),
                 [tgt[0]] * 64, perturb(tgt, 64)]
        spec = ground_truth_attribute_spec_text(r)
        can = [score_generation(c, spec, target_codes=tgt) for c in cands]
        fast = score_batch(cands, spec, target_codes=tgt)
        can_arg = max(range(len(cands)), key=lambda i: rerank_key(can[i]))
        fast_arg = max(range(len(cands)), key=lambda i: rerank_key(fast[i]))
        agree += int(can_arg == fast_arg)
    assert agree / len(rows) >= 0.95, f"reranker agreement {agree}/{len(rows)}"


# --- tier 2: GRPO shaped-reward parity vs the canonical ruler (needs the frozen VQ weights) --------
@pytest.mark.skipif(not _WEIGHTS or not _ROWS_PATH.is_file(),
                    reason="frozen VQ weights or active_rows corpus absent (staged-corpus only)")
def test_grpo_shaped_reward_parity_with_score_generation():
    """The GRPO training reward equals `score_generation` on the same codes (guards a shaping bug that
    silently changes the objective): |Δ base fidelity| <= 0.02 and identical `collapsed` flags.

    (docs/grpo/01_reward.md §8 / IMPLEMENTATION_PROMPT §5 A. The refusal / None-exclusion / collapse-
    penalty accounting is unit-tested off-GPU in ``tests/test_grpo_reward.py``.)"""
    pytest.importorskip("torch")
    from eval.behavioral_fidelity import score_generation
    from eval.grpo_reward import shaped_rewards

    rows = _load_score_rows(20)
    assert len(rows) >= 8, f"need >= 8 scoreable rows, got {len(rows)}"
    max_dfid = 0.0
    for r in rows:
        spec = ground_truth_attribute_spec_text(r)
        can = score_generation(r["target_tokens"], spec)      # NO target_codes (training-reward parity)
        reward, rec = shaped_rewards([r["target_tokens"]], spec, collapse_penalty=0.0)[0]  # base reward
        cf = can["behavioral_fidelity"]
        if cf is None:                                        # axis-less spec -> excluded both sides
            assert reward is None and rec.get("behavioral_fidelity") is None
            continue
        assert reward == pytest.approx(max(0.0, rec["behavioral_fidelity"]))   # base reward == fidelity
        max_dfid = max(max_dfid, abs(rec["behavioral_fidelity"] - cf))
        assert abs(rec["behavioral_fidelity"] - cf) <= 0.02, f"row {r.get('id')}: |Δ|={abs(rec['behavioral_fidelity'] - cf)}"
        assert bool(rec["collapsed"]) == bool(can["collapsed"]), f"row {r.get('id')} collapsed flag"
    assert max_dfid <= 0.02
