"""GRPO reward shaping + group-relative advantage (docs/grpo/01_reward.md).

The scalar the GRPO policy optimizes is the SHIPPED behavioral-fidelity ruler
(:mod:`eval.behavioral_fidelity`, batched via :func:`eval.fast_reward.score_batch`) wrapped with

  * a collapse penalty (the dominant-code false positive — Doc 01 §2/§5), and
  * the canonical "refusal / malformed on a supported row => reward 0" rule
    (:func:`eval.oracle_at_n.score_row_samples`), short-circuited BEFORE any decode, and
  * the "None fidelity (spec asserts no measurable axis) => EXCLUDED from the group" rule
    (matches :func:`eval.behavioral_fidelity.summarize_fidelity` / ``_measurable``),

then turned into a group-relative advantage ``A_i = (r_i - mean)/(std + eps_adv)`` over the G
samples of one prompt — the GRPO point (no value net; the group mean is the baseline).

**No target-LUT leakage (Invariant 2):** the reward scores against the REQUESTED spec only
(``ground_truth_attribute_spec_text(row)``); ``target_codes`` is never passed to ``score_batch`` here.
``delta_e_weight`` is accepted for interface symmetry but deliberately NOT applied — decoded ΔE needs
the target LUT and stays eval-only / veto-only (Invariant 8).

Pure numpy over ``score_batch`` records — no torch, no model — so it imports and unit-tests without a
GPU (same discipline as :func:`eval.oracle_at_n.oracle_and_best`).
"""

from __future__ import annotations

import numpy as np

from eval.fast_reward import score_batch

TOKEN_COUNT = 64


def shaped_rewards(codes_batch, spec_text, *, device=None, collapse_penalty: float = 0.25,
                   delta_e_weight: float = 0.0) -> list[tuple[float | None, dict]]:
    """Shaped reward for each of the G rollouts of ONE prompt, in input order.

    ``codes_batch`` is a list of ``list[int] | None`` (a refusal is ``None``; a truncated/over-long
    completion is a non-64 list). ``spec_text`` is the canonical requested spec
    (``ground_truth_attribute_spec_text(row)``) — NEVER the conditioning text and NEVER a target LUT.

    Returns one ``(reward | None, record)`` per input sample:
      * refusal / non-64  -> ``(0.0, {"behavioral_fidelity": None, "collapsed": True, "refused": True})``
        WITHOUT touching the decoder (the valid-64 partition below IS the short-circuit);
      * a valid-64 sample whose spec asserts no measurable axis (fidelity ``None``) ->
        ``(None, record)`` so the caller drops it from the group (no reward/advantage/gradient);
      * otherwise ``r = max(0, fidelity - collapse_penalty * collapsed)``.

    The base ``behavioral_fidelity`` equals :func:`eval.behavioral_fidelity.score_generation` on the
    same codes within the ruler's tolerance (``|Δ| <= 0.02``, identical ``collapsed`` flags) — the
    parity contract in ``tests/test_fast_reward.py``.
    """
    # Partition BEFORE decode: a refusal/malformed sample must never reach score_batch's decoder.
    valid_idx = [i for i, c in enumerate(codes_batch)
                 if c is not None and len(c) == TOKEN_COUNT]
    recs_valid = (score_batch([codes_batch[i] for i in valid_idx], spec_text, device=device)  # NO target_codes
                  if valid_idx else [])
    rec_by_idx = dict(zip(valid_idx, recs_valid))

    out: list[tuple[float | None, dict]] = []
    for i in range(len(codes_batch)):
        rec = rec_by_idx.get(i)
        if rec is None:                              # refusal / malformed -> reward 0, never decoded
            out.append((0.0, {"behavioral_fidelity": None, "collapsed": True, "refused": True}))
            continue
        f = rec.get("behavioral_fidelity")
        if f is None:                                # non-grade / axis-less -> exclude from the group
            out.append((None, rec))
            continue
        r = float(f) - float(collapse_penalty) * float(bool(rec.get("collapsed")))
        # delta_e_weight is intentionally NOT applied (leakage-adjacent; eval-only per Invariant 8).
        out.append((max(0.0, r), rec))
    return out


def group_advantages(rewards, *, eps: float = 1e-4) -> list[float | None]:
    """Group-standardized advantage ``A_i = (r_i - mean)/(std + eps)`` over one prompt's G samples.

    ``rewards`` is the per-sample ``reward | None`` list from :func:`shaped_rewards` (``None`` =
    excluded). Group mean/std are computed over the MEASURABLE (non-``None``) rewards only. Excluded
    samples map to ``None`` (no advantage, no gradient). ``std == 0`` (all G rewards identical — e.g.
    all refused) => every advantage is ``0`` (no learning signal from that prompt), guarded by ``eps``
    against divide-by-zero. Population std (``ddof=0``) to match Doc 01 §5.
    """
    measurable = [float(r) for r in rewards if r is not None]
    if not measurable:
        return [None for _ in rewards]
    mu = float(np.mean(measurable))
    sigma = float(np.std(measurable))              # population std (ddof=0)
    denom = sigma + float(eps)
    return [None if r is None else (float(r) - mu) / denom for r in rewards]
