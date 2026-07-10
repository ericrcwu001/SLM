# Two-Stage Prompt-to-LUT Migration — HANDOFF log

Branch: `feat/two-stage`. Executes the Deferred roadmap P1→P7 from
`~/.claude/plans/create-a-detailed-plan-wondrous-gem.md`, governed by ADRs 0020–0026 and
`docs/AUDIT_claude_codex_prompt_to_lut.md`.

Protocol: after each phase — `python3 -m pytest -q`, verify the phase's exit criteria, commit, append
a status entry here. GPU/A100 steps are Colab handoffs (see the STOP blocks); results are pasted back
by the human and recorded here before the next phase proceeds.

Legend: ✅ done · ⏸ waiting on Colab · ⛔ blocked by a failed gate.

---

## P1 — Eval honesty (ADR 0024) ✅  (local; no Colab)

**Goal.** Make the ruler honest before any retrain: unit-aware holdout, full stratified scoring with
CIs, exact-64 assertion, and scaffold the decoder-free OOD/refuse/interpreter eval slices.

**Changes**
- `sft/holdout.py` — holdout is now **unit-aware**: keys on `split_unit_id` (not the row id). Added
  `holdout_key(row)` and `is_holdout_row(row)`; kept the pure `is_holdout(key, frac)` predicate
  (back-compat with `test_bilevel_bridge_helpers`).
- `sft/example.py` — `supported_rows` uses `is_holdout_row` (fix of `example.py:62`). Added
  `surviving_code_positions()` + a cached `_code_token_ids()`; `build_supervised_example` now raises
  `SFTError` when a **supported** row's assistant span does not retain exactly its 64 target codes
  (partial-truncation guard — the trainer/scorer already skip+count rows that raise). Closes AUDIT F8.
- `sft/score_tokens.py` — default `--limit 0` = score the **full** unit-aware holdout. New pure
  `summarize_scores()` + `_group_bootstrap_ratio()` report overall micro token accuracy (the METRIC)
  with a **unit-clustered** bootstrap CI, macro per-family accuracy, and per-family breakdowns with
  their own CIs. Exact-64 defence-in-depth: rows whose surviving code positions != 64 are skipped
  (counted as `partial`), never scored. METRIC= sentinel contract unchanged (overall micro accuracy).
- `sft/bilevel_bridge.py` — `--score-limit` default 48 → **0** (full holdout; eval cost lever only,
  not a locked knob).
- `notebooks/sft_stage7_run.ipynb` — improve-loop `evalcell` now runs `--score-limit 0`.
- `eval/configs/gating_slice_registry.yaml` — appended the ADR-0024 decoder-free slices (SCAFFOLD,
  `frozen: false`, not ship-gated until their data is built): `eval_in_distribution_regression`,
  `eval_unseen_wording`, `eval_named_concept`, `eval_nonce_concept`, `eval_counterfactual_ranking`,
  `eval_paraphrase_consistency`, `eval_refuse_out_of_scope`, `eval_refuse_out_of_gamut`, and the
  interpreter metrics (`attribute_f1`, `route_accuracy`, `interpreter_over_refusal_rate`).
- Tests: `tests/test_sft_example.py` (7), `tests/test_score_tokens.py` (9) — GPU-free.

**Exit criteria — verified (ADR 0024)**
- Unit-aware holdout, 0 leakage: on the real corpus (production path `sft.example.supported_rows`)
  supported+materialized = 2761 → train 2641 / holdout 120; **0** split-units cross the boundary
  (was **47/131** units leaked under the old row-id carve; old holdout 169 rows → new 120).
- Full stratified scoring: default scores all held-out rows; per-family macro accuracy + unit-clustered
  group-bootstrap CIs emitted in `score_summary` (overall + 6 families present in the holdout).
- Exact-64 assertion present in both `sft/example.py` and `sft/score_tokens.py`; all 120 holdout rows
  carry exactly 64 target codes.
- OOD/refuse/interpreter slices declared in the registry (11 new metrics load via
  `eval.run_eval.load_gating_registry`).
- Locked knobs untouched: `configs/sft_default.yaml`, `sft/config.py`, `sft/train.py` not in the diff.
- `python3 -m pytest -q` → **310 passed** (was 294; +16 new).

**No Colab needed for P1.** The unit-aware holdout will drop the headline token accuracy when next
scored on the A100 (expected — it quantifies the prior 48.5% leakage inflation); that number is
recorded when P6 scores on Colab, and becomes the `eval_in_distribution_regression` baseline.

Commit: see `feat/two-stage` history (P1).

---
