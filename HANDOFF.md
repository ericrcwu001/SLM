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

## P2 — Refuse becomes load-bearing (ADR 0023) ✅  (local; no Colab)

**Goal.** Make the refuse path actually train (it was skipped every epoch) and extend the taxonomy to
the 3-way route `{grade, clarify, refuse}` with two refuse kinds `{out_of_scope, out_of_gamut}`,
keeping the five taxonomy files in exact sync.

**The load-bearing fix (AUDIT F2).** All 272 unsupported rows carried machine-local ABSOLUTE image
paths (`/Users/ericwu/.../luts/raw/...`) that never resolve on Colab, so `sft.train` skipped every
refusal row (`resolve_image` returns an absolute path unchanged → file missing → skip). Supported
rows use repo-relative paths (`luts/raw/...`) that resolve against `$SLM_ARTIFACT_ROOT`.

**Changes**
- `eval/refuse_taxonomy.py` — NEW single source of truth (pure/stdlib; lives in `eval`, the lower
  layer, so both `eval` and `data_pipeline` import it without a cycle): `ROUTES`, `REFUSE_KINDS`,
  `OUT_OF_SCOPE_CATEGORIES` (11), `OUT_OF_GAMUT_CATEGORIES` (3: infrared_false_color,
  pure_primary_cast, hue_rotation), `CLARIFY_CATEGORIES` (underspecified_intent), and helpers
  `route_for_category` / `refuse_kind_for_category` / `is_mixed_category`.
- `data_pipeline/unsupported_gen.py` — imports the taxonomy (`PURE_CATEGORIES = OUT_OF_SCOPE_*`);
  briefs + validator cues + teacher system prompts for out_of_gamut and clarify;
  `build_messages` dispatches per category kind; validator now covers all non-grade categories.
- `scripts/generate_unsupported.py` — rows now store a **portable relative** `image_path`
  (`to_portable_image_path`, anchored on `luts/`), plus `route` + `refuse_kind`; the balanced plan
  gained the 3 out_of_gamut buckets (20 total: 11 + 3 + 6 mixed).
- `scripts/migrate_unsupported_portable.py` — NEW idempotent, deterministic migration: rewrote the
  272 rows (+ `unsupported_rows.jsonl` 507, `unsupported_eval_rows.jsonl` 235) to relative paths +
  `route=refuse` + `refuse_kind=out_of_scope`; backed up each file to `*.bak_pre_portable_unsup`;
  bumped `active_manifest.json` `active_set_version` → `active_set_v2_portable_unsup` and recorded a
  `portable_unsupported_migration` block. Frozen `luts/`/images/tokenizer untouched (ADR 0026).
- Schemas: `SftRow` + `EvalRow` gained `route` / `refuse_kind` (backward-compatible defaults; added
  to `_EVALROW_FIELDS` so they round-trip).
- `eval/unsupported_metrics.py` — route-aware: `DecisionRecord` gains optional `route`/`refuse_kind`
  (derived from `is_supported` when unset → legacy scoring unchanged); NEW `out_of_scope_recall`,
  `out_of_gamut_recall`, `clarify_over_refusal_rate`; clarify excluded from the binary boundary
  metrics; `run_eval.py` passes the fields through.
- `eval/fixtures/make_smoke_rows.py` — added 6 out_of_gamut + 4 clarify smoke rows (50 supported /
  26 refuse / 4 clarify) with routes; the mock "handles" clarify (not refused).
- `sft/train.py` — generator pool excludes `route=="clarify"` (clarify is interpreter-only; never a
  generator target).
- Tests: extended `tests/test_unsupported_gen.py` (portable path, route/kind tagging, out_of_gamut
  validation); NEW `tests/test_taxonomy_sync.py` (the ADR-0023 sync test: briefs/cues cover the
  taxonomy with no drift, fixtures use only taxonomy categories, metrics split by kind).

**Exit criteria — verified (ADR 0023)**
- **0 skips**: all 272 unsupported rows in `active_rows.jsonl` are now relative and resolve locally
  (`migrate_unsupported_portable` reports `unresolved=0`); on Colab they resolve against the same
  staged `luts/` tree the 869 supported fivek rows already use (proof the images are staged).
- out_of_gamut + clarify present across all five files, driven by one source of truth; the sync test
  fails if any file drifts.
- Locked knobs untouched (`configs/sft_default.yaml`, `sft/config.py` not in the diff; only row
  selection in `train.py` changed, not a knob).
- `python3 -m pytest -q` → **322 passed** (was 310; +12 new).

**Follow-up (not blocking P2, teacher/data step).** Generating NEW out_of_gamut/clarify TRAINING
rows into the corpus is wired (`scripts.generate_unsupported`, teacher-gated) but deferred so it does
not re-shuffle the existing versioned 272-row corpus; out_of_gamut recall + clarify are already
exercised via the eval smoke fixtures. Best folded into the P4 teacher pass.

**No Colab needed for P2.**

Commit: see `feat/two-stage` history (P2).

---
