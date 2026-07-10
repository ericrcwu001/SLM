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

## P3 — behavior_v2 axes + unified tag vocabulary (ADR 0022) ✅  (local; no Colab)

**Goal.** Give color language real resolution: extend the measured behavior vector with absolute +
per-region hue, per-hue saturation, contrast shape, and matte; unify the three divergent tag tables
into one source of truth; re-measure the corpus into a new versioned `measured_behavior`.

**Changes**
- `data_pipeline/behavior_vector.py` — `measure_behavior` now emits the 9 `behavior_v2` axes
  (docs/attribute_spec.md §3b), all from the EXISTING probes (no new probe): `global_hue_deg`,
  `global_hue_magnitude`, `shadow_hue_deg` / `midtone_hue_deg` / `highlight_hue_deg`,
  `per_hue_saturation` (7 input-hue sectors), `contrast_toe_delta` / `contrast_shoulder_delta`,
  `matte_strength`. All 27 `behavior_v1` fields are retained byte-for-byte (verified: re-measure
  changed 0 v1 numeric values).
- `data_pipeline/constants.py` — `BEHAVIOR_VECTOR_VERSION → behavior_v2` AND
  `QUALITY_FILTER_VERSION → quality_v8_behavior_v2` (the cache-currency check keys on the latter,
  `run_pipeline.py:182`, so bumping it is what forces re-measurement — the ADR-0022 gotcha).
- `eval/tag_vocabulary.py` — NEW single source of truth (pure; in `eval` so both layers import it
  without a cycle): canonical `DIRECTIONAL_TAG_AXIS`, `RETIRED_ALIASES`
  (`more_magenta→tint_magenta`, `higher_contrast→more_contrast`, `desaturated→muted`, …),
  `STYLE_TAGS`, and the NEW behavior_v2 hue families (`hue_cast_*`, `sat_*_up/down`).
- `data_pipeline/instruction_gen.py` + `eval/frontier_scoring.py` — `_TAG_BEHAVIOR` and
  `TAG_DIRECTIONS` now DERIVE from the unified vocabulary; both canonicalize incoming tags, so
  legacy rows/fixtures using a retired alias still score while the alias is gone from the code
  vocabulary.
- `eval/fixtures/make_smoke_rows.py` — supported fixtures updated to canonical tags.
- `scripts/remeasure_behavior_v2.py` — NEW idempotent re-measurement: joins each supported active
  row (`id`→provenance `residual_key`→`luts/canonical_residual/<key>.npy`), reconstructs the
  absolute LUT, re-measures at behavior_v2, and writes the NEW versioned vector (backup
  `*.bak_pre_behavior_v2`, manifest `behavior_vector_version` + `behavior_v2_remeasure` block).
  Frozen `luts/`/tokenizer untouched (ADR 0026).
- Tests: extended `tests/test_behavior_vector.py` (v2 axes: identity≈0, warm→hue 90°, teal-orange
  region-hue split, matte>0, per-hue-sat 7-sector map); NEW `tests/test_tag_vocabulary.py` (the
  unify sync test: instruction_gen + frontier_scoring both source the one table; aliases retired
  but still ingest).

**Exit criteria — verified (ADR 0022)**
- behavior_v2 axes present on all 2761 supported rows (`remeasure_behavior_v2`: remeasured=2761,
  unresolved=0), all v1 fields unchanged.
- Both versions bumped; unified tag table sourced from one module (drift-guarded by the sync test).
- `python3 -m pytest -q` → **336 passed** (was 322; +14).

**No Colab needed for P3.** The behavior_v2 `measured_behavior` is the input to P4 captioning + the
oracle gate.

Commit: see `feat/two-stage` history (P3).

---

## P4 — AttributeSpec + captioner + ORACLE GATE (ADR 0021, 0026) ⏸ waiting on Colab

**Goal.** Freeze the interpreter↔generator interface in code (deterministic, round-trippable
`attribute_spec_text`), build the captioner that produces the interpreter's training data, and prove
the **hard go/no-go**: a ground-truth spec must drive the current generator ≥ the one-stage metric.

**Changes (LOCAL, done)**
- `data_pipeline/attribute_spec.py` — NEW `AttributeSpec` (behavior_v2 axes + route + confidence +
  out_of_gamut/refuse_reason + source_text) with a deterministic, round-trippable serializer/parser
  (`serialize`/`parse`; `parse(serialize(spec)) == spec`), `from_measured_behavior` (the ground-truth
  path), `measured_behavior_to_text`, and the `is_backed` backing gate (ADR 0021 §6). Bipolar tags
  encode direction with a positive magnitude (`muted=+4.8`); hues are integer degrees; canonical key
  order + fixed float precision.
- `sft/example.py` — `build_supervised_example` gained `input_field="instruction"` (backward-compat);
  `"attribute_spec_text"` selects the two-stage input. `sft/score_tokens.py` `score()` gained
  `input_field` + a `prep_row` hook.
- `sft/oracle_gate.py` — NEW: scores the CURRENT adapter on the P1 unit-aware holdout TWICE
  (instruction=baseline vs ground-truth attribute_spec_text=oracle), reusing the P1 unit-clustered
  CIs; prints `METRIC_baseline=`, `METRIC_oracle=`, `{"oracle_gate": …}` and PASS ⇔ oracle ≥ baseline.
- `notebooks/oracle_gate_run.ipynb` — NEW self-provisioning A100 notebook (checks out `feat/two-stage`,
  stages corpus, reuses `base_resized`, downloads the current adapter, runs the gate).
- `data_pipeline/captioning.py` + `scripts/generate_captions.py` — NEW teacher captioner: many
  diverse captions/LUT (literal/metaphor/mood/concept/slang) → the LUT's `attribute_spec_text`
  (grounded target), resumable, `--dry-run`; writes `caption_rows.jsonl` (the P5 interpreter corpus,
  a NEW versioned artifact).
- Tests: `test_attribute_spec.py` (11), `test_oracle_gate.py` (4), `test_captioning.py` (6).
  `python3 -m pytest -q` → **357 passed** (+21).

**Captioning deferred (not blocking the gate).** `TFY_BASE_URL` is unset locally, so per the plan's
"else hand off" clause the captioner cannot run here; and captions only feed P5, which the oracle gate
gates. The captioner is built + dry-run-validated; the full run is the first post-gate step (run
locally once `TFY_BASE_URL` is set, or hand off). Building it before the go/no-go would risk wasted
teacher spend.

### 🛑 COLAB HANDOFF — ORACLE GATE (hard go/no-go; do NOT build P5/P6 until confirmed)

Run `notebooks/oracle_gate_run.ipynb` on the A100 (Auto Connect → Run All). It:
`export SLM_ARTIFACT_ROOT=/content/slm` (LOWERCASE staged corpus; code at `/content/SLM` UPPERCASE),
stages if missing, reuses `models/base_resized`, downloads adapter
`ericrcwu/LUT_SLM_sft_adapters/bl_a0ccbcff_smokefull` (the current one-stage full-run winner), and
runs `python -m sft.oracle_gate --adapter models/sft_adapters/bl_a0ccbcff_smokefull --limit 0`.

Secrets on the remote kernel: upload `.env` or paste `HF_TOKEN` (read scope OK) via getpass; the gate
needs no write token.

**Paste back**: the `METRIC_baseline=…` line, the `METRIC_oracle=…` line, and the
`{"oracle_gate": …}` JSON line.

**Gate**: PASS ⇔ `METRIC_oracle ≥ METRIC_baseline` (the ground-truth spec is at least as good a
conditioner as the free-text instruction → the semantic-IR seam is not lossy → proceed to P5/P6).
FAIL ⇒ the two-stage move is off; stop and report.

Result: _pending Colab paste._

---
