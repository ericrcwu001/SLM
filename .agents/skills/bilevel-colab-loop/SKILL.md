---
name: bilevel-colab-loop
description: >-
  Run the bilevel autoresearch loop over the LUT-SLM QLoRA SFT on a Colab A100, driven by Codex.
  Use when the user wants to autonomously tune SFT hyperparameters (LoRA rank/alpha/dropout, LR,
  warmup, max_pixels) against held-out LUT-code token accuracy, oversee training on Colab, or
  continue/resume a bilevel run. The deterministic loop state lives in the Bilevel Python engine;
  this skill makes Codex the thin actuator that proposes candidates, triggers a remote Colab run via
  Computer Use, reads the metric back from the LOCAL .ipynb, and records it. NOT for editing the
  frozen tokenizer, decoder, or the corpus (all immutable/read-only).
---

# Bilevel loop over SFT on Colab (Codex-native)

**Engine:** `/Users/ericwu/Developer/Bilevel` (ENGINE). **Spec:** `configs/bilevel/spec.json` (SLM
repo). **Run dir:** `ENGINE/runs/sft_stage7_tokacc` (RUNDIR). **Notebook:** the connected
`notebooks/sft_stage7_run.ipynb` (NB). Run engine scripts as `python ENGINE/scripts/<x>.py …`.

The engine owns budget/patience/best-config/trace; you (Codex) are the actuator. Metric = held-out
teacher-forced **LUT-code token accuracy** (direction=**max**), measured on the A100 by
`sft.bilevel_bridge` and read back from NB. **Never** run `bilevel_driver.js` (a Claude Code
Workflow — not executable here).

## Hard invariants (read `AGENTS.md`)
- Only the `param_space` knobs in `spec.json` vary; everything else is locked by omission. The bridge
  pre-validates and rejects a candidate that would trip `SFTConfig.__post_init__`.
- `B=1` (one A100). Frozen tokenizer immutable; corpus read-only. Never enable `eval/lut_decoder.py`.
- Trust a run only if it shows `steps>0` and a finite `METRIC=` — a wrong `SLM_ARTIFACT_ROOT`
  (`/content/slm` vs `/content/SLM`) silently skips every image. `read_ipynb_metric.py` enforces this.

## 0. Prerequisites (must be green before the loop)
1. Colab A100 connected (**Auto Connect**, never "New Colab Server"); corpus staged to `/content/slm`;
   `SLM_ARTIFACT_ROOT=/content/slm`; `models/base_resized` built once (with `preprocessor_config.json`
   + correct non-null `vocab_resize_manifest.json` identity).
2. **Gate 0** passed: one observable smoke run (`steps>0`, ≥2 falling loss lines, identity binds,
   `sft.score_tokens` emits a finite `METRIC=`). Do not start the loop until Gate 0 is green.

## 1. Initialize the run (once)
1. Measure the **baseline**: write `baseline_config` (from `spec.json`) as the candidate, run it on
   Colab via the bridge (steps 3b–3d below), read `baseline_acc`.
2. `python ENGINE/scripts/init_run.py --spec configs/bilevel/spec.json --run-dir RUNDIR --metric <baseline_acc>`
3. `python ENGINE/scripts/preflight.py --spec configs/bilevel/spec.json` — note whether Level-2 will
   fire (T>K*M). If it warns "Level 2 never fires", tell the user it will run as a plain sweep.
4. Present the objective, param_space, baseline, and preflight to the user; on approval:
   `python ENGINE/scripts/approve_run.py --run-dir RUNDIR`.

## 2. The loop (repeat until stop)
1. `python ENGINE/scripts/status.py --run-dir RUNDIR` → read `t,T,no_improve,patience,best_val,
   since_strategy,level2_fires_next_strategy,frozen,guidance`. **Stop if `done` (t≥T) or
   `no_improve≥patience`** → go to §4.
2. `python ENGINE/scripts/digest.py --run-dir RUNDIR` → the trace digest.
3. **Propose ONE candidate** (respect `param_space`, `frozen`, `guidance`; be diverse vs the trace —
   vary the dominant knob `learning_rate_lora` and LoRA capacity). Write
   `RUNDIR/proposal.json` = `{"params": {…}, "hypothesis": "…"}`.
   a. `python .agents/skills/bilevel-colab-loop/scripts/write_ipynb_config.py --notebook NB --params-file RUNDIR/proposal.json`
   b. Record NB mtime. **Computer Use → "Run All"** in the Cursor Colab extension (config cell writes
      `/content/SLM/candidate.json`; the eval cell runs
      `python -m sft.bilevel_bridge --mode colab --config candidate.json --smoke-size <N> --run-id bl`).
   c. Wait until the eval cell finishes and NB is saved (autosave, or Save). 
   d. `python .agents/skills/bilevel-colab-loop/scripts/read_ipynb_metric.py --notebook NB --min-mtime <recorded>`.
4. **Record:**
   - `status=ok` → `python ENGINE/scripts/run_iteration.py --run-dir RUNDIR --proposal RUNDIR/proposal.json --pre-shaped --metric <metric>`.
   - `status!=ok` (no_metric/failed/stale) → mark the proposal blocked and record it as dropped:
     set `"blocked": true, "reason": "<why>"` in `proposal.json`, then
     `python ENGINE/scripts/run_iteration.py --run-dir RUNDIR --proposal RUNDIR/proposal.json --pre-shaped`
     (no `--metric`; the engine drops it, `evaluated=0`). Then propose a different candidate.
5. **Persist (resilience):** copy the new adapter + `RUNDIR/` to Drive/HF (see `AGENTS.md`). Loop to §2.1.

## 3. Advanced (optional, stretch) — strategy & Level-2
Only if `status.py` shows `since_strategy≥K`: read `ENGINE/prompts/strategy.md`, produce a
`{frozen,guidance,outer_cycle}` patch, apply with `ENGINE/scripts/apply_strategy_patch.py`. If
`level2_fires_next_strategy` is true, Level-2 (new search-strategy code) uses
`ENGINE/prompts/level2_*.md` + `ENGINE/scripts/validate_mechanism.py`; `safety.l2:"approve"` means
**pause and show the user the mechanism code before adopting**. Level-2 is heavy under Codex — skip it
tonight unless the user asks; Level-1 (§2) is the achievable core.

## 4. Report
`python ENGINE/scripts/report.py --run-dir RUNDIR` → present best config + Δ vs baseline, why it
stopped, and the per-candidate table. If it ran as a sweep (no Level-2), say so plainly. Remind the
user the metric is a decoder-free proxy (token accuracy), not perceptual ΔE.
