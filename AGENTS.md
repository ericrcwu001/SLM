# AGENTS.md — LUT-SLM SFT + bilevel loop (persistent context for Codex)

Prompt-to-LUT VLM: **Qwen/Qwen2.5-VL-3B-Instruct** fine-tuned with QLoRA to emit **64 VQ code
tokens** (`<lut_000>…<lut_255>`) that a **frozen** VQVAE decodes to a color-grading LUT. The frozen
tokenizer, its decoder, and the corpus are **immutable / read-only**. Never enable
`eval/lut_decoder.py`, never retrain/re-gate/re-freeze the tokenizer, never modify `data/` or `luts/`.

## Where things live (Colab case trap — this bites)
- **Repo clone:** `/content/SLM` (uppercase) — code, `active_rows.jsonl`, `models/base_resized`,
  `models/sft_adapters/…`, the run notebook.
- **Staged corpus:** `/content/slm` (lowercase) — images + `tokenizer/final/` weights.
- **Always** `export SLM_ARTIFACT_ROOT=/content/slm` (lowercase). Images/tokenizer resolve against it;
  everything else is CWD-relative to `/content/SLM`. One wrong-case value silently skips every image
  → 0 rows trained. The trainer now **aborts non-zero** on 0 rows (no fake `[sft][OK]`).
- **Bilevel engine:** `/Users/ericwu/Developer/Bilevel` (ENGINE). **Run dir:**
  `ENGINE/runs/sft_stage7_tokacc`. **Spec:** `configs/bilevel/spec.json` (this repo).

## The metric (what the loop optimizes)
Held-out, teacher-forced **LUT-code token accuracy** via `python -m sft.score_tokens`
(direction=**max**). Decoder-free proxy for LUT fidelity — faithful-ish (exact codes → target LUT) but
**not** perceptual ΔE (ΔE needs the disabled frozen decoder, which is out of bounds). Held-out slice
is `sft.holdout.is_holdout` (excluded from training). One `METRIC=<float>` sentinel line is the
contract; the last one wins.

## Running one eval (on the A100)
`python -m sft.bilevel_bridge --mode colab --config /content/SLM/candidate.json --smoke-size <N>
--run-id bl` — validates the candidate against `SFTConfig`, trains (own process group + timeout +
GPU flock), guards `rows_trained>0`, scores the holdout, prints one `METRIC=`. Bigger `--smoke-size`
= more meaningful accuracy but slower; pick per time budget.

## Loop control lives in the Python engine, not in your head
Drive the plain CLIs (`init_run.py`, `preflight.py`, `approve_run.py`, `digest.py`,
`run_iteration.py --pre-shaped --metric`, `status.py`, `report.py`). Read stop conditions from
`status.py` (`done`, `no_improve≥patience`). **Never** run `bilevel_driver.js` (a Claude Code
Workflow). See the `bilevel-colab-loop` skill for the step-by-step. Budgets: `B=1`, small `T`,
`patience` per `spec.json`.

## Locked knobs (never propose these)
`epochs`(=2), the batch triple (`per_device_batch_size`·`gradient_accumulation_steps`==
`effective_batch_size`), `num_new_tokens`(=259), `base_model_id`, quant scheme, `max_seq_len`, `seed`,
paths. Only vary the `param_space` in `spec.json`. `max_pixels` upper bound is 401408 (higher can
end-truncate the 64 target tokens).

## Gate 0 (must pass before the loop)
One smoke run showing: `training rows=…`; `[sft][skip]` count == only the abs-path unsupported rows;
≥2 finite falling loss lines; `[sft][OK] steps=N>0`; `adapter_manifest.json` identity non-null and
== `vq_v2_srgbres_17to4_cb256_t64__w91cffdd2c82f`; `sft.score_tokens` emits a finite `METRIC=`.

## Resilience (unattended overnight)
Colab **Pro+**; mount Drive; persist `models/base_resized`, each adapter, and `ENGINE/runs/…` to
Drive/HF after every eval; provision the **write** HF token (`SLM_Alpha_Write`) on the VM and push
with `slm_stage push --local-root /content/SLM` (uppercase). `vocab_resize` is unseeded — run it
**once** and reuse; do not re-run between evals.
