# AGENTS.md — LUT-SLM SFT improve loop (persistent context for Codex)

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
  → 0 rows trained. The trainer **aborts non-zero** on 0 rows (no fake `[sft][OK]`).

## The metric (what the loop optimizes)
Held-out, teacher-forced **LUT-code token accuracy** via `python -m sft.score_tokens`
(direction=**max**). Decoder-free proxy for LUT fidelity — faithful-ish (exact codes → target LUT)
but **not** perceptual ΔE (ΔE needs the disabled frozen decoder, out of bounds). Held-out slice is
`sft.holdout.is_holdout` (excluded from training). One `METRIC=<float>` sentinel line is the
contract; the last one wins.

## Running one eval (on the A100)
`python -m sft.bilevel_bridge --mode colab --config /content/SLM/candidate.json --smoke-size <N>
--run-id bl` — validates the candidate against `SFTConfig`, trains (own process group + timeout +
GPU flock), guards `rows_trained>0`, scores the holdout, prints one `METRIC=`, and (if `PUSH_HF_REPO`
+ `HF_WRITE_TOKEN` are set) uploads the adapter. Bigger `--smoke-size` = more meaningful accuracy but
slower. (The file is named `bilevel_bridge.py` for historical reasons; it is just the train→score
→upload eval unit — there is no bilevel engine involved.)

## The improve loop (Codex drives it — plain greedy, NO engine, NO skill)
Follow the goal prompt you were given. This is a simple hill-climb: there is **no** spec.json /
init_run / run_iteration / driver and **no skill** to invoke. Per run: change ONE knob off the
best-so-far config, run one eval on Colab (press "Run All"), read the `METRIC=` from the local
`.ipynb`, keep the best, stop after **≤5 runs**. Keep state in `./sft_improve_ledger.json` via
`python scripts/sft_ledger.py`. Helpers: `scripts/write_ipynb_config.py` (inject the candidate into
the notebook), `scripts/read_ipynb_metric.py` (read the result back). **B=1** always (one A100).

## Locked knobs (never propose these)
`epochs`(=2), the batch triple (`per_device_batch_size`·`gradient_accumulation_steps`==
`effective_batch_size`), `num_new_tokens`(=259), `base_model_id`, quant scheme, `max_seq_len`, `seed`,
paths. Only vary: `learning_rate_lora`, `lora_r`, `lora_alpha`, `lora_dropout`, `warmup_ratio`,
`max_grad_norm`, `weight_decay`, `max_pixels`. `max_pixels` upper bound is 401408 (higher can
end-truncate the 64 target tokens). The bridge rejects any config that violates `SFTConfig`.

## Gate 0 (must pass before the loop)
One smoke run showing: `training rows=…`; `[sft][skip]` count == only the abs-path unsupported rows;
≥2 finite falling loss lines; `[sft][OK] steps=N>0`; `adapter_manifest.json` identity non-null and
== `vq_v2_srgbres_17to4_cb256_t64__w91cffdd2c82f`; `sft.score_tokens` emits a finite `METRIC=`.

## Resilience (unattended overnight)
Colab **Pro+**; the bridge uploads each adapter to `PUSH_HF_REPO` with `HF_WRITE_TOKEN` (read-only
`HF_TOKEN` 403s). `vocab_resize` is unseeded — run it **once** and reuse; never re-run between evals
(the notebook cell guards this). Do not delete `base_resized`, adapters, or the corpus.
