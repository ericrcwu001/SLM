# Collapse fix — exposure-bias mitigation plan (index)

**Audience:** an engineer/agent implementing the fix with no prior context on this investigation.
Read this file first, then the three numbered docs in order. Each numbered doc is a
self-contained, executable implementation plan.

## The problem (validated)

The prompt→LUT generator (Qwen2.5-VL-3B QLoRA emitting 64 VQ code tokens `<lut_000>..<lut_255>`
that the frozen VQ-VAE decodes to a color LUT) works under **teacher forcing** but **collapses
free-running**. Measured on the P6 two-stage adapter (`p6_twostage_d0f9c744_smokefull`), on a **64-row
slice** (`--behavioral-limit 64`) of the **120-row** unit-aware holdout, conditioned on
`attribute_spec_text`:

| decode | behavioral fidelity | collapse rate | notes |
|---|---|---|---|
| teacher-forced argmax (perfect prefix every step) | **0.708** | 0% | optimistic ceiling — see caveat |
| free-running greedy | 0.159 | 94% | over-commits to a dominant code |
| free-running sample t=0.7 | 0.091 | 14% | diverse but wrong-direction |
| real corpus codes (upper bound of the metric) | ~0.89 | 0% | ceiling of the ruler itself |

**Diagnosis: exposure bias.** The model predicts well given a correct prefix but cannot recover from
its own outputs, so errors compound over the 64-code trajectory. This is a *training/decoding-regime*
problem, **not** a broken two-stage seam and **not** an architecture problem (the AR-64-code head
learned the mapping). `refused=0` in the run, so the gap is trajectory drift/collapse, not a
grade-vs-refuse routing failure.

**Two caveats to keep in mind (do not overclaim):**
1. **0.708 is an optimistic ceiling, not "the model understands the spec."** The 64 gold codes *are*
   the target LUT, so predicting code *i* given 63 correct gold codes is mostly local denoising with
   the answer in hand. The realistic ceiling for an inference-only fix is **`oracle@N`** (best of N
   samples), which Doc 01 measures — not 0.708.
2. The behavioral-fidelity reward scores only the *requested* axes (direction+magnitude), and its own
   ceiling is ~0.89, so it has blind spots. It is nonetheless the **true deployment objective**, not a
   learned proxy — which is why reranking against it is safe (negligible reward-hacking; self-limiting).

## The plan (three phases, gated)

1. **[`01_oracle_at_n.md`](01_oracle_at_n.md) — measure `oracle@N` (decides everything).** Cheap, no
   training. Does sampling ever *cover* a high-fidelity trajectory? This gates 02 and 03.
2. **[`02_best_of_n_reranking.md`](02_best_of_n_reranking.md) — best-of-N reranking (inference, no
   retrain).** Sample N, rerank by behavioral fidelity, return the best. Deployable immediately if
   coverage (from 01) is adequate.
3. **[`03_self_distillation.md`](03_self_distillation.md) — self-distillation (training).** Harvest
   best-of-N winners over the training split, SFT a fresh adapter on them (stable, fits the locked
   2-epoch budget), which bakes good trajectories into the weights and moves the free-running
   distribution. Composes with 02.

**Gate logic (from Doc 01):**
- `oracle@N ≳ 0.6` at feasible N → coverage is fine → do **02** (ship) and **03** (amortize); RL not needed yet.
- `oracle@N` stalls ≲ 0.3 → coverage gap → reranking cannot help; escalate to sequence-level RL
  (GRPO/MRT), which is **out of scope for these docs** (documented as reserve only).

Rejected alternatives (do not implement): scheduled sampling (biased gradient; the locked 2-epoch
budget removes its annealing runway), DAgger (no state-conditional expert for a diverged VQ prefix),
plain beam search / bare nucleus without a reranker (beam amplifies the degenerate mode).

### Gate thresholds (provisional — tune once Doc 01 runs)

The `oracle@N` (mean over rows of the max fidelity in N samples) at a **feasible N = 32** (chunk if
GPU-OOM; see Doc 01), read against the greedy baseline 0.159 and the ~0.89 ceiling:
- **`oracle@32 ≳ 0.6`** → coverage is good → do **02** (ship) and **03** (amortize). RL not needed yet.
- **`oracle@32` in (0.3, 0.6)`** → partial coverage → **02 is still worth shipping** as a quality knob,
  but treat **03's ceiling as capped**; re-measure `oracle` at higher N (64) before committing to RL.
- **`oracle@32 ≲ 0.3`** → coverage gap → reranking/distillation cannot help; escalate to sequence-level
  RL (GRPO/MRT), which is **out of scope for these docs** (reserve only).

These 0.6/0.3 cut points are provisional anchors, not measured — report the full `oracle@k` curve
(k=1,4,8,16,32,64) so the *shape* informs the call, and revise the thresholds if the curve is still
climbing at k=32.

## Prerequisites (obtain BEFORE Doc 01 — the docs assume these exist)

A fresh clone does **not** have the model weights or the staged image/tokenizer artifacts. Do these
first (all are already automated in `notebooks/phase1_behavioral_score.ipynb` CELLs 1–2 — read that
notebook if a step is unclear):

1. **Corpus rows:** `data/active_sft/active_rows.jsonl` is **git-tracked** (present after clone; 3033
   rows). Nothing to fetch.
2. **Staged artifacts (images + frozen VQ tokenizer):** set `SLM_ARTIFACT_ROOT` and stage the ~9.85 GB
   corpus: `slm_stage stage --durable-root hf://datasets/ericrcwu/LUT_SLM --local-root /content/slm`
   then `export SLM_ARTIFACT_ROOT=/content/slm`. Row `image_path`s and `tokenizer/final/model.pt`
   (the frozen decoder) resolve against this root. Requires a read HF token.
3. **`models/base_resized`** (gitignored; needed by the on-disk load path):
   `python -m sft.vocab_resize --out models/base_resized` (loads the base in fp32 on CPU ~12 GB →
   A100/High-RAM). Or use the T4-safe in-memory-resize path (see `notebooks/phase0_collapse_check.ipynb`
   CELL 3) instead of building it.
4. **The P6 adapter** (gitignored; download from HF):
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download(repo_id="ericrcwu/LUT_SLM_sft_adapters",
                     allow_patterns=["p6_twostage_d0f9c744_smokefull/*"],
                     local_dir="models/sft_adapters", token=HF_TOKEN)
   # -> models/sft_adapters/p6_twostage_d0f9c744_smokefull
   ```

**Shared model loader (build once, in Doc 01, then import everywhere).** All three docs load the same
model. Doc 01 extracts the loader block from `sft/score_tokens.score` into
`sft/loader.py:load_eval_model(cfg, resized_model, adapter) -> (model, processor)` and updates
`score_tokens` to import it (keeping its tests green). Docs 02/03 import `load_eval_model` — do **not**
copy-paste the block three times.

## Building blocks that ALREADY EXIST (committed on `feat/two-stage`)

Reuse these; do not reimplement.

| Component | Path | Signature / note |
|---|---|---|
| Free-running generate (single) | `sft/generate.py` | `generate_codes(model, processor, *, image, text, sampling=None, max_new_tokens=68, device=None) -> list[int]|None` |
| " (row, matches training) | `sft/generate.py` | `generate_codes_for_row(model, processor, row, *, input_field="instruction", bucketize=False, sampling=None, ...)` |
| Grammar + id helpers | `sft/generate.py` | `SpecialIds(tokenizer)`, `make_prefix_fn(prompt_len, ids)`, `codes_from_output(output_row, prompt_len, ids)` |
| Behavioral score of codes | `eval/behavioral_fidelity.py` | `score_generation(codes, spec, *, target_codes=None, final_dir=None, tol=1.0, collapse_floor=0.01) -> dict` (note: `dominant_share_max` is a param of `score_from_lut`, NOT `score_generation`) |
| " of an already-decoded LUT | `eval/behavioral_fidelity.py` | `score_from_lut(pred_lut, spec, *, target_lut=None, codes=None, ...) -> dict` |
| Decode codes → LUT | `eval/behavioral_fidelity.py` | `decode_codes(codes, *, final_dir=None) -> np.ndarray [17,17,17,3]` (frozen `VQVAE.decode`) |
| Aggregate records | `eval/behavioral_fidelity.py` | `summarize_fidelity(records) -> dict` |
| Requested (canonical) spec for a row | `data_pipeline/attribute_spec.py` | `ground_truth_attribute_spec_text(row, *, bucketize=False) -> str` |
| Rows + holdout split | `sft/example.py`, `sft/holdout.py` | `load_rows(path)`, `supported_rows(rows, holdout=None|True|False)`, `is_holdout_row(row)` (keys on `split_unit_id`) |
| Teacher-forced example builder | `sft/example.py` | `build_supervised_example(processor, row, cfg, *, device, input_field, augment=False)` |
| Assistant-target string from codes | `scripts/materialize_target_tokens.py:53` | `_assistant_target(codes) = "<lut_bos> " + " ".join("<lut_%03d>"%c) + " <lut_eos>"` |
| Trainer | `sft/train.py` | hand-written QLoRA loop; reads `cfg.active_rows_path`; **starts from fresh LoRA (no resume)** |
| Scorer (behavioral pass built in) | `sft/score_tokens.py` | `--behavioral-sampling greedy|sample|both`, emits `behavioral`/`behavioral_sampled` summaries |

A **record** returned by `score_generation`/`score_from_lut` has at least:
`behavioral_fidelity` (float|None), `collapsed` (bool), `degenerate_identity` (bool),
`residual_norm` (float), `route` (str), and — when inputs allow — `code_stats`
(`dominant_share`, `entropy_norm`, …) and `decoded_delta_e` (`mean`, `p95`, `max`).

## Shared conventions (all three docs)

- **Conditioning MUST match how P6 was trained:** `input_field="attribute_spec_text"`, no bucketize,
  no augment. Load config from `configs/candidate_two_stage.json` (`_load_config` parses JSON via
  `yaml.safe_load`). The scorer already defaults to `cfg.input_field`.
- **Score against the CANONICAL requested spec:** `ground_truth_attribute_spec_text(row)` (default
  `bucketize=False`). Never score against a bucketized/augmented spec.
- **Reranker score (define ONCE, import everywhere):** Doc 01 adds `rerank_key(rec)` to
  `eval/behavioral_fidelity.py`; Docs 01 and 02 import it (do not re-spell it locally). Order:
  primary `behavioral_fidelity` (needs only the requested spec + generated codes — **no target LUT
  required**, so it works at deploy); then not `collapsed`; then higher `code_stats.entropy_norm`;
  then lower `decoded_delta_e.mean` **only when that key is present** (eval, where `target_codes`
  were passed). At deploy `decoded_delta_e` is absent, so the pick never depends on it.
- **Holdout is sacred.** `is_holdout_row` keys on `split_unit_id`. Distillation (Doc 03) rewrites
  **training** rows only; holdout rows are copied unchanged so behavioral eval stays honest.
- **Model loading:** two supported paths — (a) `models/base_resized` on disk + `PeftModel.from_pretrained`
  (what `score_tokens` uses; needs `sft.vocab_resize` first), or (b) in-memory vocab resize on the raw
  base (T4-safe; see `notebooks/phase0_collapse_check.ipynb` CELL 3). Either yields an identical
  effective model (the adapter's `modules_to_save=["embed_tokens","lm_head"]` are swapped in wholesale).
- **Cost driver:** free-running generation (64 autoregressive steps). Batch it with
  `num_return_sequences` (Doc 01 adds `generate_codes_batch`) — one `.generate` call yields N samples.
- **Locked knobs (`AGENTS.md:48-53`):** `epochs=2`, the batch triple, `num_new_tokens=259`,
  `base_model_id`, quant scheme, `max_seq_len`, `seed`, **and `paths`**. Do not touch these in the
  bilevel search. These docs add **no** new tunable-search knob. Doc 03 does point the trainer at a
  different `active_rows_path` — which AGENTS.md's "paths" lock nominally covers — but the lock governs
  the *bilevel hill-climb loop*; a one-off, out-of-loop distillation experiment with **all locked
  hyperparameters identical** is a deliberate, documented exception, not a proposed search knob. Call
  it out explicitly when you run it.

## Status / definition of done

- Doc 01: `eval/oracle_at_n.py` + `generate_codes_batch` + tests; a Colab run producing the
  `oracle@N` curve and the gate decision.
- Doc 02: `eval/best_of_n.py` + tests; best-of-N behavioral fidelity on the holdout exceeds greedy
  0.159 (bounded by oracle@N).
- Doc 03: `scripts/build_distillation_corpus.py` + a `configs/candidate_distill.json` + tests; a
  retrained adapter whose **free-running greedy** behavioral fidelity beats the 0.159 baseline on the
  (untouched) holdout.
