# Doc 00 — GRPO grounding brief (canonical API map)

**Audience:** every writer of a `docs/grpo/*` doc, and the engineer who implements the loop. This is
the single source of truth for *what already exists and its exact signature*. **Do not invent an
API.** If a hook you need is not in the "Reusable hooks" appendix below, it is in the "Must build"
list — build it there, do not silently assume it exists. Every signature here was read off the file
on `feat/two-stage` (line numbers are `file:line`); if the code moves, fix this doc.

Read `docs/collapse_fix/README.md` first for the problem framing — this doc is its GRPO sequel.

---

## The job (one paragraph)

The prompt→LUT generator (Qwen2.5-VL-3B, QLoRA/NF4, emits 64 VQ code tokens `<lut_000..255>` that a
**frozen** VQ-VAE decodes to a color LUT) predicts well under teacher forcing (0.708) but **collapses
free-running greedy** (behavioral fidelity 0.087–0.159, 94 % collapse) — textbook exposure bias, not
a broken seam. Sampling *covers* good trajectories (`oracle@32 = 0.42`) and best-of-N reranking
already **ships 0.42**. **GRPO's job:** directly optimize the *free-running* behavioral reward so
**greedy** fidelity climbs toward/past the 0.42 oracle, without reward-hacking. Policy = the QLoRA
LoRA params (init from the P6 SFT adapter); reference for KL = the frozen SFT init; base 4-bit frozen
and shared. Group-relative advantage (G samples/prompt, `(r - mean)/(std+eps_adv)`), clipped surrogate +
KL over the 64-code assistant span, assistant-only masking.

---

## Reusable hooks (import these — do NOT reimplement)

### Reward — `eval/behavioral_fidelity.py` (canonical, per-sample, torch only in `decode_codes`)

| Hook | `file:line` | Signature | GRPO use |
|---|---|---|---|
| `score_generation` | `eval/behavioral_fidelity.py:202` | `score_generation(codes, spec, *, target_codes=None, final_dir=None, tol=1.0, collapse_floor=0.01) -> dict` | Canonical single-sample reward (decode + score). The **parity oracle** for the batched path; not the hot path. |
| `score_from_lut` | `:165` | `score_from_lut(pred_lut, spec, *, target_lut=None, codes=None, tol=1.0, collapse_floor=0.01, dominant_share_max=0.5) -> dict` | Torch-free scorer of an already-decoded LUT. Note `dominant_share_max` lives here, **not** on `score_generation`. |
| `rerank_key` | `:139` | `rerank_key(rec) -> tuple` | The **no-target-leakage contract**: `(fidelity, not collapsed, entropy_norm, -ΔE)`, HIGHER wins. ΔE term defaults to neutral `0.0` when no target scored, so reward never depends on a target LUT. GRPO's reward must obey this same "agreement with the requested spec only" rule. |
| `decode_codes` | `:64` | `decode_codes(codes, *, final_dir=None) -> np.ndarray [17,17,17,3]` | Frozen `VQVAE.decode` → absolute LUT, clipped [0,1]. |
| `code_histogram_stats` | `:81` | `code_histogram_stats(codes) -> dict` | `dominant_share`, `entropy_norm`, `unique_codes`, `dominant_code` — the collapse diagnostics + reward-shaping / logging inputs. |
| `behavioral_agreement` | `:103` | `behavioral_agreement(spec, mb, *, tol=1.0) -> dict` | Fraction of asserted axes backed (sign+magnitude). Underlies `behavioral_fidelity`. |
| `summarize_fidelity` | `:221` | `summarize_fidelity(records) -> dict` | Aggregate for the periodic holdout eval: `behavioral_fidelity_mean`, `collapse_rate`, `degenerate_rate`, `code_entropy_norm_mean`, `dominant_share_mean`, `decoded_delta_e_mean`. |
| `decoded_delta_e` | `:157` | `decoded_delta_e(pred_lut, target_lut) -> {mean,p95,max}` | Optional ΔE shaping / eval-only diagnostic (needs a target LUT — eval only). |

**Collapse constants** (`eval/behavioral_fidelity.py:46-61`): `TOKEN_COUNT=64`, `CODEBOOK_SIZE=256`,
`DEGENERATE_RESIDUAL_NORM=5e-4`, `COLLAPSE_RESIDUAL_NORM=0.01`, `DOMINANT_SHARE_MAX=0.5`,
`DEFAULT_TOL=1.0`. A row is `collapsed` iff `residual_norm < collapse_floor` **or**
`dominant_share >= dominant_share_max`.

**Record shape** (returned by `score_generation` / `score_from_lut` / `score_batch`): at least
`route`, `residual_norm`, `degenerate_identity` (bool), `collapsed` (bool), `behavioral_fidelity`
(float | **None** for non-grade / axis-less specs), and when inputs allow `agreement`, `code_stats`,
`decoded_delta_e`.

### Reward — `eval/fast_reward.py` (batched, GPU, parity-verified — THE hot path)

| Hook | `file:line` | Signature | GRPO use |
|---|---|---|---|
| `score_batch` | `eval/fast_reward.py:242` | `score_batch(codes_batch, spec, *, device=None, target_codes=None, tol=1.0, collapse_floor=0.01, dominant_share_max=0.5, final_dir=None) -> list[dict]` | **Reward for the G rollouts of one prompt.** One batched decode + reduced `behavior_v2` measurement; records are drop-in for `rerank_key` and numerically equal to the canonical path (`tests/test_fast_reward.py`). |
| `decode_batch` | `:95` | `decode_batch(codes_batch, *, device=None, final_dir=None) -> np.ndarray [B,17,17,17,3]` | Batched frozen decode; bit-identical to stacking `decode_codes`. |
| `measure_reduced_batch` | `:224` | `measure_reduced_batch(luts) -> list[dict]` | Only the asserted `behavior_v2` axes + `per_hue_saturation` + `residual_norm`. |

### Coverage / baseline harness — `eval/best_of_n.py`, `eval/oracle_at_n.py`

| Hook | `file:line` | Signature | GRPO use |
|---|---|---|---|
| `best_of_n_codes` | `eval/best_of_n.py:23` | `best_of_n_codes(model, processor, *, image, cond_text, spec_text=None, n=16, sampling=None, chunk=16, device=None, fast=False) -> (best_codes, best_record)` | The deployable baseline **to beat (0.42)**. `sampling` defaults `{t:0.7, top_p:0.9}`. |
| `best_of_n_for_row` | `:52` | `best_of_n_for_row(model, processor, row, *, n=16, sampling=None, input_field="attribute_spec_text", chunk=16, device=None, fast=False)` | Conditions via `input_text_for` (training parity), scores via `ground_truth_attribute_spec_text` (canonical). **Copy this condition/score split exactly.** |
| `evaluate` | `:65` | `evaluate(model, processor, cfg, *, n=16, temperature=1.0, top_p=0.9, limit=32, chunk=16, input_field=None) -> dict` | Holdout best-of-N summary; all-refused row folded as `fidelity=0.0`. |
| `oracle_at_n.run` | `eval/oracle_at_n.py:74` | `run(model, processor, cfg, *, n=32, temperature=0.7, top_p=0.9, limit=32, chunk=16, input_field=None) -> dict` | oracle@k curve + best_of_N on the holdout — the ceiling GRPO chases. |
| `oracle_and_best` | `:58` | `oracle_and_best(recs_by_row, ks=(1,4,8,16,32,64)) -> dict` | Pure-numpy aggregation. |
| `score_row_samples` | `:36` | `score_row_samples(codes_list, spec_text, target_codes) -> list[dict]` | **Refusal / non-64 → `{behavioral_fidelity:0.0, collapsed:True}`** — the canonical "reward 0 on a supported row" accounting. Reuse this rule. |

### Generation — `sft/generate.py` (the rollout engine + grammar)

| Hook | `file:line` | Signature | GRPO use |
|---|---|---|---|
| `generate_codes_batch` | `sft/generate.py:117` | `generate_codes_batch(model, processor, *, image, text, n, sampling, chunk=16, max_new_tokens=68, device=None) -> list[list[int]|None]` | **Draw the G rollouts** for one prompt (`num_return_sequences` expansion). `sampling` MUST enable sampling. |
| `generate_codes_for_row_batch` | `:157` | `generate_codes_for_row_batch(model, processor, row, *, input_field="instruction", bucketize=False, n, sampling, chunk=16, max_new_tokens=68, device=None)` | Row-conditioned rollouts (training-parity prompt). |
| `generate_codes` | `:68` | `generate_codes(model, processor, *, image, text, sampling=None, max_new_tokens=68, device=None) -> list[int]|None` | Greedy path for the success-gate eval (`sampling=None`). |
| `SpecialIds` | `:27` | `SpecialIds(tokenizer)` → `.bos .lut_eos .unsupported .model_eos .codes[256] .id_to_index` | Token-id ↔ codebook-index map for logprob gather + masking. |
| `make_prefix_fn` | `:39` | `make_prefix_fn(prompt_len, ids)` | The 64-code grammar (`BOS + 64×code + EOS` \| `<unsupported>`). Batch-agnostic (ignores `batch_id`, fixed `prompt_len`) so it survives `num_return_sequences`. **Reuse verbatim** in the trainable-model rollout. |
| `codes_from_output` | `:60` | `codes_from_output(output_row, prompt_len, ids) -> list[int]|None` | Map generated ids → 64 codebook indices, `None` on refusal. |

Constants (`sft/generate.py:21-24`): `TOKEN_COUNT=64`, `CODEBOOK_SIZE=256`, `DEFAULT_MAX_NEW_TOKENS=68`.
**⚠ `generate_*` returns token ids ONLY — no logprobs / scores.** Per-token logprob extraction is a
must-build (see below).

### Model / data / config

| Hook | `file:line` | Signature | GRPO use |
|---|---|---|---|
| `load_eval_model` | `sft/loader.py:14` | `load_eval_model(cfg, resized_model, adapter) -> (model, processor)` | Loads 4-bit NF4 base + `PeftModel.from_pretrained(adapter)`, **`.eval()` (inference-only)**. Good for the reference policy and eval; the **trainable** policy needs `is_trainable=True` (must-build). |
| `build_supervised_example` | `sft/example.py:160` | `build_supervised_example(processor, row, cfg, *, device=None, input_field="instruction", augment=False) -> dict` | The **assistant-only masking spec**: `labels[:, :n_prompt] = -100` (`:218-219`) + exact-64 guard. GRPO masks the identical 64-code span. |
| `input_text_for` | `sft/example.py:136` | `input_text_for(row, input_field, *, bucketize=False, augment_rng=None, jitter=0.3) -> str` | Conditioning text (training parity). Pass `cfg.input_field` (P6 = `attribute_spec_text`), no bucketize, no augment at rollout. |
| `supported_rows` | `sft/example.py:72` | `supported_rows(rows, *, holdout=None|True|False) -> list[dict]` | `False` → train pool (holdout excluded); `True` → holdout eval slice. |
| `is_supported_materialized` | `sft/example.py:54` | `is_supported_materialized(row) -> bool` | 64-token materialized + required fields present. |
| `surviving_code_positions` | `sft/example.py:98` | `surviving_code_positions(tokenizer, input_ids, n_prompt) -> int` | Exact-64 survival check (shared truncation guard). |
| `load_rows` / `resolve_image` / `artifact_root` / `resolve_compute_dtype` | `sft/example.py:66/50/29/33` | — | Row IO + `$SLM_ARTIFACT_ROOT` image resolution + bf16→fp16 dtype policy. |
| `is_holdout_row` | `sft/holdout.py:61` | `is_holdout_row(row, frac=0.06) -> bool` | **Sacred split**; keys on `split_unit_id` (`holdout_key`, `:51`). `DEFAULT_HOLDOUT_FRAC=0.06`. |
| `SFTConfig` / `load_config` | `sft/config.py:27/127` | `load_config(path=None) -> SFTConfig` (frozen dataclass) | Locked identity + knobs (table below). `__post_init__` (`:98-118`) enforces the locks. |

### Trainer building blocks to mirror — `sft/train.py`

| Piece | `file:line` | Note for GRPO |
|---|---|---|
| kbit prep | `sft/train.py:106` | `prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)` |
| LoRA + new-token rows | `:108-113` | `LoraConfig(r, alpha, dropout, target_modules, bias="none", task_type="CAUSAL_LM", modules_to_save=["embed_tokens","lm_head"])`; `get_peft_model`. **Keep the same target modules / modules_to_save** so the GRPO policy is the same param set as P6. |
| optimizer | `:122-123` | `AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate_lora, weight_decay=…)` |
| LR schedule | `:128-132` | cosine w/ warmup (`_lr(step)`) — GRPO may use a flat/own schedule (methodology knob). |
| accumulate / `_optim_step` | `:154-166`, `:170-202` | clip_grad_norm → `opt.step()` → `zero_grad()`; per-epoch accumulation window. Pattern to reuse for the surrogate-loss step. |
| checkpoint save | `:215-231` | `model.save_pretrained(ckpt)` + `tok.save_pretrained(ckpt)` + `build_adapter_manifest`/`write_manifest` (`sft/manifest.py:37,65`). **No resume, no SIGINT, no periodic eval, no best-ckpt — all must-build.** |

### Frozen decoder + spec (immutable)

| Hook | `file:line` | Signature | Note |
|---|---|---|---|
| `load_frozen_vqvae` | `tokenizer/frozen.py:48` | `load_frozen_vqvae(final_dir=None) -> (model, manifest)` | `lru_cache`d, integrity-verified. `frozen_final_dir()` (`:33`) resolves `$SLM_ARTIFACT_ROOT/tokenizer/final` → repo fallback. `codebook.npy` ships there. |
| `VQVAE.decode` | `tokenizer/model.py:266` | `decode(codes) -> np.ndarray [r,g,b,3] residual` | The live decode path; `embed_codes` (`:223`), `output_to_residual` (`:46`) back the batched decoder. |
| `ground_truth_attribute_spec_text` | `data_pipeline/attribute_spec.py:286` | `ground_truth_attribute_spec_text(row, *, bucketize=False) -> str` | **The requested spec for scoring** (grade → measured behavior serialized; refuse → refuse spec). Default `bucketize=False`. |
| `is_backed` | `data_pipeline/attribute_spec.py:305` | `is_backed(spec, mb, *, tol=1.0) -> (ok, issues)` | The interpreter↔LUT backing rule under `behavioral_agreement`. |
| `parse` / `serialize` | `data_pipeline/attribute_spec.py:239/152` | round-trippable | **SACRED — do not touch** (oracle gate + tests depend on `parse(serialize(x))==x`). |
| `measure_behavior` | `data_pipeline/behavior_vector.py:106` | `measure_behavior(lut_abs) -> dict` | Full `behavior_v2` re-measurement (the reward's physics). |

### Corpus / artifacts / adapters

- **Corpus:** `data/active_sft/active_rows.jsonl` — git-tracked, **3033 rows** (`cfg.active_rows_path`).
- **`$SLM_ARTIFACT_ROOT`** (`sft/example.py:29-30`): staged root for images + `tokenizer/final/`
  weights; falls back to cwd. Colab case trap: repo `/content/SLM`, staged `/content/slm`.
- **Base:** `models/base_resized` (gitignored; from `sft.vocab_resize`; 259 tokens added).
- **Existing adapter (GRPO init):** `models/sft_adapters/p6_twostage_d0f9c744_smokefull/` — **present
  on disk** (`adapter_config.json`, `adapter_model.safetensors`, `adapter_manifest.json`, tokenizer).
  Historical baselines `bl_a0ccbcff_*` are on HF (`ericrcwu/LUT_SLM_sft_adapters`).
  **`distill_r1` does NOT exist** — it is a *planned* run-id in `docs/collapse_fix/03` /
  `notebooks/phase3_distill.ipynb` only. Use P6 as the init; treat distill_r1 as "if/when it exists".
- **Configs:** `configs/candidate_*.json` — flat JSON overriding `SFTConfig` fields, parsed by
  `load_config`/`_load_config` via `yaml.safe_load`. P6 = `configs/candidate_two_stage.json`
  (`input_field="attribute_spec_text"`, `max_pixels=200704`). Add a `configs/candidate_grpo.json`.
- **HF:** dataset `hf://datasets/ericrcwu/LUT_SLM`; adapters repo `ericrcwu/LUT_SLM_sft_adapters`
  (upload with the write token, not the read-only `HF_TOKEN`).

---

## Must build (these hooks do NOT exist — searched, confirmed no GRPO/RL/PPO code anywhere)

1. **Per-token logprob extraction.** `generate_codes_batch` returns ids only. Need, for each
   (prompt, generated 64-code span): the summed/per-token log-prob under (a) the **old** policy at
   rollout time and (b) the **current** policy at loss time — a teacher-forced forward over
   `prompt + completion`, `log_softmax`, gather at the emitted code ids, masked to the 64-code span
   (reuse the `build_supervised_example:218-219` mask logic + `SpecialIds`). The grammar must be
   applied to logits (mask non-code logits with `make_prefix_fn`) so ratios are over the legal
   support only.
2. **Reference policy for KL.** The frozen SFT init (P6). Either a second frozen `load_eval_model`
   instance, or (cheaper) cache reference per-token logprobs once per rollout. Base 4-bit is shared;
   only the LoRA delta differs, so a single base + adapter-enable/disable toggle is an option.
3. **Trainable policy load.** `load_eval_model` returns an **inference** `.eval()` PeftModel. GRPO
   needs a trainable one (`PeftModel.from_pretrained(..., is_trainable=True)` on P6, or
   `get_peft_model` re-init from P6 weights) with the same targets/`modules_to_save` as `train.py`.
4. **Rollout buffer.** Per prompt: image, `cond_text`, `spec_text`, G completions, old-logprobs,
   rewards, group-normalized advantages. (No existing buffer type.)
5. **Group-relative advantage.** `A = (r - mean_group)/(std_group + eps_adv)` over the G samples of a
   prompt (the GRPO point — no value net). `group_advantages(rewards, *, eps=adv_eps)`.
6. **GRPO clipped-surrogate + KL loss.** Per-token importance ratio, clip `eps`, KL `beta` to the
   reference, over the 64-code assistant span. New module (analogous to `sft/soft_loss.py`).
7. **Reward-shaping wrapper.** Around `score_batch`: base `behavioral_fidelity`, collapse penalty
   (`residual_norm` too low / `dominant_share` too high — reuse the constants), optional decoded ΔE,
   **refusal on a supported row → reward 0** (reuse `score_row_samples`'s rule).
8. **Checkpoint / anytime harness.** Save adapter every `C` steps **and on SIGINT**; periodic holdout
   behavioral-fidelity eval (`generate_codes` greedy + `summarize_fidelity`); keep a **BEST** ckpt
   (not just latest — RL destabilizes / reward-hacks) alongside latest; **resume from latest**.
   `train.py` has none of this (`:120` "starts from fresh LoRA (no resume)").

---

## Locked vs methodology knobs

| Class | Knobs | Source of truth |
|---|---|---|
| **LOCKED — never vary** | `epochs`(=2), the batch triple (`per_device_batch_size·gradient_accumulation_steps==effective_batch_size`), `num_new_tokens`(=259), `base_model_id`, quant scheme (`load_in_4bit`/`bnb_4bit_quant_type`=nf4/`use_double_quant`/`compute_dtype`), `max_seq_len`(=1024), `seed`, **paths** | `AGENTS.md:48-53`; enforced in `SFTConfig.__post_init__` (`sft/config.py:98-118`) |
| **SFT-tunable (bilevel search)** | `learning_rate_lora`, `lora_r`, `lora_alpha`, `lora_dropout`, `warmup_ratio`, `max_grad_norm`, `weight_decay`, `max_pixels` (**≤ 401408**; higher end-truncates the 64 targets) | `AGENTS.md:48-53` |
| **Sanctioned input swap (not locked, not a search knob)** | `input_field` ∈ {`instruction`, `attribute_spec_text`(P6), `instruction_and_spec`} | `sft/config.py:73-76` |
| **Existing methodology knobs (OUTSIDE the locked bilevel search)** | `spec_bucketize`; `soft_label_weight`/`soft_label_tau`; `spec_augment`/`spec_jitter` | `sft/config.py:82-91` ("Phase 3 collapse fixes; NOT in the locked bilevel search") |
| **NEW GRPO methodology knobs (this project — flag exactly like the Phase-3 knobs)** | `group_size`(`G`), `clip_eps`(`ε`), `kl_beta`(`β`), `update_epochs`(`μ`), `adv_eps`(`eps_adv`), `rollout_temperature`/`rollout_top_p`, `ckpt_every`(`C`), `eval_every`(`E`), `grpo_lr`, reward-shape weights (`collapse_penalty`, `delta_e_weight`), `total_steps` | This doc — declare them methodology, outside the locked search. Canonical field names + defaults: Doc 03 §7 (optimization) + Doc 04 `GRPOConfig` (harness) + Doc 01 §7 (reward shape) |

**Note the epochs=2 / batch-triple locks are SFT-loop invariants.** GRPO is an out-of-loop
methodology experiment (like Doc 03 distillation): the *SFT locked identity* (base, quant,
`num_new_tokens`, `max_seq_len`, `seed`, paths) still holds, but GRPO's own optimization schedule
(steps, group size, KL) are new methodology knobs, not violations of the SFT locks. Call this out
explicitly in each doc, as `docs/collapse_fix/README.md:149-155` does for Doc 03.

---

## Sacred invariants (get one wrong and the result is meaningless)

1. **Holdout is sacred.** NEVER train on it; eval on it. Train pool = `supported_rows(rows,
   holdout=False)`; eval slice = `supported_rows(rows, holdout=True)`. Membership is unit-aware
   (`is_holdout_row` → `split_unit_id`, ADR 0024).
2. **No target-LUT leakage.** Reward = agreement of the generated codes with the **requested spec
   only**, exactly like `rerank_key`. Condition on `input_text_for(row, cfg.input_field)`; score
   against `ground_truth_attribute_spec_text(row)` (canonical, `bucketize=False`). Never score
   against a bucketized/augmented spec; the ΔE term is eval-only and must never enter the training
   reward's selection in a way that needs the target.
3. **Assistant-only masking over the 64-code span**, exactly as `build_supervised_example`
   (`labels[:, :n_prompt] = -100`). The surrogate loss and both logprob passes cover only those 64
   positions.
4. **Refusal on a supported row ⇒ reward 0** (reuse `score_row_samples`'s
   `{fidelity:0.0, collapsed:True}` rule); a `None` fidelity (spec asserts no measurable axis) is
   excluded, matching `summarize_fidelity`.
5. **Do not touch `attribute_spec.serialize`/`parse`.** The interpreter↔LUT round-trip is relied on
   by the oracle gate and tests.
6. **Frozen tokenizer is immutable.** Never retrain/re-gate/re-freeze it; never enable
   `eval/lut_decoder.py`; never modify `data/` or `luts/`. Decode only via `decode_codes`/
   `decode_batch`/`load_frozen_vqvae`.
7. **SFT locked identity holds** (`base_model_id`, quant, `num_new_tokens`, `max_seq_len`, `seed`,
   paths); the GRPO policy is the **same LoRA param set** as P6 (same `target_modules` +
   `modules_to_save=["embed_tokens","lm_head"]`).
8. **Grammar-constrained rollouts.** Generate under `make_prefix_fn`; store old-policy per-token
   logprobs from the same constrained pass so ratios are over the legal support.
9. **Numbers must match the shipped ruler.** Batched reward (`score_batch`) is parity-verified
   against `score_generation`; keep it that way (extend `tests/test_fast_reward.py` if you touch it).

---

## Success gate & verification

- **Headline:** free-running **greedy** behavioral fidelity on the untouched holdout climbs toward /
  past the **oracle 0.42**, measured with `generate_codes(sampling=None)` + `summarize_fidelity`
  (same path as `sft.score_tokens` greedy behavioral pass, `sft/score_tokens.py:155-183,309-322`).
  Compare directly against the **best-of-N 0.42** baseline (`eval.best_of_n.evaluate`) and the greedy
  0.159 P6 baseline.
- **Anti-reward-hacking watch (must all stay healthy while fidelity rises):** decoded ΔE
  (`decoded_delta_e_mean`), `collapse_rate` / `degenerate_rate`, code `entropy_norm`, KL-to-reference,
  and rollout entropy. A fidelity gain with rising collapse/ΔE or collapsing KL/entropy = hacking →
  reject that checkpoint (keep BEST, not latest).
- **Coverage sanity:** re-run `eval.oracle_at_n.run` on the GRPO adapter — the greedy pick should move
  up *toward* oracle@N, and oracle@N itself should not regress.
- **Parity:** the training reward on a handful of rows must equal `score_generation` on the same codes
  (guard against a shaping bug that silently changes the objective).

---

## Appendix — quick import cheatsheet

```python
# reward (hot path, batched, GPU)
from eval.fast_reward import score_batch, decode_batch
# reward (canonical / parity oracle)
from eval.behavioral_fidelity import (score_generation, score_from_lut, rerank_key,
    decode_codes, code_histogram_stats, summarize_fidelity,
    DEGENERATE_RESIDUAL_NORM, COLLAPSE_RESIDUAL_NORM, DOMINANT_SHARE_MAX, DEFAULT_TOL)
# rollouts + grammar
from sft.generate import (generate_codes, generate_codes_batch, generate_codes_for_row_batch,
    SpecialIds, make_prefix_fn, codes_from_output)
# baselines / coverage
from eval.best_of_n import best_of_n_for_row, evaluate as bestofn_evaluate
from eval.oracle_at_n import run as oracle_run, oracle_and_best, score_row_samples
# model / data / config
from sft.loader import load_eval_model                 # inference-only; trainable variant = must-build
from sft.example import (build_supervised_example, input_text_for, supported_rows,
    is_supported_materialized, surviving_code_positions, load_rows, resolve_image, artifact_root)
from sft.holdout import is_holdout_row
from sft.config import SFTConfig, load_config
# spec / decode (immutable)
from data_pipeline.attribute_spec import ground_truth_attribute_spec_text, is_backed
from data_pipeline.behavior_vector import measure_behavior
from tokenizer.frozen import load_frozen_vqvae, frozen_final_dir
```
