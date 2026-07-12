# GRPO implementation prompt (hand this to the implementer agent, verbatim)

> This file is a **self-contained, copy-pasteable prompt**. Paste everything below the line into a fresh
> implementer agent working in this repo. It builds the entire checkpointed GRPO loop from
> `docs/grpo/`. Nothing above the line is part of the prompt.

---

You are implementing **checkpointed GRPO** for the prompt→LUT generator in this repo (branch
`feat/two-stage`). The full design already exists and is authoritative — **read it first and follow it;
do not invent APIs or re-derive the math.**

## 0. Read these before writing any code (in order)

- `docs/grpo/README.md` — index, canonical terminology (symbols ↔ config fields), invariants, gate.
- `docs/grpo/00_grounding.md` — **the canonical API map.** Every reusable hook with its exact
  `file:line` signature, the "must build" list, the locked-vs-methodology knob table. **If a hook you
  need is not listed here, it is a must-build — do not assume it exists.**
- `docs/grpo/01_reward.md` — reward + group-relative advantage.
- `docs/grpo/02_rollout.md` — rollouts + old-policy per-token logprobs + buffer.
- `docs/grpo/03_grpo_loss.md` — clipped surrogate + KL loss, real-tensor pseudocode, stability.
- `docs/grpo/04_training_loop.md` — the loop + anytime checkpoint/resume/keep-BEST harness + config.
- `docs/grpo/05_eval_and_gates.md` — the greedy holdout gate, guard panel, BEST selection, early stop.
- Also skim `docs/collapse_fix/README.md` for the exposure-bias framing (this is its sequel).

Verify each signature you rely on against the live code (`grep`/open the file:line) before using it —
the docs pin line numbers but code can drift.

## 1. Goal

The generator predicts well teacher-forced (0.708) but **collapses free-running greedy** (behavioral
fidelity 0.159, 94% collapse). Sampling *covers* good trajectories (`oracle@32 ≈ 0.42`) and best-of-N
reranking already **ships 0.42**, but the greedy policy stays weak. **GRPO's job: directly optimize the
free-running behavioral reward so free-running GREEDY behavioral fidelity climbs from ~0.159 toward/past
oracle 0.42, without reward-hacking.** Policy = P6 QLoRA LoRA params; reference for KL = frozen P6 init;
4-bit NF4 base shared and frozen; group-relative advantage (no value net); clipped surrogate + KL over
the 64-code assistant span. The loop must be **anytime**: interrupting at any moment leaves a usable,
holdout-validated adapter, and you always keep a guard-vetoed BEST checkpoint.

## 2. Deliverables (exact files)

Build these; there is **no existing GRPO/RL/PPO code** in the repo (confirmed — searched). Reuse the
hooks in `00_grounding.md` verbatim; do not reimplement them.

| File | What | Doc |
|---|---|---|
| `eval/grpo_reward.py` | `shaped_rewards(codes_batch, spec_text, *, device, collapse_penalty=0.25, delta_e_weight=0.0) -> list[(reward\|None, record)]` (wraps `eval.fast_reward.score_batch`); `group_advantages(rewards, *, eps=1e-4) -> list[float]`. Pure numpy. | 01 |
| `sft/rollout.py` | `code_logprobs(model, batch) -> (logp, sel)` — the SINGLE grammar-masked teacher-forced per-token logprob extractor (used no-grad here, with-grad in the loss); `rollout_row(model, processor, row, cfg, *, G, sampling, chunk=16, device=None) -> list[RolloutSample]`; the `RolloutSample` type + a `row_id`-grouped buffer. | 02 |
| `sft/grpo_loss.py` | `grpo_loss(logp_new, logp_old, logp_ref, adv, sel, *, clip_eps, kl_beta) -> (loss, stats)`. Analogous to `sft/soft_loss.py`; torch lazy-imported. | 03 |
| `sft/grpo/config.py` | `GRPOConfig` (frozen dataclass composing `SFTConfig`) + `load_grpo_config(path) -> GRPOConfig`. | 04 |
| `configs/candidate_grpo.json` | Flat JSON: SFT half identical to `configs/candidate_two_stage.json`, plus the GRPO methodology block. | 04 |
| `sft/grpo_train.py` | The loop + checkpoint/resume/SIGINT/keep-BEST harness + model-stack loader (two adapters, one base) + rollout buffer orchestration + `holdout_greedy_eval`/`is_best`. `main()` → `train(gcfg, ...)`, single `{"grpo_summary": {...}}` line, fail-loud on a no-op. | 04, 05 |
| `tests/test_grpo_reward.py`, `tests/test_rollout.py`, `tests/test_grpo_loss.py`, `tests/test_grpo_train.py`; extend `tests/test_fast_reward.py` | Off-GPU unit tests (see §5). | 01–05 |

**Canonical config block** for `configs/candidate_grpo.json` (defaults reconciled across the docs — use
these exactly):

```json
{
  "learning_rate_lora": 0.0002, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
  "warmup_ratio": 0.03, "max_grad_norm": 1.0, "weight_decay": 0.0, "max_pixels": 200704,
  "input_field": "attribute_spec_text",

  "group_size": 8, "rollout_temperature": 0.7, "rollout_top_p": 0.9, "prompts_per_round": 8,
  "grpo_lr": 5e-6, "warmup_steps": 10, "grad_accum": 8, "update_epochs": 1,
  "clip_eps": 0.2, "kl_beta": 0.05, "adv_eps": 1e-4, "entropy_coef": 0.0, "total_steps": 500,
  "ckpt_every": 20, "eval_every": 20, "eval_limit": 64,
  "collapse_penalty": 0.25, "delta_e_weight": 0.0
}
```

## 3. Ordered build steps (bottom-up; unit-test each off-GPU before the next)

1. **Reward wrapper + advantage — `eval/grpo_reward.py`** (Doc 01 §9). `shaped_rewards` wraps
   `eval.fast_reward.score_batch` (NO `target_codes`): refusal/non-64 → `0.0` short-circuited **before**
   decode (reuse `eval.oracle_at_n.score_row_samples`'s rule); `None` fidelity → excluded (return
   `None`, drop from the group); else `r = max(0, fidelity − collapse_penalty·collapsed)`. Keep
   `delta_e_weight=0` (eval-only ΔE would leak the target). `group_advantages` standardizes each
   prompt's measurable rewards, `A_i = (r_i − mean)/(std + adv_eps)`, guarding `std=0 ⇒ A=0`. Pure
   numpy. → parity test (§5).
2. **Rollout + logprobs — `sft/rollout.py`** (Doc 02). `code_logprobs(model, batch) -> (logp, sel)`:
   `logits[:, :-1]` predict tokens `[:, 1:]`; restrict to the 256 code columns (`eval.vocab.code_token`,
   as in `sft/score_tokens.py:212`), fp32 `log_softmax`, gather the emitted code, mask to the 64-code
   span (`labels != -100` ∩ `is_code`, exactly `build_supervised_example`). `rollout_row` draws G
   grammar-constrained samples (reuse `SpecialIds`/`make_prefix_fn`/`codes_from_output`; **inline
   `generate_codes_batch`'s `.generate` body** to keep the token `sequences`), conditions on
   `input_text_for(row, cfg.input_field)`, and stores per-sample codes + old-logprobs (dropout OFF).
   Cache `ref_logprobs` from the **frozen P6 reference adapter** (`set_adapter("reference")`), NOT
   `disable_adapter()`. → logprob-alignment + conditioning-parity tests (§5).
3. **Loss — `sft/grpo_loss.py`** (Doc 03 §2–§5). Per-token ratio `exp(logp_new − logp_old)` (clamp
   log-ratio to ±20); clipped surrogate `min(ρ·A, clip(ρ,1−ε,1+ε)·A)`; k3 KL `exp(s)−s−1` with
   `s = logp_ref − logp_new` (clamped); masked token-mean over the 64 code positions:
   `loss = −((surr − kl_beta·kl)·sel).sum()/sel.sum().clamp(min=1)`. Return logging `stats` (mean ratio,
   clip fraction, mean KL, rollout entropy) for the guard panel. fp32 `log_softmax`; grad through LoRA +
   `modules_to_save` only. → `ρ≡1` first-step identity + KL≥0 + masked-mean parity tests (§5).
4. **Config — `sft/grpo/config.py` + `configs/candidate_grpo.json`** (Doc 04). `GRPOConfig` **composes**
   `SFTConfig` (do NOT add GRPO fields to `SFTConfig` — its `__post_init__` locks stay byte-identical).
   `load_grpo_config` routes SFT-known keys → `SFTConfig(**...)`, the rest → `GRPOConfig`; **unknown keys
   are a hard error.** `__post_init__` validates `group_size>=2`, `0<clip_eps<1`, `kl_beta>=0`,
   intervals `>=1`. → config round-trip + unknown-key-rejection tests.
5. **Model stack + loop + harness — `sft/grpo_train.py`** (Doc 04, 05). One shared 4-bit NF4 base +
   kbit-prep, then **two adapters**: `policy` (`PeftModel.from_pretrained(base, init_adapter,
   adapter_name="policy", is_trainable=True)`) and `reference` (`load_adapter(init_adapter,
   adapter_name="reference")`, frozen). `set_adapter`/`use_cache` toggles at the rollout↔update boundary.
   Loop = round (sample `prompts_per_round` prompts × G rollouts → reward → advantages → buffer) then
   `update_epochs` passes accumulating `grad_accum` prompt-groups per optimizer step. Reuse
   `sft/train.py`'s optim-step/flush/save/manifest patterns. Build the anytime harness: atomic `latest/`
   (three-step dir swap) every `ckpt_every` steps + a SIGINT/SIGTERM flag handler that saves at the next
   safe boundary; `trainer_state.pt` (opt + step + rng + best_summary); `resume_or_init` from `latest/`;
   separate `best/`; `eval_log.jsonl`; **fail-loud `[grpo][ABORT]` on a no-op round**.
6. **Periodic eval + guard-vetoed BEST** (Doc 05). Every `eval_every` steps, `holdout_greedy_eval` calls
   `sft.score_tokens._run_behavioral(..., sampling=None)` on `supported_rows(rows, holdout=True)` (limit
   `eval_limit`), merges the `summarize_fidelity` panel with the loop's KL/entropy/advantage-std
   telemetry, appends to `eval_log.jsonl`, and `is_best` promotes `best/` **iff** greedy fidelity is a
   new high AND every guard is healthy vs the init (collapse/degenerate/ΔE not rising, entropy not
   collapsing, KL in band). Submit `best/`, never `latest/`.

## 4. Invariants you must not violate (from `00_grounding.md` / README)

1. **Holdout is sacred** — train on `holdout=False`, eval on `holdout=True`; never roll out from or
   update on a holdout row (`sft/holdout.py:61`, `split_unit_id`).
2. **No target-LUT leakage** — reward = agreement with the **requested spec only** (condition on
   `input_text_for(row, cfg.input_field)`, score against `ground_truth_attribute_spec_text(row)`,
   `bucketize=False`; never pass `target_codes` to the training reward). ΔE is eval-only and can only
   *veto* a checkpoint, never promote one.
3. **Assistant-only 64-code span** — surrogate + both logprob passes cover only the 64 code positions,
   exactly `build_supervised_example` (`labels[:, :n_prompt] = -100`).
4. **Grammar-constrained rollouts** — sample under `make_prefix_fn`; old-logprobs over the same 256-code
   legal support; old-logprobs cached at rollout time, never recomputed post-update.
5. **Refusal on a supported row ⇒ reward 0**; `None`-fidelity row excluded from the group.
6. **Reference = frozen P6**, not the bare base (`disable_adapter()` is wrong); never updated. Same LoRA
   param set as P6 (`modules_to_save=["embed_tokens","lm_head"]`).
7. **SFT locked identity holds** — never touch `base_model_id`, quant scheme, `num_new_tokens=259`,
   `max_seq_len=1024`, `seed`, paths. All GRPO knobs are **methodology, outside the locked bilevel
   search** — the bilevel loop must never propose them.
8. **Anytime + keep BEST** — `latest/` (resume only) and `best/` (deployable) are separate dirs; submit
   `best/`.
9. **Numbers match the shipped ruler** — training reward equals `score_generation` on the same codes
   (`|Δfidelity| ≤ 0.02`, identical `collapsed`); keep `tests/test_fast_reward.py` green. Decode only via
   the frozen path; never touch `attribute_spec.serialize`/`parse` or the frozen tokenizer.

## 5. Acceptance criteria

**A. Local unit tests (off-GPU, must all pass before any Colab run):**
- **Reward parity** — on ~8 sampled train rows, `|shaped base reward − score_generation(codes,
  spec)["behavioral_fidelity"]| ≤ 0.02` and identical `collapsed` flags. (Extend
  `tests/test_fast_reward.py`.)
- **Refusal / `None` accounting** — `None` codes and a length-63 list → reward `0.0` without hitting the
  decoder; a valid sample on a non-grade spec is dropped from the group (does not change `mean`/`std`).
- **Collapse penalty** — a dominant-code sample (`collapsed=True`) with nonzero fidelity gets a negative
  advantage when a healthy sample of equal raw fidelity is in the group (the Doc 01 §5 scenario).
- **Logprob alignment** — for a greedy rollout, `argmax` of the grammar-masked step distribution equals
  the emitted code at all 64 positions; `code_logprobs` no-grad on a rollout's own sequence equals a
  from-scratch teacher-forced forward of `prompt + codes`.
- **`ρ≡1` first-step identity** — on the first inner update after a rollout (`logp_new == logp_old`),
  `ρ` within `1 ± 1e-4` and `L^clip == A` per token; with `update_epochs>1`, `ρ` then moves off 1.
- **KL ≥ 0** elementwise; **masked-mean parity** (token-mean == per-sequence mean at `|o|=64`); **grad
  locality** (grads on LoRA + `embed_tokens`/`lm_head` only, nothing in the NF4 base).
- **Config** — `load_grpo_config` round-trips `candidate_grpo.json`, rejects an unknown key, and leaves
  the `SFTConfig` locks intact.
- **Harness (stubbed, no GPU)** — `save_latest`→`resume_or_init` restores step/rng/best_summary;
  `is_best` promotes `best/` only on a guard-clean improvement; the SIGINT flag causes a clean
  save+exit; atomic `latest/` swap survives a mid-swap interrupt.

**B. Colab smoke** (A100): `python -m sft.grpo_train --config configs/candidate_grpo.json
--resized-model models/base_resized --run-id grpo_smoke --total-steps 4 --prompts-per-round 2` — rollouts
produce ≥1 gradable sample (else `[grpo][ABORT]`), loss finite, `latest/` + `eval_log.jsonl` appear, and
a SIGINT mid-run leaves a `latest/` that **re-loads and resumes** (Ctrl-C, restart, confirm `step`
continues and `best_summary` is preserved).

**C. Colab behavioral-fidelity gate** (the real acceptance): promote `best/`, then
```
python -m sft.score_tokens --config configs/candidate_grpo.json --resized-model models/base_resized \
  --adapter models/sft_adapters/grpo_<run>/best --behavioral-sampling both --behavioral-limit 0
python -m eval.oracle_at_n --config configs/candidate_grpo.json --resized-model models/base_resized \
  --adapter models/sft_adapters/grpo_<run>/best --limit 0 --n 32 --chunk 16 --temperatures 0.7,1.0
python -m eval.best_of_n --config configs/candidate_grpo.json \
  --adapter models/sft_adapters/grpo_<run>/best --limit 0 --n 16 --temperature 1.0
```
**Pass = free-running GREEDY `behavioral_fidelity_mean` beats 0.159 and moves toward/past oracle 0.42,
reported head-to-head against best-of-N ≈0.42, with every anti-hacking guard healthy across
`eval_log.jsonl`** (collapse/degenerate/ΔE not rising, entropy not collapsing, KL in band) and `oracle@32`
not regressing on `best/`. The gate is free-running greedy fidelity — the `METRIC=` teacher-forced token
accuracy is blind to collapse and is **not** the gate.

## 6. Working notes

- Match the existing code style: `sft/train.py` (loop/optim/save), `sft/soft_loss.py` (a pure
  torch-lazy-imported loss module), `eval/fast_reward.py` (batched scorer parity-checked against the
  canonical one). Emit one machine-readable summary line like `sft/train.py` does.
- Prerequisites (weights + staged artifacts + P6 adapter) and the `SLM_ARTIFACT_ROOT` case trap are in
  `docs/grpo/README.md` §Prerequisites — obtain them before the Colab steps.
- Commit only when asked; branch off `feat/two-stage`. Do not touch `data/`, `luts/`, the frozen
  tokenizer, or `attribute_spec.serialize`/`parse`.

## 7. Seams to reconcile during the build (verified non-blocking, but pin them once)

Two cross-doc descriptions are worded loosely; the file + call site are unambiguous, so just fix the
placement once and keep it consistent:

- **`group_advantages` placement.** Prose in Docs 01/02/03 variously calls it "owned by 02" / "filled by
  03" / a "must-build in eval/grpo_reward.py". **Canonical:** define it in `eval/grpo_reward.py` alongside
  `shaped_rewards` (Doc 01, pure numpy, GPU-free, unit-testable), and CALL it in Doc 04's rollout round
  (`adv = group_advantages([...], eps=gcfg.adv_eps)`), writing each `A_i` into the rollout buffer slot
  Doc 02 reserves. It is not a method on the buffer.
- **One `code_logprobs`, two sketches.** Doc 02 shows an illustrative `code_span_logprobs(..., n_prompt=,
  ids=)` (slice by prompt length); Doc 03 §3 gives the canonical `code_logprobs(model, batch) -> (logp,
  sel)` (labels-based, mirrors `sft.score_tokens`). **Build only the Doc 03 form** and use it for BOTH
  old-policy (no-grad, at rollout time) and new-policy (with-grad, in the update). For old-policy logprobs
  the rollout must construct a teacher-forced `labels` batch from each sampled sequence — the same
  `build_supervised_example` masking (`labels[:, :n_prompt] = -100`, code positions only). The two forms
  are numerically identical at the `BOS + 64 + EOS` grammar; do not ship both.
