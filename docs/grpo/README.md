# GRPO — checkpointed group-relative policy optimization (index)

**Audience:** an engineer/agent implementing the checkpointed GRPO loop with no prior context on this
investigation. Read this file first, then the six numbered docs in order. Each numbered doc is a
self-contained, code-referenced implementation plan; together they specify one runnable loop. This is
the **sequel** to `docs/collapse_fix/` — read that README first for the exposure-bias framing.

## The problem (validated in `docs/collapse_fix/`)

The prompt→LUT generator (Qwen2.5-VL-3B QLoRA/NF4 emitting 64 VQ code tokens `<lut_000>..<lut_255>`
that a **frozen** VQ-VAE decodes to a color LUT) predicts well **teacher-forced** (0.708) but
**collapses free-running greedy**. Measured on the P6 two-stage adapter
(`p6_twostage_d0f9c744_smokefull`), conditioned on `attribute_spec_text`, unit-aware holdout:

| decode / method | behavioral fidelity | collapse rate | role for GRPO |
|---|---|---|---|
| teacher-forced argmax (perfect gold prefix) | 0.708 | 0% | optimistic ceiling — **not** the target (answer in hand) |
| free-running **greedy** (P6) | **0.159** | 94% | **the baseline the gate must beat** |
| free-running sample t=0.7 (P6) | 0.091 | 14% | diverse but wrong-direction |
| best-of-N sample+rerank (ships today) | **≈0.42** | low | the shipped number GREEDY must reach |
| `oracle@32` (coverage ceiling) | **≈0.42** | — | the ceiling GRPO folds into the weights |
| real corpus codes (ruler's own ceiling) | ~0.89 | 0% | upper bound of the metric itself |

**Diagnosis: exposure bias**, not a broken two-stage seam and not an architecture problem. Sampling
*covers* good trajectories (`oracle@32 ≈ 0.42`) and best-of-N reranking already **ships 0.42** — but
they cost 16–32 samples + a reranker at deploy. The covered-but-unselected good trajectories are not
baked into the free-running distribution, so one deterministic pass collapses.

## Motivation — why GRPO, and why *checkpointed*

- **Close the greedy↔oracle gap directly.** GRPO optimizes the **free-running behavioral reward** — the
  true deployment objective, not a learned proxy — so **greedy** fidelity climbs from ~0.159 toward/past
  the **oracle 0.42**. One deterministic pass reaches what best-of-N needs many samples + a reranker to
  reach today. Best-of-N and self-distillation (`docs/collapse_fix/02`, `/03`) leave the greedy policy
  itself weak; GRPO moves the policy.
- **Anytime / checkpointed is the headline requirement, not a nicety.** RL destabilizes and
  reward-hacks, and Colab preempts. So the loop **saves `latest/` every `C` steps and on SIGINT/SIGTERM**,
  runs a **periodic holdout greedy eval**, and keeps a **guard-vetoed BEST checkpoint** separate from
  latest. Interrupting at any moment leaves a usable, holdout-validated adapter; you always submit
  `best/`, never `latest/`.
- **No value net (the GRPO point).** The reward is a dense, deterministic, terminal score from a cheap
  oracle (frozen decoder + numpy `behavior_v2`) computed for every rollout, so the **group mean is the
  baseline** — a critic would add parameters and instability for zero benefit, and GRPO fits the tiny
  QLoRA budget (one trainable adapter, frozen 4-bit base) far better than PPO.

## Reading order

0. **[`00_grounding.md`](00_grounding.md)** — the canonical API map. Every reusable hook with its exact
   `file:line` signature, the "must build" list, the locked-vs-methodology knob table, the sacred
   invariants. **Do not invent an API** — if a hook is not here, it is a must-build. Read first.
1. **[`01_reward.md`](01_reward.md)** — the scalar the policy optimizes: `reward(codes, requested_spec)`
   = behavioral fidelity (reuse the shipped ruler) + collapse penalty + refusal→0, then the
   group-relative advantage. No target-LUT leakage.
2. **[`02_rollout.md`](02_rollout.md)** — draw `G` grammar-constrained rollouts per prompt and capture
   the **old-policy per-token logprobs** over the 64-code span. The rollout buffer.
3. **[`03_grpo_loss.md`](03_grpo_loss.md)** — the clipped surrogate + KL-to-reference over the 64-code
   assistant span, against real tensors. Numerical-stability notes for 4-bit + LoRA.
4. **[`04_training_loop.md`](04_training_loop.md)** — wiring 01/02/03 into one loop; the
   checkpoint/resume/SIGINT/keep-BEST **anytime** harness; `GRPOConfig` + `configs/candidate_grpo.json`.
5. **[`05_eval_and_gates.md`](05_eval_and_gates.md)** — the stopping rule: the free-running greedy
   holdout gate, the reward-hacking guard panel, guard-vetoed BEST selection, early-stop/divergence.

## Architecture (one paragraph)

The **policy** is the P6 QLoRA LoRA params (init from `models/sft_adapters/p6_twostage_d0f9c744_smokefull`);
the **reference** for KL is the **frozen P6 init** (a second, frozen named adapter on the *same* 4-bit
base — **not** `disable_adapter()`, which yields the bare pre-SFT base); the **4-bit NF4 base is shared
and frozen**. Each round: sample `G` completions per prompt under the 64-code grammar
(`sft.generate.make_prefix_fn`, reused verbatim) and store old-policy per-token logprobs
[02] → score each against the **requested spec only** with the shipped behavioral-fidelity ruler
(`eval.fast_reward.score_batch`, parity-verified against `eval.behavioral_fidelity.score_generation`),
shaped by a collapse penalty, refusal→0 [01] → turn the `G` rewards into group-relative advantages
`A_i = (r_i − mean)/(std + eps_adv)` [01] → take `μ` clipped-surrogate + KL policy updates over **only
the 64 code positions** (assistant-only masking, exactly as `sft.example.build_supervised_example`)
[03] → checkpoint [04]. A periodic holdout **greedy** eval keeps a guard-vetoed BEST adapter [05]. No
value net; the group mean is the baseline.

## Canonical terminology (symbols ↔ config fields ↔ owning doc)

The docs are reconciled to this vocabulary. Field names are the flat keys on `GRPOConfig` /
`configs/candidate_grpo.json`; **all GRPO knobs are methodology, outside the locked bilevel search**
(flagged like the Phase-3 soft-loss knobs, `sft/config.py:82-91`).

| Concept | Symbol | Config field | Default | Owns default | Owns field |
|---|---|---|---|---|---|
| Group size | `G` | `group_size` | 8 | 03 §7 | 04 |
| Clip epsilon | `ε` | `clip_eps` | 0.2 | 03 §7 | 04 |
| KL weight | `β` | `kl_beta` | 0.05 | 03 §7 | 04 |
| Inner update passes | `μ` | `update_epochs` | 1 | 03 §7 | 04 |
| Advantage std-guard | `eps_adv` | `adv_eps` | 1e-4 | 01 §7 / 03 §7 | 04 |
| Rollout temperature | — | `rollout_temperature` | 0.7 | 03 §7 / 02 | 04 |
| Rollout top-p | — | `rollout_top_p` | 0.9 | 03 §7 / 02 | 04 |
| GRPO learning rate | — | `grpo_lr` | 5e-6 | 03 §7 | 04 |
| Entropy bonus | — | `entropy_coef` | 0.0 (off) | 03 §7 | 04 |
| Checkpoint interval | `C` | `ckpt_every` | 20 | 04 | 04 |
| Eval interval | `E` | `eval_every` | 20 | 04 | 04 |
| Periodic eval slice | — | `eval_limit` | 64 | 04 / 05 | 04 |
| Collapse penalty weight | — | `collapse_penalty` | 0.25 | 01 §2/§5 | 04 |
| ΔE shaping weight (eval-only) | — | `delta_e_weight` | 0.0 (off) | 01 §3 | 04 |
| Total step budget | — | `total_steps` | 500 | 04 | 04 |
| Early-stop patience / bad-window | `P` / `K` | (methodology) | — | 05 | 05 |

Canonical function/type names shared across docs: `shaped_rewards(...)` and `group_advantages(...)` [01];
`code_logprobs(model, batch) -> (logp, sel)` — the SINGLE teacher-forced per-token logprob extractor,
called under `no_grad` for the old/reference policy and under grad for the current policy — and
`rollout_row(...) -> list[RolloutSample]` [02]; `grpo_loss(logp_new, logp_old, logp_ref, adv, sel, *,
clip_eps, kl_beta) -> (loss, stats)` [03]; `RolloutGroup` (per-prompt buffer wrapper),
`holdout_greedy_eval(...)`, `is_best(...)` [04/05].

## Invariants (get one wrong and the result is meaningless)

The full list is in [`00_grounding.md`](00_grounding.md); the load-bearing ones, in one place:

1. **Holdout is sacred.** Train on `supported_rows(rows, holdout=False)`; eval on
   `supported_rows(rows, holdout=True)`. Never roll out from or update on a holdout row. Membership is
   unit-aware (`sft/holdout.py:61`, `split_unit_id`).
2. **No target-LUT leakage.** Reward = agreement of generated codes with the **requested spec only** —
   condition on `input_text_for(row, cfg.input_field)`, score against
   `ground_truth_attribute_spec_text(row)` (`bucketize=False`), never pass `target_codes` to the
   training reward. Same contract as `rerank_key`. ΔE is **eval-only** (`delta_e_weight` shapes nothing
   by default and can only *veto* a checkpoint, never promote one).
3. **Assistant-only, 64-code span.** The surrogate loss and both logprob passes cover only the 64 code
   positions, exactly as `build_supervised_example` (`labels[:, :n_prompt] = -100`).
4. **Grammar-constrained rollouts.** Sample under `make_prefix_fn`; compute old-policy logprobs over the
   **same 256-code legal support**, so the importance ratio spans legal tokens only.
5. **Refusal on a supported row ⇒ reward 0** (reuse `score_row_samples`'s rule); a `None`-fidelity row
   (spec asserts no measurable axis) is **excluded** from the group, matching `summarize_fidelity`.
6. **Reference = frozen P6 init**, never the bare base and never updated. Same LoRA param set as P6
   (`target_modules` + `modules_to_save=["embed_tokens","lm_head"]`).
7. **SFT locked identity holds.** GRPO never overrides `base_model_id`, quant scheme,
   `num_new_tokens=259`, `max_seq_len=1024`, `seed`, or paths. GRPO's optimization schedule is **new
   methodology** (like `docs/collapse_fix/README.md:149-155`'s distillation exception), not a lock
   violation; the bilevel loop must never propose a GRPO knob.
8. **Anytime + keep BEST.** `latest/` (atomic, every `C` steps + on SIGINT) is for **resume only**;
   `best/` is the guard-vetoed deployable, a **separate** directory. Submit `best/`, never `latest/`.
9. **Numbers match the shipped ruler.** The training reward equals `score_generation` on the same codes
   (within tolerance `|Δfidelity| ≤ 0.02`, identical `collapsed` flags); keep `tests/test_fast_reward.py`
   green. Decode only via the frozen path (`decode_codes`/`decode_batch`); never touch
   `attribute_spec.serialize`/`parse` or the frozen tokenizer.

## The success gate

**Headline:** free-running **greedy** behavioral fidelity on the untouched holdout climbs from ~0.159
**toward/past the oracle 0.42**, measured with `generate_codes(sampling=None)` + `summarize_fidelity`
(the `sft.score_tokens --behavioral-sampling both` path), reported **directly against the best-of-N ≈0.42
baseline** (`eval.best_of_n.evaluate`) and the greedy-0.159 baseline. Teacher-forced token accuracy (the
`METRIC=` sentinel) is **blind to the collapse we are fixing** and is *not* the gate.

**A fidelity gain is necessary, not sufficient.** BEST is `argmax(greedy_fidelity)` **subject to every
reward-hacking guard staying healthy** relative to the init: `collapse_rate` and `degenerate_rate` not
rising, eval-only `decoded_delta_e_mean` not rising (the sharpest hacking detector — the reward is blind
to un-asserted structure), `code_entropy_norm_mean` not collapsing, KL-to-reference within its band,
rollout entropy above a floor. A step that raises fidelity but trips a veto is logged (so the hack is
visible) but does not overwrite BEST. **Coverage sanity:** re-run `eval.oracle_at_n.run` on `best/` —
`oracle@32` must not regress and greedy should move up toward `oracle@1`/`best_of_N`. See
[`05_eval_and_gates.md`](05_eval_and_gates.md) for thresholds (provisional anchors, calibrate on the
first run's init reading).

## Build-order checklist (maps to the numbered docs)

No GRPO/RL/PPO code exists in the repo — every item below is a **must-build**; the existing hooks it
reuses are in `00`. Build and unit-test bottom-up (off-GPU where possible), then run the Colab gate.

- [ ] **Reward wrapper + advantage — `eval/grpo_reward.py`** [01]. `shaped_rewards(codes_batch,
      spec_text, *, device, collapse_penalty=0.25, delta_e_weight=0.0)` over `eval.fast_reward.score_batch`
      (refusal/non-64 → 0 short-circuit *before* decode; `None`-fidelity → excluded); `group_advantages(
      rewards, *, eps=adv_eps)`. Pure numpy, no GPU. **Parity test** vs `score_generation`.
- [ ] **Rollout + logprobs + buffer — `sft/rollout.py`** [02]. `code_logprobs(model, batch) -> (logp,
      sel)` (grammar-masked, teacher-forced, shared with 03); `rollout_row(...)` (G grammar-constrained
      samples for one supported row → `list[RolloutSample]` with codes + old-logprobs + optional cached
      `ref_logprobs`); the `RolloutSample`/buffer type grouped by `row_id`. Reuse `SpecialIds`,
      `make_prefix_fn`, `codes_from_output` verbatim; inline `generate_codes_batch`'s `.generate` body to
      keep the `sequences`.
- [ ] **GRPO loss — `sft/grpo_loss.py`** [03] (analogous to `sft/soft_loss.py`, torch lazy-imported).
      `grpo_loss(logp_new, logp_old, logp_ref, adv, sel, *, clip_eps, kl_beta) -> (loss, stats)`: per-token
      ratio, clipped surrogate, k3 KL to reference, masked token-mean over the 64-code span; emits the
      anti-hacking stats (mean ratio, clip fraction, mean KL, rollout entropy). fp32 `log_softmax` over
      the 256 code columns; clamp log-ratios.
- [ ] **Config — `sft/grpo/config.py` + `configs/candidate_grpo.json`** [04]. `GRPOConfig` composes
      (does not mutate) `SFTConfig`; `load_grpo_config` routes SFT keys → `SFTConfig`, GRPO keys →
      `GRPOConfig` (unknown keys are a hard error). SFT half identical to `candidate_two_stage.json`
      (`input_field="attribute_spec_text"`, `max_pixels=200704`).
- [ ] **Model stack loader** [04]. Two adapters on ONE shared 4-bit base: `policy` (trainable,
      `is_trainable=True` from P6) + `reference` (frozen, also P6). `set_adapter` toggle; `use_cache`
      flip at rollout↔update; AdamW over `requires_grad` params only.
- [ ] **Checkpoint / anytime harness — `sft/grpo_train.py`** [04]. The round/step loop; atomic `latest/`
      (three-step dir swap) every `C` steps + SIGINT/SIGTERM flag handler; `trainer_state.pt` (opt + step
      + rng + best_summary); `resume_or_init`; separate guard-vetoed `best/`; `eval_log.jsonl`; fail-loud
      on a no-op round.
- [ ] **In-loop periodic eval + guard-vetoed BEST — `holdout_greedy_eval` / `is_best`** [05]. Import
      `sft.score_tokens._run_behavioral` (greedy) on the holdout every `E` steps; merge its
      `summarize_fidelity` panel with the loop's KL/entropy/advantage-std telemetry; promote `best/` only
      on a guard-clean new high.
- [ ] **Tests (off-GPU)** — reward parity + refusal/`None` accounting (`tests/test_grpo_reward.py`,
      extend `tests/test_fast_reward.py`); logprob alignment + `ρ≡1` first-step identity + KL≥0 + masked-
      mean parity (`tests/test_rollout.py`, `tests/test_grpo_loss.py`); checkpoint round-trip / resume /
      keep-BEST / SIGINT flag with stubs (`tests/test_grpo_train.py`).
- [ ] **Colab run + gate** [05]. Smoke (`--total-steps 4`, SIGINT→resume), then the run; submit `best/`
      whose free-running greedy holdout fidelity beats 0.159 and moves toward/past oracle 0.42 with the
      guard panel healthy.

See [`IMPLEMENTATION_PROMPT.md`](IMPLEMENTATION_PROMPT.md) for a self-contained prompt that hands this
whole build to an implementer agent.

## Prerequisites (same as `docs/collapse_fix/`)

A fresh clone lacks weights and staged artifacts. Obtain before any run (automated in
`notebooks/phase1_behavioral_score.ipynb` CELLs 1–2):

1. **Corpus:** `data/active_sft/active_rows.jsonl` is git-tracked (3033 rows).
2. **Staged artifacts** (images + frozen VQ tokenizer): stage the ~9.85 GB corpus and
   `export SLM_ARTIFACT_ROOT=/content/slm`. `image_path`s and `tokenizer/final/` resolve against it.
   (Colab case trap: repo `/content/SLM`, staged `/content/slm`.)
3. **`models/base_resized`** (gitignored): `python -m sft.vocab_resize --out models/base_resized`.
4. **P6 adapter** (gitignored; init AND reference): `snapshot_download("ericrcwu/LUT_SLM_sft_adapters",
   allow_patterns=["p6_twostage_d0f9c744_smokefull/*"], local_dir="models/sft_adapters")`. `distill_r1`
   does **not** exist — use P6.

## Status / definition of done

A GRPO run whose promoted `best/` adapter's **free-running greedy** behavioral fidelity on the untouched
holdout beats the 0.159 baseline and moves **toward/past oracle 0.42**, reported head-to-head against
best-of-N ≈0.42, with `eval_log.jsonl` showing every anti-hacking guard stayed healthy across the run —
and a `latest/` that re-loads and resumes after a SIGINT. Deliverables: `eval/grpo_reward.py`,
`sft/rollout.py`, `sft/grpo_loss.py`, `sft/grpo/config.py`, `sft/grpo_train.py`,
`configs/candidate_grpo.json`, and the test modules above.
