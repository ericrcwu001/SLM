# Doc 05 — Evaluation, gates & reward-hacking guards

**Prereqs:** read [`00_grounding.md`](00_grounding.md) (the canonical API map) and
`docs/collapse_fix/README.md` (the exposure-bias framing). This doc is the *stopping rule* for the
checkpointed GRPO loop: how to know it's working, which checkpoint to keep, when to stop, and how to
catch the loop cheating. It consumes the reward defined in [`01_reward.md`](01_reward.md), the rollout
buffer in [`02_rollout.md`](02_rollout.md), the surrogate/KL logging from
[`03_grpo_loss.md`](03_grpo_loss.md), and it wires into the checkpoint/anytime harness in
[`04_training_loop.md`](04_training_loop.md).

## What we are measuring, in one paragraph

GRPO optimizes the **free-running** behavioral reward so that **greedy** decoding stops collapsing.
The gate is therefore the *free-running greedy* behavioral fidelity of the current policy on the
**untouched holdout** — the same ruler `sft.score_tokens` already prints. Teacher-forced token
accuracy (the `METRIC=` sentinel at `sft/score_tokens.py:364`) is **blind to the collapse** we are
fixing (it scores next-token given the *gold* prefix) — it is a secondary lens, **not** the gate.
The number to move is `rep["behavioral"]["behavioral_fidelity_mean"]`
(`eval/behavioral_fidelity.py:237`), and the loop must keep the checkpoint whose greedy fidelity is
highest **while the reward-hacking guards stay healthy** — never just the latest step.

## Baselines and the target (read the live numbers, don't hardcode)

Measured on the P6 two-stage adapter, conditioned on `attribute_spec_text`, unit-aware holdout
(`docs/collapse_fix/README.md`):

| Decode / method | behavioral fidelity | collapse rate | Role for GRPO |
|---|---|---|---|
| free-running **greedy** (P6) | **0.159** | 94% | **the baseline the gate must beat** |
| free-running sample `t=0.7` (P6) | 0.091 | 14% | low end of the brief's 0.087–0.159 range (`00_grounding.md:17`); exact figures per `docs/collapse_fix/README.md:19` |
| best-of-N sample+rerank (deploy today) | **≈0.42** | low | the shipped number **GREEDY must reach** |
| `oracle@32` (coverage ceiling) | **≈0.42** | — | the ceiling GRPO folds into the weights |
| ruler's own ceiling (real corpus codes) | ~0.89 | 0% | upper bound of the metric itself |

**The headline gate:** GREEDY behavioral fidelity climbs from ~0.159 toward/past **oracle 0.42** —
i.e. one deterministic pass reaches what best-of-N needs 16–32 samples + a reranker to reach today.
That is the whole point of GRPO here: bake the covered-but-unselected good trajectories into the
policy.

> ⚠ **Do not trust a hardcoded ceiling.** `eval/best_of_n.py:6` still carries a stale in-code note
> ("0.16 greedy → ~0.30") from a smaller/earlier slice; the brief's canonical figure is ~0.42
> (`oracle@32`). The commands below **re-measure `oracle@N` and best-of-N on the exact eval slice**
> you gate against, so the comparison is apples-to-apples. Gate against *that* run's numbers, not a
> figure in a comment or in this table.

## The two eval commands (run per candidate checkpoint)

Both are already built — **reuse verbatim**. Point them at the checkpoint under test and at the GRPO
config so conditioning matches P6 (`input_field="attribute_spec_text"`).

> ⚠ **Config matters.** `sft.score_tokens` defaults `--config configs/sft_default.yaml`
> (`sft/score_tokens.py:335`), which may set `input_field="instruction"`. **Always pass the GRPO
> config** so the behavioral pass conditions on the same spec text the policy was trained on. Add
> `configs/candidate_grpo.json` (flat `SFTConfig` overrides, `input_field="attribute_spec_text"`),
> per `00_grounding.md`.

### 1. Free-running behavioral fidelity — the gate (`sft.score_tokens`)

```bash
python -m sft.score_tokens \
  --config          configs/candidate_grpo.json \
  --resized-model   models/base_resized \
  --adapter         models/sft_adapters/grpo_<run>/best \
  --behavioral-sampling both \
  --behavioral-limit 0            # 0 = FULL holdout (honest final gate)
```

- `--behavioral-sampling both` runs **greedy** and **sample `t=0.7`** in one pass
  (`sft/score_tokens.py:314-325`); it emits two sub-summaries:
  - `rep["behavioral"]` — **greedy** (`sampling=None`, `generate_codes_for_row` (`sft/generate.py:100`)
    → `generate_codes` (`:68`)). **This is the gate.**
  - `rep["behavioral_sampled"]` — sample `t=0.7`, `top_p=0.9`. A diversity sanity check, not the gate.
- Each sub-summary is a `summarize_fidelity` dict (`eval/behavioral_fidelity.py:221`) with the full
  guard panel keys: `behavioral_fidelity_mean`, `collapse_rate`, `degenerate_rate`,
  `decoded_delta_e_mean`, `code_entropy_norm_mean`, `dominant_share_mean`, `residual_norm_median`
  (`:234-245`), plus `scored`/`refused` (`sft/score_tokens.py:183-184`).
- `--behavioral-limit 0` scores the whole holdout for the *final* gate; the cheap **in-loop periodic**
  eval uses the `eval_limit` knob (`GRPOConfig`, `04_training_loop.md`; `64` in `candidate_grpo.json`,
  matching the Phase-1 slice). `sft.score_tokens`'s own CLI `--behavioral-limit` default is `48`.
- The `METRIC=` line (`:364`) is the teacher-forced token accuracy — **ignore it as the gate**; it
  cannot see collapse. It stays only because the bilevel SFT contract parses it.

### 2. Coverage / ceiling — did GRPO fold oracle into greedy? (`eval.oracle_at_n`)

```bash
python -m eval.oracle_at_n \
  --config          configs/candidate_grpo.json \
  --resized-model   models/base_resized \
  --adapter         models/sft_adapters/grpo_<run>/best \
  --limit 0 --n 32 --chunk 16 --temperatures 0.7,1.0
```

- Prints the `oracle@k` curve (`k∈{1,4,8,16,32,64}`), `best_of_N`, and `best_pick_collapse_rate`
  per temperature (`eval/oracle_at_n.py:99-103`), plus a gate band line (`:134`).
- **Read two things:** (a) `oracle@32` must **not regress** below the init's (~0.42) — GRPO should
  never *destroy* coverage; (b) the greedy gate from command 1 should move **up toward** `oracle@1`
  / `best_of_N`. Success = the greedy↔oracle gap closes.
- Optional apples-to-apples best-of-N on the same slice: `python -m eval.best_of_n --config
  configs/candidate_grpo.json --adapter <best> --limit 0 --n 16 --temperature 1.0`
  (`eval/best_of_n.py:65`).

## The reward-hacking guard panel

RL will happily raise the *requested-axis* reward by degenerate means: over-committing to one code
(collapse), or producing a LUT that backs the asserted axes while drifting arbitrarily elsewhere
(ΔE blowup — the reward is blind to un-asserted structure). **Fidelity going up is necessary, not
sufficient.** Every guard below must stay healthy *as fidelity rises*; a fidelity gain bought with a
red-flagged guard is reward hacking and that checkpoint is **not eligible to become BEST**.

Guards split into two sources: the **holdout eval panel** (from `summarize_fidelity`, command 1) and
the **training-side telemetry** (KL / rollout entropy / advantage spread), which the eval harness must
pull from the loss+rollout modules (`03_grpo_loss.md`, `02_rollout.md`) since `summarize_fidelity`
cannot see them.

| Signal | Source (`file:line`) | Healthy | Red flag (veto BEST / stop) |
|---|---|---|---|
| **greedy `behavioral_fidelity_mean`** | `behavioral_fidelity.py:237` | ↑ toward 0.42 | *this is the objective, not a guard* |
| `collapse_rate` | `:238` (rule `:187`: `resid<0.01 OR dom_share≥0.5`) | ↓ from 0.94 | rises **> init** while fidelity rises, or ≥ +0.10 over running BEST |
| `degenerate_rate` | `:239` (`resid<5e-4`) | ≈0 | any climb above ~0.02 (near-identity LUTs) |
| `decoded_delta_e_mean` (eval-only) | `:242` | flat / ↓ | rises **> +1.0 ΔE** over BEST while fidelity rises = gaming requested axes, drifting from target |
| `code_entropy_norm_mean` | `:243` | stays mid/high | drops below ~0.5× init = code distribution collapsing |
| KL(policy‖ref) per token | GRPO loss (`03_grpo_loss.md`) | small, controlled | mean per-token KL beyond the `kl_beta` band (e.g. > ~3× target) = left the SFT manifold |
| rollout entropy (per-token, over legal code support) | rollout (`02_rollout.md`) | above a floor | collapses toward 0 → the G samples become identical |
| group advantage std | rollout buffer (`02_rollout.md`) | > `eps` for most prompts | ≈0 across most prompts = no learning signal / saturated reward |

**Why ΔE is the sharpest hacking detector here:** the reward scores agreement with the *requested
spec only* (the `rerank_key` no-leakage contract, `behavioral_fidelity.py:139`; ΔE is eval-only). So
the policy *can* satisfy the asserted axes with a LUT that is otherwise wrong, and only the eval-only
`decoded_delta_e_mean` (which needs the target LUT, `:157`) will catch it. Fidelity ↑ **with** ΔE ↑ is
the canonical hack. Never move ΔE into the training reward's selection (that would leak the target);
it lives strictly in the eval panel.

> **Thresholds are provisional anchors, not laws** (same posture as
> `docs/collapse_fix/README.md`'s gate bands). Calibrate the "+0.10 collapse", "+1.0 ΔE",
> "0.5× entropy", "3× KL" cut points on the first real run's init reading, and record the calibrated
> values in `configs/candidate_grpo.json`'s notes. The *shape* over eval steps matters more than any
> single crossing.

## "Best checkpoint" — a guard-vetoed argmax

**BEST is not `argmax(fidelity)`. It is `argmax(fidelity)` subject to every guard being healthy
relative to the init.** Concretely, at each periodic eval step the harness computes the guard panel
and updates BEST **iff**:

```
greedy_fidelity > best_fidelity_so_far
  AND collapse_rate      <= init_collapse_rate            + collapse_margin
  AND degenerate_rate    <= degenerate_ceiling            (~0.02)
  AND decoded_delta_e_mean<= best_delta_e_so_far          + delta_e_margin
  AND code_entropy_norm   >= 0.5 * init_entropy_norm
  AND kl_per_token         within the kl band
```

A step that raises fidelity but trips any veto is logged (so we *see* the hack) but does **not**
overwrite BEST. Keep `best/` and `latest/` as **separate** checkpoint dirs (`04_training_loop.md`):
`latest/` is only for resume; **submission is always `best/`.**

## Early-stop / divergence criteria

Greedy eval is **deterministic** (argmax) — re-running command 1 on a fixed checkpoint yields
identical fidelity, so an eval-to-eval *drop* is a real policy regression, not sampling variance.
That makes patience-based stopping trustworthy. Let `P` = eval-step patience, `K` = consecutive-bad
window (both methodology knobs).

- **Success plateau → stop, submit BEST.** Greedy fidelity ≥ ~0.42 **and** no new (guard-clean) BEST
  for `P` evals.
- **Divergence → stop, submit BEST.** Any of: KL runaway (guard red), rollout entropy collapse toward
  0, group advantage std ≈0 for `K` consecutive rollout steps (the reward saturated — no gradient
  signal), or holdout greedy fidelity falling below BEST for `K` consecutive evals. RL destabilized;
  the latest weights are worse than BEST by definition.
- **No-learning → stop, revisit knobs.** Fidelity flat within ±0.02 for many evals with KL≈0
  (policy not moving) → the GRPO knobs need tuning (`G`, `clip_eps`, `kl_beta`, rollout
  `temperature`, lr — all methodology, see `00_grounding.md`), not more steps.

## Anytime-submission story (interrupt → keep BEST)

This is the headline requirement, and eval is what makes it safe:

1. The loop runs a **periodic holdout greedy eval every `E` steps** (`E` = `eval_every`, config;
   methodology knob) and, on a
   guard-clean new high, atomically writes the adapter to `best/` (+ a `best_eval.json` with the full
   guard panel and the step). BEST is therefore **always current on disk**, independent of when you
   interrupt.
2. `latest/` is written every `C` steps **and on `SIGINT`** (`04_training_loop.md`) for **resume
   only**. A `SIGINT` may land mid-divergence, so **never submit `latest/`** — submit `best/`.
3. Therefore **interrupting at any moment yields a usable, guard-vetoed adapter** (`best/`) whose
   quality is a real holdout eval, not a hope. Resume reads `latest/` + optimizer/step + the running
   BEST-score state.

`sft/train.py` has **none** of this — no resume/SIGINT/periodic-eval/best-ckpt code anywhere; its
save block (`:215-231`) writes a single checkpoint at the end with `save_pretrained` +
`build_adapter_manifest` — so the periodic-eval / BEST-selection / SIGINT / resume machinery is a
must-build (see below and `04_training_loop.md`).

## Invariants (eval-specific; the general set is in `00_grounding.md`)

1. **Eval on the holdout only, never the train pool.** `supported_rows(rows, holdout=True)`
   (`sft/example.py:72`); membership is unit-aware via `is_holdout_row` → `split_unit_id`
   (`sft/holdout.py:61`). The GRPO loop trains on `holdout=False`; the gate reads `holdout=True`.
2. **No target-LUT leakage in the gate's selection.** Condition on `input_text_for(row,
   cfg.input_field)`; score against `ground_truth_attribute_spec_text(row)` (`bucketize=False`,
   `data_pipeline/attribute_spec.py:286`) — exactly the condition/score split in
   `eval/oracle_at_n.run` (`:85-89`) and `best_of_n_for_row` (`:52`). `decoded_delta_e` is
   **eval-only** and never enters BEST selection except as a *veto* (higher ΔE can only reject, never
   promote, a checkpoint).
3. **The gate is free-running GREEDY behavioral fidelity**, not teacher-forced token accuracy. The
   `METRIC=` sentinel (`sft/score_tokens.py:364`) is blind to collapse and is not the gate.
4. **Refusal on a supported row ⇒ fidelity 0; `None`-fidelity rows excluded.** This is already the
   behavior of `_run_behavioral` (`:174-179`) and `score_row_samples` (`eval/oracle_at_n.py:36`) —
   reuse them, don't re-account.
5. **BEST is guard-vetoed and kept separate from latest.** Latest never gets submitted.
6. **Numbers must match the shipped ruler.** The in-loop guard panel comes from `summarize_fidelity`;
   the training reward must equal `score_generation` on the same codes (parity, below). Keep the
   `fast_reward` parity test green (`tests/test_fast_reward.py:118,137`).
7. **Decode only via the frozen path** (`decode_codes`/`decode_batch`/`load_frozen_vqvae`); never
   enable `eval/lut_decoder.py`. The eval's LUTs come from the same frozen VQ-VAE as training's
   reward.

## Verification / validation (do these before trusting a gate number)

- **Reward parity (guards against a shaping bug silently changing the objective).** On a handful of
  rollout rows, assert the training reward equals canonical `score_generation(codes, spec)` within the
  established tolerance — `|Δfidelity| ≤ 0.02` and identical `collapsed` flags, the same bounds
  `tests/test_fast_reward.py:136-137` asserts for `score_batch` vs `score_generation`. If the shaped
  reward and the eval fidelity ever diverge in *sign* of movement, the shaping is hacking the metric.
- **Eval determinism.** Run command 1 twice on `best/`; `behavioral_fidelity_mean` must be identical
  (greedy = argmax). A difference means non-determinism leaked in (wrong grammar/seed/dtype) — fix
  before gating.
- **Coverage sanity.** Command 2 on `best/`: `oracle@32` ≥ init `oracle@32` (no coverage
  destruction) and the greedy gate has moved up toward `oracle@1`/`best_of_N`. If `oracle@N`
  regressed while greedy rose, the policy narrowed onto a hackable mode — reject.
- **Guard-panel monotonicity.** Persist each periodic eval's panel (`best_eval.json` + an append-only
  `eval_log.jsonl`) and confirm fidelity rose **without** any guard trending red across the run — the
  visual is the real acceptance test, per `docs/collapse_fix` style.
- **Final head-to-head.** BEST greedy fidelity on the **full** holdout (`--behavioral-limit 0`)
  reported directly against **best-of-N ≈0.42** (`eval/best_of_n.evaluate`) and **greedy 0.159**. The
  deliverable is "GREEDY now ≥ best-of-N" with the guard panel healthy.

## What to build vs reuse

**Reuse verbatim (already on `feat/two-stage`):**
- `sft.score_tokens --behavioral-sampling both` (`:305-328`) — the greedy+sample behavioral gate and
  its `summarize_fidelity` panel. For the **in-loop periodic** eval, import
  `sft.score_tokens._run_behavioral(model, processor, holdout_rows, input_field=cfg.input_field,
  bucketize=False, sampling=None, device=…)` (`:155`) directly on the current (adapter-enabled,
  `.eval()`, `no_grad`) policy — it already scores canonical spec, folds refusals to 0, and returns
  the full panel via `summarize_fidelity`.
- `eval.oracle_at_n.run` / `.main` (`:74`/`:106`) — coverage curve + `best_pick_collapse_rate`.
- `eval.best_of_n.evaluate` (`:65`) — apples-to-apples best-of-N on the eval slice.
- `eval.behavioral_fidelity.summarize_fidelity` (`:221`), `rerank_key` (`:139`), `score_generation`
  (`:202`) for the parity check, and the collapse constants (`:46-61`).
- `sft.generate.generate_codes_for_row` (`:100`, greedy) — `_run_behavioral` (`sft/score_tokens.py:163,169`)
  calls it; it wraps `generate_codes` (`:68`).

**Must build (do not exist — see `00_grounding.md` "Must build"):**
- **In-loop periodic holdout eval + guard panel assembler** — calls `_run_behavioral` (greedy) on the
  holdout every `E` steps, then merges the `summarize_fidelity` panel with the training-side KL /
  rollout-entropy / advantage-std telemetry from `02_rollout.md` / `03_grpo_loss.md` into one
  `best_eval.json` record. (`train.py` has no periodic eval.)
- **Guard-vetoed BEST selection** — the constrained-argmax rule above; writes `best/` only on a
  guard-clean new high, atomically, with a manifest (`build_adapter_manifest`/`write_manifest`,
  `sft/manifest.py:37/65`).
- **SIGINT-safe `latest/` + resume** — save every `C` steps and on `SIGINT`; resume reads
  `latest/` + optimizer/step + running-BEST-score state. (`sft/train.py` has none; its save block at
  `:215-231` writes once at the end.)
- **Reward-parity + eval-determinism check harness** — the two verification checks above, runnable
  standalone so a run can be audited after the fact.
- `configs/candidate_grpo.json` — flat JSON (`input_field="attribute_spec_text"` + the SFT-locked/tunable
  keys) plus the GRPO methodology knobs (`group_size`/`G`, `clip_eps`, `kl_beta`, `rollout_temperature`/
  `rollout_top_p`, `ckpt_every`/`C`, `eval_every`/`E`, `grpo_lr`, reward-shape weights, `P`, `K`,
  `total_steps`), loaded by `load_grpo_config` into a `GRPOConfig` that composes `SFTConfig`
  (`04_training_loop.md`). All GRPO knobs are flagged as **methodology, outside the
  locked bilevel search** — exactly like the Phase-3 soft-loss knobs
  (`docs/collapse_fix/README.md:149-155`). The SFT locked identity (`base_model_id`, quant,
  `num_new_tokens`, `max_seq_len`, `seed`, paths) still holds.
