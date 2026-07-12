# Doc 01 — Reward design (the behavioral reward + group-relative advantage)

**Audience:** the engineer implementing the GRPO loop. Read [`00_grounding.md`](00_grounding.md)
first (the canonical API map) and `docs/collapse_fix/README.md` for the problem framing. This doc
defines the scalar the policy optimizes: `reward(generated_codes, requested_spec)`, and how G
per-prompt rewards become the group-relative advantages GRPO trains on. Siblings:
[`02_rollout.md`](02_rollout.md) (draws the G rollouts + logprobs this doc scores),
[`03_grpo_loss.md`](03_grpo_loss.md) (consumes the advantages),
[`04_training_loop.md`](04_training_loop.md) (checkpoint/anytime harness),
[`05_eval_and_gates.md`](05_eval_and_gates.md) (holdout success gate + anti-hacking watch).

---

## The one-paragraph job

The generator collapses free-running (greedy behavioral fidelity 0.087–0.159, 94 % collapse) while
sampling *covers* good trajectories (`oracle@32 = 0.42`). GRPO closes that gap by directly
optimizing the **free-running behavioral reward** — the true deployment objective, not a learned
proxy — so greedy fidelity climbs toward/past the 0.42 oracle. The reward is **not a new metric**: it
is the shipped ruler (`eval/behavioral_fidelity.py`, batched via `eval/fast_reward.py`) wrapped with a
collapse penalty and the "refusal ⇒ 0" rule, then turned into a **group-relative advantage** (the
GRPO point — no value net). Every number this doc produces must match the ruler
`docs/collapse_fix/` and `eval/best_of_n.py` already ship.

---

## 1. Reward = behavioral fidelity (reuse the canonical scorer)

The base reward for one generated 64-code sequence is its **behavioral fidelity** against the
**requested** spec: the fraction of the spec's asserted axes whose re-measured behavior moves the
right way by roughly the right amount. That is exactly the number best-of-N reranks on
(`docs/collapse_fix/02`), so a GRPO win is directly comparable to the 0.42 best-of-N baseline.

**Hot path (score all G rollouts of a prompt in one batched decode)** —
`eval/fast_reward.py:242`:

```python
score_batch(codes_batch, spec, *, device=None, target_codes=None,
            tol=1.0, collapse_floor=0.01, dominant_share_max=0.5,
            final_dir=None) -> list[dict]
```

One batched frozen-VQ decode of the B candidates on `device` + a reduced `behavior_v2` measurement
(only the asserted axes). Records are drop-in for `rerank_key` and match the canonical per-sample
path within the ruler's tolerance (verified in `tests/test_fast_reward.py:118-170`:
`|Δ behavioral_fidelity| ≤ 0.02`, **identical `collapsed` flags**, and reranker-argmax agreement
≥ 95 %; batched-vs-per-sample decode agrees within `1e-5` on CPU).

**Parity oracle (per sample, the validation reference — NOT the hot path)** —
`eval/behavioral_fidelity.py:202`:

```python
score_generation(codes, spec, *, target_codes=None, final_dir=None,
                 tol=1.0, collapse_floor=0.01) -> dict
```

> ⚠ **Signature gotcha:** `score_generation` takes `collapse_floor` but **not**
> `dominant_share_max` — it forwards to `score_from_lut` (`:165`), where the `0.5` dominant-share
> threshold is baked in as the default. `score_batch` exposes `dominant_share_max` directly. Keep
> both at their defaults so the two paths agree.

Each record carries (see `00_grounding.md` "Record shape"):

| key | meaning | reward use |
|---|---|---|
| `behavioral_fidelity` | axes-backed fraction in `[0,1]`, or **`None`** for a non-grade / axis-less spec | the **base reward** |
| `collapsed` (bool) | `residual_norm < collapse_floor` **OR** `dominant_share >= dominant_share_max` | collapse penalty trigger |
| `residual_norm` (float) | RMS of the decoded residual | collapse diagnostic / logging |
| `degenerate_identity` (bool) | `residual_norm < 5e-4` (≈identity) | logging / anti-hacking watch |
| `code_stats` | `dominant_share`, `entropy_norm`, `unique_codes`, … (`:81`) | penalty + entropy logging |
| `route` | `grade` / `refuse` / `clarify` (`eval/refuse_taxonomy.py:24-26`) | only `grade` rows get a fidelity |

`behavioral_fidelity` is `n_backed / n_axes` from `behavioral_agreement` (`:103`) via the sacred
`is_backed` rule (`data_pipeline/attribute_spec.py:305`). A collapsed output moves nothing, so every
asserted axis is unbacked and fidelity → 0 — the metric already punishes the failure mode we are
fighting.

### 1a. NO target-LUT leakage (the load-bearing invariant)

Reward = agreement of the generated codes with the **requested spec only** — the same
no-leakage contract as `rerank_key` (`eval/behavioral_fidelity.py:139`), whose ΔE tie-break
**defaults to a neutral 0.0 when no target is scored** so the pick never depends on a target LUT.
Copy the condition/score split from `best_of_n_for_row` (`eval/best_of_n.py:52-62`) **verbatim**:

```python
cond_text = input_text_for(row, cfg.input_field)          # CONDITION (training parity; P6 = attribute_spec_text)
spec_text = ground_truth_attribute_spec_text(row)         # SCORE (canonical, bucketize=False)
recs      = score_batch(codes_batch, spec_text, device=device)   # NO target_codes passed
```

- **Condition** on `input_text_for(row, cfg.input_field)` (`sft/example.py:136`) — never bucketized,
  never augmented at rollout.
- **Score** against `ground_truth_attribute_spec_text(row)` (`data_pipeline/attribute_spec.py:286`,
  `bucketize=False`) — the requested behavior serialized from the interpreter, **not** the target
  codes.
- **Do NOT pass `target_codes`** into `score_batch` for the training reward. Its only effect is to
  add the `decoded_delta_e` column, which needs the target LUT — that is **eval-only** (see §3).

### 1b. Refusal / malformed ⇒ reward 0 (reuse the canonical rule)

`generate_codes_batch` returns `list[int] | None`; a refusal (`<unsupported>` emitted) is `None`,
and a truncated/over-long completion is a non-64 list. On a **supported** row either is a **miss**:

```python
if codes is None or len(codes) != 64:
    reward = 0.0          # canonical rule, eval/oracle_at_n.py:44-45
```

This is exactly `score_row_samples`'s accounting (`eval/oracle_at_n.py:36-48`) —
`{"behavioral_fidelity": 0.0, "collapsed": True}`. Do not route a malformed sample through the
decoder; short-circuit to 0.

### 1c. `None` fidelity ⇒ **excluded** (not scored as 0)

A *valid* 64-code sample whose spec asserts **no measurable axis** (route ≠ `grade`, or a grade spec
with zero gradeable axes) returns `behavioral_fidelity = None`. Such a sample is **dropped from the
group** — it does not get a reward, an advantage, or a gradient — matching `_measurable`
(`eval/oracle_at_n.py:51-55`) and `summarize_fidelity` (`eval/behavioral_fidelity.py:229`). Counting
it as 0.0 would silently deflate the objective. (In practice the train pool is
`supported_rows(holdout=False)` grade rows, so this is an edge guard, not the common path.)

---

## 2. Collapse penalty (shape the dominant-code false positive)

A pure neutral-code collapse already scores fidelity ≈ 0, so the base reward handles it. The penalty
exists for the **dominant-code** collapse: greedy can over-commit to a single *non-neutral* code
(e.g. one code filling 40/64 positions → residual RMS ~0.05), which can **spuriously back an axis or
two** and earn nonzero fidelity while being a degenerate, non-generalizing trajectory. The comment at
`eval/behavioral_fidelity.py:56-60` documents exactly this mode; `DOMINANT_SHARE_MAX = 0.5` catches
it. Without the penalty, such a sample can land *above* the group mean and get reinforced — the worst
possible outcome (§5 shows the numbers).

Shaped reward:

```python
r = fidelity - collapse_penalty * float(rec["collapsed"])      # collapse_penalty default 0.25
r = max(0.0, r)                                           # keep the [0,1] floor (matches refusal=0)
```

`rec["collapsed"]` already encodes both triggers with the shipped constants
(`eval/behavioral_fidelity.py:46-61`, re-exported by `eval/fast_reward.py:45-53`):

```
collapsed = (residual_norm < COLLAPSE_RESIDUAL_NORM=0.01)  OR  (dominant_share >= DOMINANT_SHARE_MAX=0.5)
```

**Do not re-derive the thresholds — import the constants.** A smooth variant (penalize
`max(0, dominant_share - 0.5)` and `max(0, 0.01 - residual_norm)/0.01` continuously) is a permitted
alternative if the binary flag proves too sparse; the binary flag is the default because it is the
same boundary the eval gate reports (`collapse_rate`), so training and eval agree on "collapsed".

`collapse_penalty` is a **GRPO methodology knob** (see §7 / knob table): default 0.25. Setting it 0
recovers pure fidelity.

---

## 3. Optional decoded-ΔE term (eval-only — OFF in the training reward)

`decoded_delta_e(pred_lut, target_lut)` (`eval/behavioral_fidelity.py:157`) gives node-wise CIEDE2000
against the target's own decoded LUT. It is the reranker's *tie-break* (`rerank_key`), neutral-by-
default so it never decides a deploy pick.

**In the training reward it is OFF by default (`delta_e_weight = 0`).** Computing ΔE requires decoding
`target_codes` — i.e. re-introduces a dependency on the target LUT that the no-leakage invariant
(§1a) forbids for training. Keep ΔE **eval-only** (it is a first-class anti-hacking signal in
[`05_eval_and_gates.md`](05_eval_and_gates.md): `decoded_delta_e_mean` must not rise as fidelity
climbs). If you ever enable a small ΔE shaping term for experiments, flag it loudly as leakage-
adjacent and never let it become the primary objective:

```python
if delta_e_weight > 0:                       # DISCOURAGED; leakage-adjacent, experiment-only
    r -= delta_e_weight * (rec["decoded_delta_e"]["mean"] / DE_NORM)   # needs target_codes -> eval-only
```

---

## 4. Reward range & normalization

- **Base fidelity** ∈ `[0, 1]` (a fraction of axes).
- **Shaped reward** `r` ∈ `[0, 1]` after the `max(0, ·)` floor (refusal, `None`-exclusion aside).
- **No global reward normalization is needed.** GRPO's advantage (§5) subtracts the group mean and
  divides by the group std, so it is invariant to an affine shift and self-normalizing in scale.
  Absolute reward magnitude never reaches the gradient; only the **within-group ordering and relative
  spacing** do. This is why the collapse penalty's *size relative to fidelity spread* matters (it
  must be big enough to push a dominant-code sample below the group mean) while its absolute value is
  otherwise irrelevant.

---

## 5. Group-relative advantage (the GRPO point — no value net)

For each prompt, [`02_rollout.md`](02_rollout.md) draws **G** completions under the grammar-
constrained sampler (`generate_codes_batch`, `sft/generate.py:117`, `num_return_sequences`), and this
doc scores them in one `score_batch` call → rewards `r_1..r_G`. The advantage of sample *i* is the
group-standardized reward:

```
mu    = mean(r_1..r_G)                 # over the measurable samples of THIS prompt only
sigma = std(r_1..r_G)                  # population std
A_i   = (r_i - mu) / (sigma + eps_adv) # eps_adv default 1e-4 (config field: adv_eps)
```

Every emitted code-token in sample *i*'s 64-code span carries the **same** scalar `A_i` into the
clipped surrogate ([`03_grpo_loss.md`](03_grpo_loss.md)) — GRPO assigns a sequence-level terminal
reward, no per-token credit assignment.

**Why no value network:** the reward is a **dense, deterministic, terminal** score from a cheap
oracle (frozen decoder + numpy `behavior_v2`), computed for *every* rollout. GRPO's insight is that
the **group mean is already an unbiased, low-variance baseline** for the prompt — the other G−1
samples estimate the expected return under the current policy at that prompt — so a learned `V(s)`
would add parameters, its own instability, and a second forward pass for **zero** benefit. Removing
the critic is also why GRPO fits the tiny QLoRA budget here (one trainable adapter, `.eval()` frozen
base) far more comfortably than PPO.

**Group edge cases (handle in the buffer, [`02_rollout.md`](02_rollout.md)):**
- **`sigma = 0`** (all G rewards identical — e.g. all refused → all 0, or all identical fidelity):
  `A_i = 0 / (0 + eps_adv) = 0` → **no gradient from that prompt**. Correct: a group with no reward
  variation carries no learning signal. `eps_adv` prevents the div-by-zero.
- **All-refused group:** all `r_i = 0` ⇒ `sigma = 0` ⇒ zero advantage. The prompt is wasted;
  [`04_training_loop.md`](04_training_loop.md) may resample or log a refusal rate, but the reward math
  is safe.
- **`None`-fidelity samples:** excluded *before* `mu`/`sigma` (§1c). Compute the group stats over the
  measurable subset only; excluded samples get no advantage and no gradient.

### Worked numeric example (G = 6, one grade prompt asserting 4 axes)

Spec `route=grade warmer=+2.3 muted=+2.0 matte=+2.5 contrast=-1.0` → 4 gradeable axes, so
`behavioral_fidelity ∈ {0, .25, .5, .75, 1}`. `collapse_penalty = 0.25`, `eps_adv = 1e-4`.

| # | outcome | `behavioral_fidelity` | `collapsed` | shaped `r` |
|---|---|---|---|---|
| 1 | backs 3/4 axes, healthy | 0.75 | no | 0.75 |
| 2 | backs 2/4, healthy | 0.50 | no | 0.50 |
| 3 | backs 1/4, healthy | 0.25 | no | 0.25 |
| 4 | **refused** (`None` codes) | — | — | **0.00** (§1b) |
| 5 | dominant code 40/64 backs 2/4 | 0.50 | **yes** | `max(0, .50−.25)=` **0.25** |
| 6 | neutral collapse, backs 0 | 0.00 | yes | `max(0, 0−.25)=` **0.00** |

`mu = (0.75+0.50+0.25+0+0.25+0)/6 = 0.2917`; `sigma ≈ 0.2668`; `sigma+eps_adv ≈ 0.2669`.

| # | `A_i = (r_i − mu)/(sigma+eps_adv)` | effect |
|---|---|---|
| 1 | **+1.72** | strongly reinforced |
| 2 | **+0.78** | reinforced |
| 3 | −0.16 | mildly suppressed |
| 4 | −1.09 | refusal strongly suppressed |
| 5 | −0.16 | dominant-code collapse suppressed |
| 6 | −1.09 | neutral collapse strongly suppressed |

**What the collapse penalty bought:** sample 5 had raw fidelity 0.50 — *tied with the healthy sample
2*. Without the penalty its reward would be 0.50, `mu` would rise to 0.3333 (`sigma ≈ 0.2764`), and
sample 5 would get `A ≈ +0.60` — tying healthy sample 2, which also lands at +0.60 without the
penalty — i.e. GRPO would **reinforce a degenerate dominant-code trajectory just as hard as the
healthy one**. The 0.25 penalty flips it to −0.16, so the policy is pushed *away* from the collapse
mode even when it accidentally scores. This is the whole reason the penalty exists.

---

## 6. Reward invariants

1. **No target-LUT leakage.** Reward is agreement with the **requested** spec only
   (`ground_truth_attribute_spec_text(row)`, `bucketize=False`), condition on
   `input_text_for(row, cfg.input_field)`. Never pass `target_codes` to the training reward; ΔE stays
   eval-only. Same contract as `rerank_key`. (§1a, §3)
2. **Refusal / non-64 on a supported row ⇒ reward 0.** Reuse `score_row_samples`'s rule
   (`eval/oracle_at_n.py:44-45`); short-circuit before decode. (§1b)
3. **`None` fidelity ⇒ excluded** from the group (no reward/advantage/gradient), matching
   `_measurable` / `summarize_fidelity`. Never scored as 0. (§1c)
4. **Parity with the shipped ruler.** The training reward on any set of codes must match
   `score_generation` on the same codes (up to the collapse-penalty/floor shaping this doc adds).
   `score_batch` agrees with `score_generation` to within the ruler's tolerance
   (`tests/test_fast_reward.py:118-170`: `|Δ behavioral_fidelity| ≤ 0.02`, identical `collapsed`
   flags, reranker-argmax agreement ≥ 95 %); if you touch either path, extend that test. (§8)
5. **Determinism.** `score_generation`/`score_batch` are deterministic given codes (frozen decoder +
   numpy `behavior_v2`, no sampling) — the reward is a pure function of `(codes, spec)`, same inputs
   same reward for a given path. But the batched and per-sample paths are *different decode
   implementations* (`decode_batch` reimplements `embed_codes`+decoder+`output_to_residual`;
   `decode_codes` goes through `VQVAE.decode`), so cross-path they agree only to `~1e-5` on CPU —
   occasionally enough to flip a near-boundary `is_backed` and shift fidelity, bounded at `≤ 0.02`
   (invariant 4). A CUDA batched decode adds the same order of float noise. Validate parity on CPU
   *within tolerance* (§8).
6. **Holdout-safe.** The reward reads only `codes` + the row's requested spec + (never) its target.
   It touches no holdout state. The *training* reward is computed only over
   `supported_rows(holdout=False)`; the identical function scores the holdout **for eval only**
   ([`05_eval_and_gates.md`](05_eval_and_gates.md)). Holdout membership is unit-aware
   (`sft/holdout.py:61`, `split_unit_id`).
7. **Import the collapse constants; never re-derive them.** `COLLAPSE_RESIDUAL_NORM=0.01`,
   `DOMINANT_SHARE_MAX=0.5`, `DEGENERATE_RESIDUAL_NORM=5e-4`, `DEFAULT_TOL=1.0` from
   `eval/behavioral_fidelity.py:46-61`. So training's "collapsed" == the eval gate's `collapse_rate`.
8. **Never touch `attribute_spec.serialize`/`parse`** (`data_pipeline/attribute_spec.py:152/239`) or
   the frozen decoder. Decode only via `decode_batch` / `decode_codes` / `load_frozen_vqvae`.

---

## 7. Reward knobs are GRPO methodology knobs (not the locked bilevel search)

Exactly as `docs/collapse_fix/README.md:149-155` frames Doc 03's distillation: the **SFT locked
identity holds** (`base_model_id`, quant/NF4, `num_new_tokens=259`, `max_seq_len=1024`, `seed`,
paths; `sft/config.py:98-118`) and the GRPO policy is the **same LoRA param set as P6**
(`sft/train.py:108-113`). But GRPO's own optimization schedule is **new methodology, outside the
locked bilevel hill-climb** — declare it as such, do not treat it as a lock violation. The reward's
share of those knobs:

| knob | default | meaning |
|---|---|---|
| `collapse_penalty` | 0.25 | collapse penalty weight (§2). 0 ⇒ pure fidelity |
| `delta_e_weight` | **0** | ΔE shaping weight (§3). Keep 0 — leakage-adjacent |
| `G` (`group_size`) | (see 02/03) | group size — how many rollouts define `mu`/`sigma` |
| `eps_adv` (`adv_eps`) | 1e-4 | advantage denominator floor (§5) |
| `tol`, `collapse_floor`, `dominant_share_max` | 1.0 / 0.01 / 0.5 | **inherited from the ruler — do not vary** (invariant 7) |

(`clip_eps`, `kl_beta`, rollout `temperature`/`top_p`, `C`, GRPO `lr`, eval interval, total steps live
in the sibling docs' knob tables.)

---

## 8. Verification

- **Parity (must pass before any run):** on ~8 sampled train rows, assert
  `abs(score_batch([codes], spec, device="cpu")[0]["behavioral_fidelity"] -
  score_generation(codes, spec)["behavioral_fidelity"]) <= 0.02` and **exact** equality for
  `collapsed` (`... ["collapsed"] == ...["collapsed"]`); optionally assert the reranker-argmax pick
  agrees. Do **not** assert exact fidelity equality — the two paths use different decode
  implementations (~1e-5 diff that can shift fidelity by up to 0.02, invariant 5). This mirrors
  `tests/test_fast_reward.py:118-138` and guards against a shaping bug silently changing the
  objective. Extend that test.
- **Refusal accounting:** feed `None` and a length-63 list; assert reward `0.0` and that neither hits
  the decoder.
- **`None`-exclusion:** a valid sample on a non-grade spec must be dropped from the group (assert it
  does not change `mu`/`sigma`), not scored 0.
- **Penalty behaves:** construct a dominant-code sample (one code × 40, arbitrary others) with
  nonzero fidelity; assert `collapsed=True` and that its advantage is negative when a healthy sample
  of equal raw fidelity is in the group (the §5 scenario as a unit test).
- **Determinism:** score the same codes twice; identical reward on CPU.
- **End-to-end gate** (in [`05_eval_and_gates.md`](05_eval_and_gates.md)): greedy behavioral fidelity
  on the untouched holdout climbs toward/past **oracle 0.42** (`generate_codes(sampling=None)` +
  `summarize_fidelity`), while the anti-hacking watch — `decoded_delta_e_mean`, `collapse_rate`,
  `degenerate_rate`, `code_entropy_norm_mean`, KL-to-ref, rollout entropy — stays healthy. A fidelity
  gain with rising collapse/ΔE or collapsing entropy/KL = reward hacking ⇒ reject that checkpoint
  (keep BEST, not latest).

---

## 9. What to build vs reuse

**Reuse verbatim (import, do NOT reimplement):**

- `eval.fast_reward.score_batch` (`:242`) — hot-path reward for the G rollouts.
- `eval.behavioral_fidelity.score_generation` (`:202`) — parity oracle.
- `eval.behavioral_fidelity.rerank_key` (`:139`) — the no-leakage contract this reward obeys.
- `eval.behavioral_fidelity.code_histogram_stats` (`:81`), `summarize_fidelity` (`:221`) —
  diagnostics / holdout aggregate.
- Collapse constants `COLLAPSE_RESIDUAL_NORM`, `DOMINANT_SHARE_MAX`, `DEGENERATE_RESIDUAL_NORM`,
  `DEFAULT_TOL` (`eval/behavioral_fidelity.py:46-61`).
- `data_pipeline.attribute_spec.ground_truth_attribute_spec_text` (`:286`) — the spec to score
  against; `sft.example.input_text_for` (`:136`) — the text to condition on; the condition/score
  split copied from `eval.best_of_n.best_of_n_for_row` (`:52-62`).
- `eval.oracle_at_n.score_row_samples`'s refusal rule (`:36-48`) — refusal/non-64 ⇒ reward 0.

**Build (thin; these do not exist — confirmed no RL code in the repo):**

1. **Reward-shaping wrapper** around `score_batch` (grounding "must build" #7). A small new module
   (e.g. `eval/grpo_reward.py`), analogous to how `fast_reward` wraps `behavioral_fidelity`:
   ```python
   def shaped_rewards(codes_batch, spec_text, *, device, collapse_penalty=0.25, delta_e_weight=0.0):
       # Partition BEFORE decode: a refusal/malformed sample (None or len != 64) must never reach
       # score_batch's decoder. Score only the valid-64 subset, then re-assemble in input order.
       valid_idx  = [i for i, c in enumerate(codes_batch) if c is not None and len(c) == 64]
       recs_valid = (score_batch([codes_batch[i] for i in valid_idx], spec_text, device=device)  # NO target_codes
                     if valid_idx else [])
       rec_by_idx = dict(zip(valid_idx, recs_valid))
       out = []
       for i in range(len(codes_batch)):
           rec = rec_by_idx.get(i)
           if rec is None:                              # refusal / malformed -> reward 0, never decoded
               out.append((0.0, {"behavioral_fidelity": None, "collapsed": True, "refused": True}))
               continue
           f = rec["behavioral_fidelity"]
           if f is None:                                # non-grade / axis-less -> exclude from advantage
               out.append((None, rec)); continue
           r = f - collapse_penalty * float(rec["collapsed"])
           out.append((max(0.0, r), rec))
       return out                                       # (reward|None, record) per sample, input order
   ```
   (The valid-64 partition IS the short-circuit — malformed codes never reach the decoder. `delta_e_weight`
   is accepted for interface symmetry but deliberately NOT applied to the reward: ΔE stays
   eval-only/veto-only, Invariant 8.)
2. **Group-relative advantage** — `group_advantages(rewards, *, eps=adv_eps)` (grounding "must build"
   #5): standardize the measurable rewards of each prompt's G samples,
   `A_i = (r_i − mu)/(sigma + eps_adv)`, exclude `None`, guard `sigma = 0 ⇒ A = 0`. Lives in the rollout
   buffer ([`02_rollout.md`](02_rollout.md)); consumed by [`03_grpo_loss.md`](03_grpo_loss.md).

Both are pure numpy over `score_batch` records — no torch, no model, unit-testable without a GPU
(same discipline as `eval.oracle_at_n.oracle_and_best`).
