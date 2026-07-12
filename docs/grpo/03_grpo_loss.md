# Doc 03 — GRPO objective & loss math

**Prereq:** [`00_grounding.md`](00_grounding.md) (canonical API map — do not invent hooks),
[`docs/collapse_fix/README.md`](../collapse_fix/README.md) (problem framing). Sibling docs:
[`01_reward.md`](01_reward.md) (reward shaping), [`02_rollout.md`](02_rollout.md) (rollout buffer +
group-relative advantage), this doc (the loss), [`04_training_loop.md`](04_training_loop.md)
(checkpoint/anytime harness), [`05_eval_and_gates.md`](05_eval_and_gates.md) (success gate).

## Scope

The GRPO **loss** only: given a rollout buffer of `G` grammar-constrained completions per prompt,
their old-policy per-token logprobs, and their group-relative advantages, produce the scalar to
`.backward()`. Full math (importance ratio, clipped surrogate, KL to the frozen reference, advantage
broadcast, assistant-only masking), pseudocode against **real** tensors (`model(**batch).logits`, the
`code_ids` gather from `sft/score_tokens.py`), starting hyperparameters flagged as methodology knobs,
and the numerical-stability notes that bite under 4-bit + LoRA. What upstream produces the reward and
the advantage is [`01`](01_reward.md)/[`02`](02_rollout.md); what wraps this in a step/checkpoint loop
is [`04`](04_training_loop.md).

## Goal (one paragraph)

The policy is the P6 QLoRA LoRA params (init from `models/sft_adapters/p6_twostage_d0f9c744_smokefull/`);
the reference for KL is the **frozen P6 init**; the 4-bit NF4 base is shared and frozen. For each
training prompt we sample `G` completions under the 64-code grammar (`sft.generate.make_prefix_fn`),
score each with the behavioral-fidelity reward ([`01`](01_reward.md)), turn the `G` rewards into
group-relative advantages ([`02`](02_rollout.md)), and take a clipped-surrogate + KL step over **only
the 64 code positions** (assistant-only masking, exactly as `sft.example.build_supervised_example`).
No value net — the group mean is the baseline (the GRPO point). No new SFT lock is broken: base id,
quant, `num_new_tokens`, `max_seq_len`, `seed`, and paths are untouched; the GRPO optimization
schedule is new **methodology**, flagged like the Phase-3 soft-loss knobs (`sft/config.py:82-91`).

---

## 1. Notation

For one prompt `q` (a corpus row conditioned on `input_text_for(row, cfg.input_field)`), the old
policy `π_old` (the trainable policy frozen at rollout time — [`02`](02_rollout.md)) samples a group of
`G` completions `{o_1, …, o_G}`. Each grade completion is exactly `TOKEN_COUNT = 64` code tokens
(`<lut_000..255>`); the emitted codebook indices are `o_i = (a_{i,1}, …, a_{i,64})`.

- `r_i` — the shaped scalar reward of completion `o_i` ([`01_reward.md`](01_reward.md); base =
  `behavioral_fidelity`, collapse penalty, refusal→0). Reward = agreement with the **requested spec
  only** (`ground_truth_attribute_spec_text(row)`), never a target LUT (`rerank_key` rule).
- `Â_i` — group-relative advantage of `o_i` ([`02_rollout.md`](02_rollout.md)):

  ```
  Â_i = (r_i − mean_j r_j) / (std_j r_j + eps_adv)        eps_adv ≈ 1e-4
  ```

  A **scalar per completion**, shared by all 64 of its code positions (outcome supervision — GRPO has
  no per-token credit assignment and no value net). If all `G` rewards are equal (e.g. a whole group
  collapses to reward 0), `std = 0 → Â_i = 0 →` that group contributes only the KL term. Restated here
  for completeness; it is **owned by [`02`](02_rollout.md)** — do not recompute it in the loss module.
- `π_θ` — the **current** policy (LoRA + `modules_to_save` params, with grad). `π_ref` — the **frozen**
  P6 reference (no grad). `π_old` — the policy at rollout time (no grad; its logprobs are cached in the
  buffer by [`02`](02_rollout.md)).

At a code position `t`, the only grammar-legal tokens are the 256 code ids
(`make_prefix_fn`, `sft/generate.py:52-53`), so **every logprob below is a `log_softmax` over those
256 columns only** — identical to the renormalized support HF sampled from under
`prefix_allowed_tokens_fn`. This keeps `π_old`, `π_θ`, `π_ref` on the same support and the ratios
meaningful (invariant 8, [`00`](00_grounding.md)).

---

## 2. The objective (equations)

**Per-token importance ratio** for token `t` of completion `o_i`:

```
ρ_{i,t}(θ) = exp( logπ_θ(a_{i,t} | q, a_{i,<t}) − logπ_old(a_{i,t} | q, a_{i,<t}) )
```

**Clipped surrogate** (PPO-style, advantage broadcast to every token of `o_i`):

```
L^clip_{i,t} = min( ρ_{i,t} · Â_i ,  clip(ρ_{i,t}, 1−ε, 1+ε) · Â_i )
```

**KL penalty to the reference** — the unbiased, always-non-negative k3 estimator (Schulman), per
token, so it can never push the KL negative and destabilize:

```
let  s_{i,t} = logπ_ref(a_{i,t}|·) − logπ_θ(a_{i,t}|·)
D^KL_{i,t}  = exp(s_{i,t}) − s_{i,t} − 1        ( ≥ 0 always )
```

**Per-token objective and loss.** With the assistant-only code mask `m_{i,t} ∈ {0,1}` (1 on the 64
code positions of `o_i`, 0 elsewhere — §4), maximize

```
J(θ) = (1/G) Σ_i  ( 1/|o_i| ) Σ_t  m_{i,t} · [ L^clip_{i,t} − β · D^KL_{i,t} ]
loss  = − J(θ)
```

Because every grade completion has `|o_i| = Σ_t m_{i,t} = 64` (fixed), the per-sequence
length-normalization `1/|o_i|` collapses to a plain **token-mean over the masked positions of the
group** — implement it as one masked mean (§3), no per-sample loop needed. (The GRPO/DAPO
"length-bias" debate is moot here: fixed 64-token completions.) Token-mean and
sequence-mean-of-means coincide **only when every completion in the group is a 64-code grade**; when a
group contains refusals the §2 outer `(1/G)` divides by all `G` completions while the §3 token-mean
(`n = sel.sum()`) normalizes by grade code positions only (refusals contribute 0 masked positions —
see §4), so the two differ by `(#grade / G)`. The token-mean is the intended GRPO normalization here.

**Sanity identity (parity test target).** On the **first** inner update after a rollout,
`θ = θ_old` so `ρ_{i,t} = 1`, the clip is inactive, and `∇loss = −(1/N) Σ m·Â·∇logπ_θ + β∇KL` — plain
REINFORCE with a group baseline plus KL. Assert `ρ ≈ 1.0` (to fp tolerance) on the first inner step in
a test; a ratio far from 1 there means the old-logprob cache or the grammar mask is wrong.

---

## 3. Pseudocode against real tensors

Build one teacher-forced batch per rollout (or stack the `G` rollouts) whose **assistant target is the
sampled codes**, not the gold codes — mirror `build_supervised_example` but substitute the sampled
assistant string (see §6 "must build"). Then the code-position gather is line-for-line the
`sft/score_tokens.py:247-283` pattern.

```python
import torch, torch.nn.functional as F
from eval.vocab import code_token

# ---- one-time setup (per process) ----
tok      = processor.tokenizer
code_ids = torch.tensor([tok.convert_tokens_to_ids(code_token(k)) for k in range(256)],
                        device=model.device)                 # 256 code vocab ids, codebook order
#   ↑ identical to sft/score_tokens.py:212  and  sft/train.py:141
id2idx   = torch.zeros(len(tok), dtype=torch.long, device=model.device)   # full RESIZED vocab
#   ↑ MUST be len(tok) (== 151924 here), NOT tok.vocab_size + 259. `PreTrainedTokenizerFast.vocab_size`
#     is the BASE vocab (151643) and excludes ALL added tokens — the 22 pre-existing Qwen2.5-VL specials
#     AND the 259 LUT tokens — so tok.vocab_size + 259 = 151902 under-sizes the map while max code id
#     <lut_255> = 151923, giving an IndexError (device-side assert on CUDA) on the first forward.
#     Equivalent: torch.zeros(int(code_ids.max()) + 1, ...). This id2idx map is a local convenience;
#     score_tokens.py:277 / soft_loss.py:77 instead map gold->index via a broadcast argmax with no sizing.
id2idx[code_ids] = torch.arange(256, device=model.device)    # vocab id -> codebook index (0..255)

def code_logprobs(model, batch):
    """Per-token logprob of the EMITTED code at each of the 64 code positions.
    Returns logp [B, T-1] (0.0 off the code span) and the boolean mask sel [B, T-1]."""
    logits = model(**batch).logits[:, :-1, :]                # predicts token t+1  (score_tokens:248)
    gold   = batch["input_ids"][:, 1:]                       # emitted/sampled tokens (score_tokens:249)
    labels = batch["labels"][:, 1:]                          # -100 on the prompt (build_supervised_example:219)
    sel    = (labels != -100) & torch.isin(gold, code_ids)   # the 64 code positions  (score_tokens:250-252)
    code_logits = logits[..., code_ids].float()              # [B, T-1, 256] legal support only; fp32 (§5)
    logp   = F.log_softmax(code_logits, dim=-1)              # renormalized over the 256 codes
    gidx   = id2idx[gold].clamp_(0, 255)                     # gold codebook index; garbage off-span, masked
    logp_t = logp.gather(-1, gidx[..., None]).squeeze(-1)    # [B, T-1] logprob of the emitted code
    return logp_t.masked_fill(~sel, 0.0), sel

# ---- inside the GRPO step: adv is Â_i broadcast per rollout, shape [B,1] ----
logp_new, sel = code_logprobs(policy_model, batch)           # grad flows (LoRA + lm_head/embed only)
with torch.no_grad():
    logp_old, _ = /* from the rollout buffer — cached by 02, DO NOT recompute post-update */
    logp_ref, _ = code_logprobs(ref_model, batch)            # frozen P6 (§6.2; NOT disable_adapter!)

logratio = (logp_new - logp_old).clamp(-20.0, 20.0)          # overflow guard (§5)
ratio    = torch.exp(logratio)
surr1    = ratio * adv
surr2    = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
policy   = torch.min(surr1, surr2)                           # per-token clipped surrogate

s        = (logp_ref - logp_new).clamp(-20.0, 20.0)
kl       = torch.exp(s) - s - 1.0                            # k3, per token, >= 0

per_tok  = policy - kl_beta * kl
n        = sel.sum().clamp(min=1)
loss     = -(per_tok * sel).sum() / n                        # masked token-mean (== §2 with |o|=64)
(loss / accum).backward()                                    # accumulate like sft/train.py:186
```

Notes wired to real code:
- `code_ids`, the `argmax(t+1)` alignment, and the `sel = assistant-mask & is_code` selection are
  **exactly** `sft/score_tokens.py:212,248-252`; reuse them so the training reward's code positions
  are the same 64 the scorer measures.
- `logp_old` is **read from the buffer** ([`02`](02_rollout.md)), computed once with `code_logprobs`
  right after `generate_codes_batch` and **before** any optimizer step this iteration. Never recompute
  it against the updated policy — that would make `ρ ≡ 1` and silently delete the clip.
- Restricting to `code_logits = logits[..., code_ids]` before `log_softmax` is what makes the ratio
  span the **legal grammar support only** (invariant 8). Doing `log_softmax` over the full vocab would
  put mass on illegal tokens the sampler could never emit and bias `ρ`.
- With `G` rollouts stacked into `B`, pad to equal length and rely on `sel` (mask) — do not let padding
  positions leak into the mean.

---

## 4. Assistant-only masking & refusals

The mask `m` is the **64-code span**, identical to `build_supervised_example` (`labels[:, :n_prompt] =
-100`, `sft/example.py:218-219`) intersected with `is_code` (`torch.isin(gold, code_ids)`). Both
logprob passes and the surrogate cover only these 64 positions (invariant 3, [`00`](00_grounding.md)).
The `<lut_bos>`/`<lut_eos>` control tokens and the whole prompt (incl. image tokens) are masked out.

**Refusal completions** (`<unsupported>` on a supported row) get **reward 0** (`score_row_samples`
rule, invariant 4) and therefore a below-mean `Â_i < 0`, but they have **zero code positions**, so
under the 64-code mask they contribute **no direct token gradient**. They still shape learning
indirectly: reward 0 lowers the group mean, raising the relative advantage of the successful grade
rollouts in the same group. **Open decision (flag when you implement):** if free-running refusals
persist, extend `m` for a refusal completion to cover its single emitted `<unsupported>` token so the
negative advantage directly suppresses it — a small, principled generalization of "assistant-only
span," but a deliberate departure from the literal "64-code span" wording of invariant 3. Default:
code-span-only (indirect), revisit only if `05`'s refusal rate climbs.

---

## 5. Numerical stability (these bite under 4-bit + LoRA)

- **`log_softmax` in fp32.** The NF4 base runs bf16/fp16 compute (`resolve_compute_dtype`), so raw
  logits are low-precision. Cast the 256-column `code_logits` to `.float()` before `log_softmax`
  (cheap — 256 wide). `sft/soft_loss.py:79` restricts to the 256 code columns before `log_softmax`
  (though it does not cast to fp32 — its `code_logits` at `:75` stay bf16); the `.float()` here is
  this doc's own fp32 recommendation, not something `soft_loss` already does.
- **Clamp the log-ratios** before `exp` (`clamp(-20, 20)`). Early in training `π_θ` can drift far from
  `π_old`/`π_ref` on a rare code, and `exp(large)` → `inf` → NaN loss. The PPO clip bounds the
  *surrogate* but not the *KL* term, so the clamp on `s = logp_ref − logp_new` is what actually
  protects the k3 estimator.
- **Grad through LoRA only.** After `prepare_model_for_kbit_training` + `get_peft_model`
  (`sft/train.py:106-113`), only the LoRA adapters and `modules_to_save=["embed_tokens","lm_head"]`
  have `requires_grad`; the NF4 base is frozen. Build the optimizer over
  `[p for p in model.parameters() if p.requires_grad]` (`sft/train.py:122`). `π_old`/`π_ref` under
  `torch.no_grad()`. This is the **same param set as P6** (invariant 7).
- **Gradient checkpointing on.** A GRPO step forwards prompt (long, image tokens) + 66 completion
  tokens for `G` samples through the policy **and** the reference — memory-heavy. Keep
  `cfg.gradient_checkpointing` (passed at `sft/train.py:106`); it composes with 4-bit.
- **`std=0` groups** → `Â=0` → finite (the `+eps_adv` in [`02`](02_rollout.md) guards the divide); such
  a group trains on KL only. Expected, not a bug.
- **`clip_grad_norm_` then step**, mirroring `_optim_step` (`sft/train.py:154-166`): clip →
  `opt.step()` → `opt.zero_grad()`. Reuse `cfg.max_grad_norm`.

---

## 6. What to build vs reuse

### Reuse verbatim (do NOT reimplement)
- **Grammar + id maps:** `SpecialIds`, `make_prefix_fn`, `codes_from_output` (`sft/generate.py:27-65`).
- **Rollouts:** `generate_codes_batch` (`sft/generate.py:117`) draws the `G` samples ([`02`](02_rollout.md)).
- **Code-position gather:** the `code_ids` / `argmax(t+1)` / `sel = mask & is_code` block
  (`sft/score_tokens.py:212,248-252`) and the `gold_idx` trick (`:277`) — copied into `code_logprobs`.
- **Assistant mask contract:** `build_supervised_example` (`sft/example.py:160-223`), incl. the
  exact-64 guard.
- **Reward:** `eval.fast_reward.score_batch` (hot path) / `eval.behavioral_fidelity.score_generation`
  (parity oracle) via the [`01`](01_reward.md) shaping wrapper. The loss module consumes scalar
  advantages only — it never calls the reward directly.
- **Optim step shape:** `_optim_step` (`sft/train.py:154-166`), `AdamW` over `requires_grad` params.

### Must build (new — no GRPO/RL code exists anywhere; [`00`](00_grounding.md) §"Must build")
1. **`code_logprobs(model, batch) -> (logp[B,T-1], sel[B,T-1])`** — §3. The per-token logprob
   extractor `generate_*` does not provide (they return ids only — `codes_from_output`,
   `sft/generate.py:60-65`). Used for both
   `logp_new` (grad) and `logp_old`/`logp_ref` (no grad).
2. **Reference-policy handle.** The reference is the **frozen P6 SFT init** (P6's LoRA weights) —
   **not** the bare base. ⚠ `peft`'s `model.disable_adapter()` returns the **bare NF4 base**, so it is
   the wrong reference here. Correct options: (a) load P6 as a **second frozen PEFT adapter** on the
   *shared* NF4 base and `set_adapter("ref")` vs `set_adapter("policy")` (cheapest — shares the
   expensive 4-bit base, duplicates only LoRA + `modules_to_save`); or (b) a second frozen
   `load_eval_model(cfg, base, P6)` instance (`sft/loader.py:14`, `.eval()`); or (c) cache `logp_ref`
   per rollout once (memory-light, recompute-free). Pick one in [`02`](02_rollout.md)/[`04`](04_training_loop.md);
   the loss just needs a `logp_ref` tensor.
3. **Trainable policy load.** `load_eval_model` is `.eval()`/inference-only. Load P6 with
   `is_trainable=True` (or `get_peft_model` re-init from P6) using the **same** `target_modules` +
   `modules_to_save` as `sft/train.py:108-113`.
4. **GRPO loss module — `sft/grpo_loss.py`** (this doc), analogous to `sft/soft_loss.py`: a pure
   `grpo_loss(logp_new, logp_old, logp_ref, adv, sel, *, clip_eps, kl_beta) -> (loss, stats)` that
   returns the scalar plus logging stats (mean ratio, clip fraction, mean KL, mean `|Â|`, rollout
   entropy) for the `05` anti-hacking watch. Keep torch lazy-imported (module imports without the
   `sft` extra), matching `soft_loss.py:15`.

The **assistant-string-from-sampled-codes** helper (materialize `"<lut_bos> " + " ".join("<lut_%03d>"
% c) + " <lut_eos>"`) is needed to build the teacher-forced rollout batch; re-spell it locally to keep
the module torch-free, exactly as `docs/collapse_fix/03_self_distillation.md:94-96` sanctions, and pin
it with an equality test against `scripts/materialize_target_tokens._assistant_target`.

---

## 7. Hyperparameters (starting values — ALL methodology knobs)

Flag every one of these like the Phase-3 soft-loss knobs (`sft/config.py:82-91`): they live **outside**
the locked bilevel search (`learning_rate_lora, lora_r, lora_alpha, lora_dropout, warmup_ratio,
max_grad_norm, weight_decay, max_pixels`) and outside the locked SFT identity (`base_model_id`, quant,
`num_new_tokens=259`, `max_seq_len=1024`, `seed`, paths). Add them to `configs/candidate_grpo.json` as
new flat fields; the bilevel loop must never propose them.

⚠ **Plumbing:** `sft/config.py:load_config` (`:136-137`) keeps only keys already on the frozen
`SFTConfig` dataclass (`field_names = {f.name for f in fields(SFTConfig)}`), so any flat JSON field not
present on `SFTConfig` would be **silently dropped on load**. [`04`](04_training_loop.md) resolves this by
**composing** rather than mutating: a separate frozen `GRPOConfig` wraps the `SFTConfig` and
`load_grpo_config` routes SFT-known keys to `SFTConfig` and the GRPO methodology keys to `GRPOConfig`
(unknown keys are a hard error, guarding typos). The `SFTConfig` locks stay byte-identical. The loss
module itself is unaffected (it takes `clip_eps`/`kl_beta`/the `sel` mask as function args); this is a
plumbing task owned by [`04`](04_training_loop.md).

Field names are the flat keys on `GRPOConfig` / `configs/candidate_grpo.json` ([`04`](04_training_loop.md)) — no `grpo_` prefix except `grpo_lr` (disambiguated from the SFT `learning_rate_lora`).

| Knob | Field | Start | Range | Note |
|---|---|---|---|---|
| Group size | `group_size` `G` | **8** | 4–16 | Samples/prompt for the group baseline. Coverage exists at 32 (`oracle@32=0.42`); 8 balances baseline variance vs the `G`× forward cost. Owned by [`02`](02_rollout.md). |
| Clip epsilon | `clip_eps` `ε` | **0.2** | 0.1–0.3 | Standard PPO/GRPO. Inactive when inner-epochs = 1 (see below). |
| KL beta | `kl_beta` `β` | **0.05** | 0.0–0.2 | DeepSeekMath GRPO's default is 0.04; biased slightly **higher** here (0.04–0.1) given the collapse/reward-hack risk, to keep the policy near P6. Raise if `05`'s KL-to-ref or ΔE degrade. |
| Inner epochs | `update_epochs` `μ` | **1** | 1–4 | Updates per rollout buffer. **`μ=1` ⇒ `ρ≡1` ⇒ the clip never fires** (pure baseline REINFORCE + KL). Set `μ=2–4` to actually exploit the clipped objective, at the cost of off-policyness. |
| Rollout temperature | `rollout_temperature` | **0.7** | 0.5–1.0 | Matches `oracle_at_n.run` default (t=0.7) where `oracle@32=0.42` coverage was measured. Owned by [`02`](02_rollout.md). |
| Rollout top-p | `rollout_top_p` | **0.9** | 0.8–1.0 | " |
| GRPO learning rate | `grpo_lr` | **5e-6** | 1e-6–2e-5 | **Much smaller than SFT's 2e-4** — policy-gradient steps are noisier; too high reward-hacks fast. Flat or short-warmup schedule (own schedule, methodology). |
| Adv epsilon | `adv_eps` `eps_adv` | **1e-4** | 1e-6–1e-3 | `std` divide guard ([`02`](02_rollout.md)). |
| Entropy bonus | `entropy_coef` | **0.0** | 0.0–0.01 | Optional; KL-to-ref already resists collapse. Turn on only if rollout entropy craters. |

`C` (checkpoint interval), eval interval, and total GRPO steps are also methodology knobs but belong to
the harness — [`04_training_loop.md`](04_training_loop.md). Reward-shape weights (collapse penalty, ΔE)
are [`01_reward.md`](01_reward.md).

---

## 8. Invariants (get one wrong and the number is meaningless)

1. **Assistant-only, 64-code span.** `m` = `build_supervised_example` prompt mask
   (`labels[:, :n_prompt]=-100`, `sft/example.py:219`) ∩ `is_code`. Surrogate, KL, and **both** logprob
   passes cover only those 64 positions (invariant 3, [`00`](00_grounding.md)).
2. **Legal-support ratios.** Every logprob is `log_softmax` over the **256 code columns only** (the
   `make_prefix_fn` support), so `π_old`, `π_θ`, `π_ref` share the support the sampler used. Never
   `log_softmax` over the full vocab.
3. **`logp_old` is cached at rollout time**, before any update this iteration ([`02`](02_rollout.md)).
   Recomputing it post-update forces `ρ≡1` and deletes the clip.
4. **Reference = frozen P6**, not the bare base. `disable_adapter()` gives the base and is the **wrong**
   reference (§6.2).
5. **No target-LUT leakage.** The loss consumes advantages derived from a reward scored against the
   **requested spec only** (`ground_truth_attribute_spec_text(row)`, `rerank_key` rule); ΔE is
   eval-only ([`01`](01_reward.md)). The loss module must not see a target LUT.
6. **Refusal ⇒ reward 0** (`score_row_samples`); `None`-fidelity rows excluded (matches
   `summarize_fidelity`). Refusal completions carry `Â<0` but 0 code positions (§4).
7. **Grad through LoRA + `modules_to_save` only** (same param set as P6, invariant 7); base NF4 frozen;
   `π_old`/`π_ref` under `no_grad`.
8. **SFT locked identity holds.** GRPO's schedule (steps, `G`, `β`, `ε`, `μ`, lr) is **methodology**,
   not a lock violation — state this like `docs/collapse_fix/README.md:149-155`. Do not touch
   `base_model_id`, quant, `num_new_tokens`, `max_seq_len`, `seed`, paths.
9. **Never touch `attribute_spec.serialize`/`parse`** and never decode outside the frozen path
   (`decode_codes`/`decode_batch`) — inherited, but restated because the reward runs inside the step.

---

## 9. Verification

- **Ratio-identity test (the cheapest real check).** On the first inner update after a rollout,
  `logp_new == logp_old` ⇒ assert `ρ` within `1 ± 1e-4` and `L^clip == Â` per token. A failure means the
  buffer/mask/grammar-support are misaligned. (Set `μ>1` and confirm `ρ` then *moves* off 1.)
- **KL non-negativity.** Assert `D^KL ≥ 0` elementwise (the k3 estimator guarantees it); a negative
  value means the clamp or the ref logprobs are wrong.
- **Gradient locality.** Assert `loss.backward()` populates grads on LoRA + `embed_tokens`/`lm_head`
  and on **nothing** in the NF4 base (all base params `requires_grad=False`).
- **Reward parity (guards the objective).** For a handful of rollout codes, the shaped training reward
  must equal `score_generation` on the same codes ([`01`](01_reward.md)); `score_batch` is already
  parity-verified vs `score_generation` (`tests/test_fast_reward.py`) — keep it that way.
- **Masked-mean parity.** For fixed-64 grade completions, the §3 masked token-mean must equal the §2
  per-sequence `(1/|o|)Σ` average (they coincide at `|o|=64`) — a 2-line unit test on synthetic
  tensors.
- **End-to-end (owned by [`05`](05_eval_and_gates.md)):** free-running **greedy** behavioral fidelity
  on the untouched holdout (`generate_codes(sampling=None)` + `summarize_fidelity`, the
  `sft/score_tokens.py:155-183` path) climbs toward/past **oracle 0.42** vs the best-of-N 0.42 and
  greedy-0.159 baselines, while the anti-hacking watch (decoded ΔE, collapse/degenerate rate,
  `entropy_norm`, KL-to-ref, rollout entropy) stays healthy — the loss module emits those stats (§6.4)
  each step so `04` can keep the **BEST** checkpoint, not the latest.
