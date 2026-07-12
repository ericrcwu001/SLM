# Doc 02 — Rollout & generation (drawing the G grammar-constrained samples)

**Prereq:** [`00_grounding.md`](00_grounding.md) (the canonical API map — do not invent hooks) and
`docs/collapse_fix/README.md` (the exposure-bias framing). **Siblings:** [`01_reward.md`](01_reward.md)
(scores the codes this doc produces), [`03_grpo_loss.md`](03_grpo_loss.md) (recomputes per-token
logprobs *with* grad and consumes the buffer), [`04_training_loop.md`](04_training_loop.md) (calls the
rollout each step; checkpoint/resume), [`05_eval_and_gates.md`](05_eval_and_gates.md) (greedy holdout
gate).

## Scope (one paragraph)

For each training prompt, draw **G** free-running, grammar-constrained rollouts, and capture — for each
rollout — the 64 committed codebook indices **and the old-policy per-token logprob over the 64-code
assistant span**. Generation reuses `sft/generate.py`'s grammar helpers verbatim
(`make_prefix_fn`/`SpecialIds`/`codes_from_output`); `generate_codes_batch`'s `.generate` body is
**inlined** (it returns codes only and discards the full token `sequences` the buffer needs). The logprob
capture does **not** exist yet and is the one real build in this doc (`generate_*` returns codebook
indices only — no logprobs/scores — via `codes_from_output`, `sft/generate.py:97`/`:152`). The output is
a **rollout buffer** of `(prompt, codes, old_logprobs, reward-slot, advantage-slot)` records that Doc 01
fills with rewards and Doc 03 turns into a clipped surrogate + KL loss.

---

## What already exists — reuse verbatim (do NOT reimplement)

| Piece | `file:line` | Rollout use |
|---|---|---|
| `generate_codes_batch(model, processor, *, image, text, n, sampling, chunk=16, max_new_tokens=68, device=None) -> list[list[int]|None]` | `sft/generate.py:117` | Draw G samples for one prompt — but it returns codebook indices only and discards the token `sequences` the buffer needs, so **inline its `.generate` body** (`sft/generate.py:133-153`) rather than calling it. `make_prefix_fn`/`SpecialIds`/`codes_from_output` are the truly-verbatim reuses. |
| `generate_codes_for_row_batch(...)` | `sft/generate.py:157` | Same, conditioned straight from a corpus row via `input_text_for`. |
| `generate_codes(..., sampling=None, ...)` | `sft/generate.py:68` | Greedy — **eval gate only** (Doc 05), never a rollout (G identical rows). |
| `SpecialIds(tokenizer)` → `.bos .lut_eos .unsupported .model_eos .codes[256] .id_to_index` | `sft/generate.py:27` | Token-id ↔ codebook-index map for the logprob gather + grammar mask. |
| `make_prefix_fn(prompt_len, ids)` | `sft/generate.py:39` | The 64-code grammar; batch-agnostic, survives `num_return_sequences`. |
| `codes_from_output(output_row, prompt_len, ids) -> list[int]|None` | `sft/generate.py:60` | Map a generated id row → 64 codebook indices; `None` on refusal. |
| `input_text_for(row, cfg.input_field)` | `sft/example.py:136` | **Conditioning text (training parity)** — pass `cfg.input_field` (P6 = `attribute_spec_text`), no bucketize, no augment. |
| `ground_truth_attribute_spec_text(row, bucketize=False)` | `data_pipeline/attribute_spec.py:286` | **The requested spec Doc 01 scores against** — a *different* call from the conditioning text. |
| `build_supervised_example(...)` mask logic (`labels[:, :n_prompt] = -100`) | `sft/example.py:218-219` | The assistant-only boundary the logprob gather reuses (same `n_prompt`). |
| `supported_rows(rows, holdout=False)` | `sft/example.py:72` | The **train** pool to roll out on (holdout excluded — sacred). |

The condition/score split is already implemented correctly in `eval/best_of_n.py:52-62`
(`best_of_n_for_row`): condition on `input_text_for(row, input_field)`, score on
`ground_truth_attribute_spec_text(row)`. **Copy that split exactly** — it is the no-target-leakage
contract (invariant 2).

---

## The 64-code grammar (recap)

Every rollout is constrained by `make_prefix_fn` (`sft/generate.py:39-57`) to exactly one of:

```
grade:   <lut_bos> <lut_NNN>×64 <lut_eos> </model_eos>     # the 64 committed codes
refuse:  <unsupported> </model_eos>                        # a routing refusal
```

Position 0 → `{bos, unsupported}`; after `bos`, the 64 code positions → the 256 code ids; then
`lut_eos`; then model EOS. The grammar is **batch-agnostic** (it ignores `batch_id` and slices on a
fixed `prompt_len`), which is why one `generate` call with `num_return_sequences=G` stays correct — it
is reused untouched. This constraint matters twice: (a) at sampling time (only legal tokens are drawn),
and (b) at logprob time — the old-policy logprob must be computed over the **same legal support** so the
importance ratio in Doc 03 spans legal tokens only (invariant 8).

Because all rollout prompts are **supported** training rows, a `<unsupported>` rollout is a *mistake* →
reward 0 (invariant 4); see "Refusals & edge cases".

---

## Conditioning parity — the prompt half must match training byte-for-byte

Two distinct strings per row, never conflated:

- **Conditioning (the prompt the model sees):** `input_text_for(row, cfg.input_field)` —
  `attribute_spec_text` for P6, `bucketize=False`, `augment_rng=None` at rollout. This is what
  `generate_codes_for_row_batch` already does internally (`sft/generate.py:168`), so the rollout prompt
  is byte-identical to the trainer's prompt half (`build_supervised_example` → `input_text_for`,
  `sft/example.py:177`).
- **Scoring (Doc 01's reward):** `ground_truth_attribute_spec_text(row)` — the canonical requested
  spec, **not** the conditioning text. When `input_field == "attribute_spec_text"` these coincide; for
  `instruction`/`instruction_and_spec` they differ, and the reward must still use the canonical spec.

Never condition on a bucketized/augmented spec, and never score on the raw input text. The ΔE term is
eval-only (needs a target LUT) and never enters the training-reward selection (invariant 2).

---

## Sampling & group size G

- Rollout **must sample** — `generate_codes_batch` asserts this in spirit (`sft/generate.py:123`:
  greedy with `n>1` returns `n` identical rows). Group-relative advantage needs *diversity*: if all G
  samples are identical, `std_group = 0` and the advantage `(r - mean)/(std + eps_adv)` is meaningless.
- **`G ≥ 2`** (practically 8–16). Reuse the coverage knobs already validated on this model:
  `oracle_at_n` uses `t=0.7, top_p=0.9` (`eval/oracle_at_n.py:74`); `best_of_n` ships `t=1.0, top_p=0.9`
  (`eval/best_of_n.py:103`). Start there.
- `G`, rollout `temperature`, and `top_p` are **GRPO methodology knobs**, flagged exactly like the
  Phase-3 soft-loss knobs (`sft/config.py:82-91`) — they live in `configs/candidate_grpo.json`, are
  **outside** the locked bilevel search, and do **not** touch the SFT locked identity (`base_model_id`,
  quant, `num_new_tokens`, `max_seq_len`, `seed`, paths). State this in the config's provenance block
  (as `docs/collapse_fix/README.md:149-155` does for the distillation exception).

---

## Batching: G samples per prompt, looped over the P prompts in the step

Roll out **per prompt** with `generate_codes_batch(..., n=G, chunk=16)` — one `.generate` call yields up
to `chunk` samples via `num_return_sequences`, `ceil(G/chunk)` calls per prompt (`sft/generate.py:146-153`).
Loop that over the P prompts of the training minibatch. **Do not** try to fuse P×G into a single
`.generate` call:

- `make_prefix_fn` is built for **one** `prompt_len` (`sft/generate.py:39`); a P-prompt batch has
  varying prompt lengths, so a fused call would need a batch-id-aware grammar and left-padding — new,
  fragile code for no throughput win over per-prompt `num_return_sequences` batching.
- Qwen2.5-VL prompts carry **per-prompt image features** (`pixel_values`, `image_grid_thw`); mixing
  images in one padded batch is exactly what the per-prompt path avoids.

The per-prompt call already amortizes the expensive part (the shared image/prompt prefix is encoded once
per `.generate`, then G continuations share the KV-cache of that prefix).

---

## Old-policy per-token logprobs — **MUST BUILD** (the crux of this doc)

`generate_codes_batch` returns codebook indices only — no logprobs/scores (`codes_from_output`,
`sft/generate.py:152`). GRPO needs, per rollout, the
log-probability the **policy that generated it** assigned to each emitted code token, over the 64-code
assistant span. Build **one** function and use it two ways: under `torch.no_grad()` here for the *old*
policy (stored in the buffer), and *with* grad in Doc 03 for the *current* policy (the ratio numerator).

### Why a dedicated teacher-forced forward (not `generate`'s `scores`)

You *can* ask `generate(..., return_dict_in_generate=True, output_scores=True)` for per-step scores, but
those scores are post-**warper** (temperature/top-p) and their exact contents are transformers-version
dependent. The GRPO ratio must be over the model's **grammar-masked, un-warped** distribution (temp 1,
no nucleus truncation), and it must be computed by the *identical* code path for old and current policy
or the ratio is biased. So compute logprobs from a clean forward and gather at the grammar-legal
support. (If you later want the `output_scores` shortcut as a speed lever, gate it behind a parity test
against this function, mirroring how `eval/fast_reward.py` is parity-checked against the canonical
scorer in `tests/test_fast_reward.py`.)

### Alignment (get this exactly right)

For a grade rollout the full sequence is
`prompt[0..n_prompt-1]` then `BOS(@n_prompt) c0(@n_prompt+1) … c63(@n_prompt+64) LUT_EOS(@n_prompt+65) EOS`.
`n_prompt` is the prompt length used by `generate` (`sft/generate.py:141`), identical to
`build_supervised_example`'s `n_prompt` (`sft/example.py:198`). Logits at position `i` predict token
`i+1`, so code `c_j` (absolute index `n_prompt+1+j`) is predicted by **logits at position `n_prompt+j`**,
for `j = 0..63` (the BOS position through the 63rd code position). The 64 code logprobs therefore come
from logits at positions `[n_prompt .. n_prompt+63]`.

### The function (new — e.g. `sft/rollout.py`)

> **Canonical name = `code_logprobs`.** There is ONE logprob extractor, shared with
> [`03_grpo_loss.md`](03_grpo_loss.md) §3: `code_logprobs(model, batch) -> (logp, sel)`, batched over
> the sequences in `batch`, returning the per-token logprob (0.0 off the code span) and the boolean
> 64-code mask. This doc calls it under `torch.no_grad()` for the **old** policy; Doc 03 calls the
> **same** function under grad for the **current** policy. The per-sequence sketch below
> (`code_span_logprobs`, returning a bare `[64]` for one sequence) is that operation shown one sequence
> at a time to make the `n_prompt+j` alignment explicit — the shipped implementation is the single
> batched `code_logprobs`; the loop's rollout path just calls it under `no_grad` (Doc 04 references
> `code_logprobs`).

```python
# sft/rollout.py  (NEW)  — per-sequence alignment sketch of the canonical batched `code_logprobs`
import torch
from sft.generate import SpecialIds, make_prefix_fn, codes_from_output, TOKEN_COUNT

def code_span_logprobs(model, forward_inputs, *, n_prompt, ids: SpecialIds):
    """Per-token logprob of the emitted code tokens, over the grammar-legal (256-code) support.

    `forward_inputs` is the processor output for prompt+completion (input_ids, attention_mask,
    pixel_values, image_grid_thw) — for a rollout, just the row returned by `generate`, which already
    IS prompt+completion, plus the prompt's vision tensors. Returns a [n_code] tensor (n_code == 64 for
    a grade). Wrap in torch.no_grad() for the OLD policy (this doc); Doc 03 calls the SAME body under
    grad for the CURRENT policy so old/new logprobs share one definition."""
    out = model(**forward_inputs)                       # logits: [1, L, V]
    logits = out.logits[0]
    code_cols = torch.as_tensor(ids.codes, device=logits.device)       # 256 legal code ids (col k == codebook idx k)
    emitted = forward_inputs["input_ids"][0, n_prompt + 1 : n_prompt + 1 + TOKEN_COUNT]  # c0..c63 token ids
    steps = logits[n_prompt : n_prompt + TOKEN_COUNT][:, code_cols]     # [64, 256] restricted to legal support
    logp = torch.log_softmax(steps, dim=-1)                            # grammar-masked log-probs
    cols = torch.as_tensor([ids.id_to_index[int(t)] for t in emitted], device=logits.device)
    return logp.gather(1, cols[:, None]).squeeze(1)                    # [64]

@torch.no_grad()
def rollout_row(model, processor, row, cfg, *, G, sampling, chunk=16, device=None):
    """Draw G grammar-constrained samples for ONE supported training row; return G buffer records."""
    from sft.example import input_text_for, resolve_image
    ids = SpecialIds(processor.tokenizer)
    image = resolve_image(row["image_path"])
    cond  = input_text_for(row, cfg.input_field)                       # training-parity conditioning
    # inline generate_codes_batch's `.generate` body (it returns codes only, dropping the sequences):
    #   run generate with return_dict_in_generate to get `sequences` (prompt+completion), then a
    #   teacher-forced forward over each sequence for clean logprobs. (generate_codes_batch would give
    #   the codes directly but discard the token rows we need; capture them once here.)
    ...  # build prompt inputs exactly as sft/generate.py:133-141 -> forward_inputs per sample
    records = []
    for seq in sequences:                                              # each: prompt+completion token ids
        codes = codes_from_output(seq, n_prompt, ids)                  # 64 codes or None (refusal)
        lp = code_span_logprobs(model, {**vision, "input_ids": seq[None], "attention_mask": ...},
                                n_prompt=n_prompt, ids=ids) if codes and len(codes) == 64 else None
        records.append(RolloutSample(row_id=row["id"], cond_text=cond,
                                     spec_text=ground_truth_attribute_spec_text(row),
                                     n_prompt=n_prompt, seq=seq.cpu(), codes=codes,
                                     old_logprobs=(lp.cpu() if lp is not None else None),
                                     reward=None, advantage=None,
                                     refused=(codes is None), valid64=(codes is not None and len(codes)==64)))
    return records
```

Notes:
- **Grammar mask == restricting the softmax to `ids.codes` columns.** At every interior code position the
  legal set is always the full 256-code set, so slicing logits to `code_cols` before `log_softmax`
  reproduces exactly what `make_prefix_fn` allows there — no need to call the prefix fn inside the
  forward. `code_cols[k]` is the token id for codebook index `k`, so column `k` ↔ codebook index `k`,
  and `ids.id_to_index[emitted_token_id]` is that column.
- **Reuse the `generate` output row directly.** The row returned by `model.generate` already contains
  `prompt + completion`, so `full_ids = seq` needs no re-concatenation; forward it with the prompt's
  `pixel_values`/`image_grid_thw` and an all-ones attention mask.
- **Dropout must be OFF during rollout and both logprob passes.** LoRA dropout (`cfg.lora_dropout`)
  perturbs the distribution; if old-logprobs are captured with dropout on and current-logprobs with a
  different mask, the ratio is corrupted. Put the model in eval-dropout for generation + the old-logprob
  forward, and have Doc 03 do the same for the current-policy forward. Flag this as a GRPO detail.

### Reference-policy logprobs for KL (optional to cache here)

Doc 03's KL term needs the **reference** (frozen P6 init) logprob over the same span. The cheap option
is to compute it **once per rollout** here — one extra forward through the reference (same
`code_span_logprobs`) — and store `ref_logprobs` in the buffer, so Doc 03 doesn't re-run the reference
each accumulation microstep. **The reference must stay P6, not the pre-SFT base:** a plain
`disable_adapter()` on the shared 4-bit base gives the bare `models/base_resized`, so KL would regularize
toward the base instead of the P6 init. Keep a *second, frozen P6 adapter* selected via `set_adapter`
(or a separate `load_eval_model` P6 instance) for this forward. This is an optimization; the exact
reference mechanism lives in Doc 03. If you skip it, leave a `ref_logprobs=None` slot.

---

## The rollout buffer

One record per `(prompt, sample)`; a step produces `P × G` of them. Fields:

| Field | Filled by | Purpose |
|---|---|---|
| `row_id`, `cond_text`, `spec_text` | this doc | Provenance; `spec_text = ground_truth_attribute_spec_text(row)` for the reward. |
| `n_prompt`, `seq` (prompt+completion token ids) | this doc | The teacher-forced forward for the current-policy pass (Doc 03) and (if not cached) the reference pass — kept on CPU to save VRAM. |
| `codes` (list[int]\|None) | this doc | The 64 committed codebook indices; `None` = refusal. Feeds Doc 01's `score_batch`. |
| `old_logprobs` ([64]\|None) | this doc | Per-token old-policy logprob (grammar-masked). The ratio denominator in Doc 03. |
| `ref_logprobs` ([64]\|None) | this doc (optional) | Cached reference logprobs for the KL term (else Doc 03 recomputes). |
| `refused`, `valid64` (bool) | this doc | Routing/edge flags; `refused ⇒ reward 0`. |
| `reward` (float) | **Doc 01** | Shaped behavioral-fidelity reward from `score_batch`. |
| `advantage` (float) | **Doc 03** | `(r − mean_group)/(std_group + eps_adv)` over the G samples of this `row_id`. |

Keep the (image-heavy) **prompt vision tensors cached per prompt**, not per sample — G samples of one
row share one `pixel_values`/`image_grid_thw`. Store `seq`/logprob tensors on CPU; move to GPU per
microstep in Doc 03. Advantage is computed **within a `row_id` group** (Doc 03), so the buffer must keep
samples grouped by prompt (a `list[list[RolloutSample]]` or a `row_id` key).

---

## Device / VRAM notes (4-bit base)

- Base is 4-bit NF4 (`sft/loader.py:14`), a few GB resident; the LoRA delta is tiny. The **trainable**
  policy (Doc 03 must-build: `PeftModel.from_pretrained(..., is_trainable=True)` / `get_peft_model` from
  P6) still `.generate`s fine.
- Rollout is `torch.no_grad()` — no activation graph. Peak memory ≈ KV-cache for `chunk` concurrent
  sequences × full length. At `max_pixels=200704` (448², ~256 vision tokens) the prompt is a few hundred
  tokens and the completion is 66, so full length ≪ `max_seq_len=1024` — **no truncation risk at
  rollout** (unlike training, where `max_pixels` can end-truncate the 64 targets). `chunk=16` bounds the
  concurrent-sequence memory; lower it if OOM.
- The old-logprob forward is one full-sequence forward per sample (batch it at `chunk`). It is cheaper
  than the 64-step autoregressive generate, but it *is* a second pass over each sequence — budget for it.
- Turn **gradient checkpointing off** for rollout (it only helps the backward pass, which doesn't run
  here); Doc 03 re-enables it for the loss forward.

---

## Refusals & edge cases

- **Refusal (`<unsupported>`) on a supported row ⇒ reward 0** (invariant 4; `codes is None`). It still
  belongs in the group: a reward-0 sample drags the group mean down and gives positive advantage to the
  grade samples, i.e. GRPO learns to *stop refusing*. Its assistant span is not 64 codes, so its
  surrogate is over its emitted routing token(s) — **flagged open point** below; Doc 03 owns the exact
  refusal-span loss. For v1, record `old_logprobs=None` for refusals and let Doc 03 decide (simplest:
  include the position-0 routing token so the gradient can push `bos` over `unsupported`).
- **Non-64 / short generation.** `codes_from_output` can return fewer than 64 codes (`sft/generate.py:65`
  slices `[:TOKEN_COUNT]`). Treat `len(codes) != 64` as invalid → `valid64=False`, reward 0 (same rule as
  `eval/oracle_at_n.py:36` `score_row_samples` and the distillation 64-guard,
  `docs/collapse_fix/03_self_distillation.md` invariant 6). Do not train the surrogate on a malformed
  span.
- **`None`-fidelity rows** (spec asserts no measurable axis) are excluded from the aggregate, matching
  `summarize_fidelity` — but note these are a *reward* concern (Doc 01), not a rollout concern; the
  rollout still produces codes for them.
- **Degenerate group** (all G samples score identically ⇒ `std_group ≈ 0`). The `eps_adv` in
  `(r−mean)/(std+eps_adv)` keeps it finite (advantage ≈ 0 → ~no update for that prompt), which is correct;
  raising `temperature`/`G` reduces how often it happens.

---

## Invariants

1. **Holdout is sacred.** Roll out only on `supported_rows(rows, holdout=False)`; never generate on a
   holdout row (`sft/holdout.py:61`, `split_unit_id`). Doc 05's greedy eval is the only thing that
   touches the holdout.
2. **No target-LUT leakage.** Condition on `input_text_for(row, cfg.input_field)`; the reward scores
   against `ground_truth_attribute_spec_text(row)` only (Doc 01). ΔE stays eval-only. This is the
   `eval/best_of_n.py:55-62` split — copy it.
3. **Grammar-constrained rollouts.** Sample under `make_prefix_fn`; compute old-logprobs over the same
   256-code legal support, so Doc 03's ratio spans legal tokens only.
4. **Assistant-only span.** Logprobs (and Doc 03's surrogate) cover exactly the 64 code positions
   `[n_prompt .. n_prompt+63]`, the same boundary as `build_supervised_example`
   (`sft/example.py:218-219`).
5. **Refusal / non-64 on a supported row ⇒ reward 0** (`eval/oracle_at_n.py:36` rule); never train the
   surrogate on a malformed span.
6. **Old-logprobs come from the SAME weights that generated**, with dropout off, via the SAME function
   Doc 03 uses for the current policy. On the first gradient step of an on-policy rollout old ≈ new and
   the ratio starts at 1.
7. **SFT locked identity holds.** `G`, `temperature`, `top_p`, `chunk` are GRPO **methodology** knobs
   (config'd, outside the locked bilevel search); they do not alter `base_model_id`, quant,
   `num_new_tokens`, `max_seq_len`, `seed`, or paths.

---

## What to build vs reuse

**Reuse (import, do not reimplement):** `SpecialIds`, `make_prefix_fn`, `codes_from_output`
(`sft/generate.py`) — the truly-verbatim generator reuses; `generate_codes_batch` /
`generate_codes_for_row_batch` are **not** imported for rollout (they return codes only and drop the
`sequences`), their `.generate` body is inlined instead; `input_text_for`,
`resolve_image`, `supported_rows`, `build_supervised_example`'s `n_prompt` boundary (`sft/example.py`);
`ground_truth_attribute_spec_text` (`data_pipeline/attribute_spec.py:286`); the condition/score split of
`best_of_n_for_row` (`eval/best_of_n.py:52-62`).

**Build (new — this doc's deliverables, e.g. `sft/rollout.py`):**
1. `code_logprobs(model, batch) -> (logp, sel)` (Doc 03 §3) — grammar-masked, teacher-forced per-token
   logprob over the 64-code span; the SINGLE shared extractor, called under `no_grad` here (old policy)
   and under grad in Doc 03 (current policy). The per-sequence `code_span_logprobs` sketch above is the
   same op shown for alignment.
2. `rollout_row(...)` — draw G samples for one supported row and emit G buffer records (codes +
   old_logprobs + slots). Wraps the reused generator; captures the full sequences once.
3. The **`RolloutSample` / buffer** type (grouped by `row_id`).
4. *(Optional here, else Doc 03)* cached `ref_logprobs` from a **second, frozen P6 adapter** (selected via
   `set_adapter`) or a separate P6 instance — **not** `disable_adapter()`, which yields the pre-SFT base,
   not the P6 init.

**Depends on (other docs' must-builds):** the **trainable** policy load (`is_trainable=True`) and the
**reference** policy — both specified in [`03_grpo_loss.md`](03_grpo_loss.md); the reward that fills the
`reward` slot — [`01_reward.md`](01_reward.md).

---

## Verification

- **Logprob parity / alignment.** For a greedy rollout, `argmax` of the grammar-masked step
  distribution must equal the emitted code at every one of the 64 positions (proves the `n_prompt+j`
  alignment). Independently, `exp(old_logprobs).sum()`-style sanity: each per-token prob ∈ (0,1], and
  the sequence logprob monotonically decreases as tokens are added.
- **Old == teacher-forced.** `code_span_logprobs` under no_grad, fed a rollout's own sequence, must equal
  a from-scratch teacher-forced forward of `prompt + codes` (same numbers) — this is the contract Doc 03
  relies on when it recomputes with grad.
- **Reward parity (hands off to Doc 01).** The codes this doc emits, scored by `score_batch`, must match
  `score_generation` on the same codes (`tests/test_fast_reward.py` parity) — guards against a
  conditioning/scoring mixup.
- **Diversity check.** With `sampling` on and `G≥8`, a typical group has `std_group > 0` and >1 unique
  code sequence; if groups are degenerate, raise `temperature`/`G` before blaming the loss.
- **Conditioning parity.** Assert the rollout prompt string equals `build_supervised_example`'s prompt
  half for the same row + `cfg.input_field` (both route through `input_text_for`).

## Open questions / assumptions

1. **Refusal-span loss.** This doc records refusals as reward-0 with `old_logprobs=None`; the exact
   surrogate over a refusal's routing token (position-0 `bos` vs `unsupported`) is deferred to Doc 03.
   Assumed acceptable because rollout prompts are all supported (refusal is always the wrong choice).
2. **`output_scores` shortcut.** Assumed *not* used for v1 (warper contamination + version drift);
   flagged as a future speed lever behind a parity test.
3. **Reference-logprob caching location** (here vs Doc 03) is left to the implementer; both are noted.

## Deliverable

`sft/rollout.py` (`code_logprobs` — the canonical batched extractor Doc 03 shares; `rollout_row`; the
`RolloutSample`/buffer type) + unit tests (logprob alignment/parity, conditioning parity, refusal/non-64
flagging) — producing a per-step, `row_id`-grouped rollout buffer whose `reward` slot Doc 01 fills and
whose `advantage` slot + surrogate Doc 03 consumes.
