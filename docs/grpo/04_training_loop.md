# Doc 04 — Checkpointed training loop & QLoRA integration

**Prereq:** [`00_grounding.md`](00_grounding.md) (the canonical API map — read it, do not invent APIs),
[`docs/collapse_fix/README.md`](../collapse_fix/README.md) (problem framing). Siblings:
[`01_reward.md`](01_reward.md) (reward + advantage), [`02_rollout.md`](02_rollout.md) (rollouts +
per-token logprobs), [`03_grpo_loss.md`](03_grpo_loss.md) (clipped surrogate + KL),
[`05_eval_and_gates.md`](05_eval_and_gates.md) (holdout eval + success/anti-hacking gates).

**Audience:** the engineer wiring the pieces from 01/02/03 into one loop and running it on an A100.
This doc owns the **orchestration and the ANYTIME / checkpointed property** — the headline
requirement: *interrupting at any moment leaves a usable adapter, and we always keep the BEST one, not
just the latest.* It does **not** re-derive the reward (Doc 01), the rollout/logprob mechanics
(Doc 02), or the loss math (Doc 03); it consumes their entry points.

---

## Goal (one paragraph)

Close the greedy-vs-oracle gap by directly optimizing the free-running behavioral reward. The loop
repeats: **sample G rollouts per prompt [02] → reward each [01] → group-relative advantages [01] →
μ clipped-surrogate + KL policy updates [03] → checkpoint** (μ = `update_epochs`). The policy is the QLoRA LoRA params
initialized from the P6 SFT adapter; the KL reference is the *frozen* P6 init; the 4-bit NF4 base is
shared and frozen. Success (Doc 05): free-running **greedy** behavioral fidelity on the untouched
holdout climbs toward/past the **oracle 0.42**, with the anti-hacking diagnostics staying healthy.

---

## Files (this doc owns these; siblings own the rest)

- **New — entrypoint:** `sft/grpo_train.py` — the loop, checkpoint/resume/SIGINT harness, keep-BEST.
  Mirrors `sft/train.py`'s structure (`main()` → `train(cfg, ...)`, lazy heavy imports, single
  machine-readable summary line, fail-loud on a no-op).
- **New — config:** `sft/grpo/config.py` (`GRPOConfig`, `load_grpo_config`) + `configs/candidate_grpo.json`.
- **New — tests:** `tests/test_grpo_train.py` (harness logic — checkpoint round-trip, resume, keep-BEST
  selection, SIGINT flag — all stubbable without a GPU; mirror `test_build_distillation_corpus.py`'s
  "pure logic, no model" discipline).
- **Consumed from siblings (do NOT reimplement here):**
  - Doc 02: `rollout_row(policy, processor, row, cfg, *, G, sampling, …) -> list[RolloutSample]` — G
    grammar-constrained completions **with old-policy and reference per-token logprobs over the 64-code
    span** (this loop wraps that list into a `RolloutGroup`, below), plus
    `code_logprobs(model, batch) -> (logp, sel)` for the update pass (the SAME extractor Doc 02 uses
    under `no_grad` and Doc 03 uses under grad). (Grounding "must build" #1, #2.)
  - Doc 01: `shaped_rewards(codes_batch, spec_text, *, device, collapse_penalty, delta_e_weight) ->
    list[(reward|None, record)]` (wrapper around `eval.fast_reward.score_batch`,
    `eval/fast_reward.py:242`) and `group_advantages(rewards) -> list[float]` =
    `(r - mean_g)/(std_g + adv_eps)`.
  - Doc 03: `grpo_loss(cur_lp, old_lp, ref_lp, adv, sel, *, clip_eps, kl_beta) -> (loss, stats)` over
    the 64-code assistant span (new module `sft/grpo_loss.py`, analogous to `sft/soft_loss.py`; `sel` is
    the 64-code mask `code_logprobs` returns).
  - Doc 05: `holdout_greedy_eval(policy, processor, cfg, *, limit) -> dict` (`generate_codes(sampling=None)`
    `sft/generate.py:68` + `summarize_fidelity` `eval/behavioral_fidelity.py:221`) and the
    `is_best(new_summary, best_summary) -> bool` **anti-hacking-gated** BEST selector.

> If a sibling entry point above does not yet exist, it is a must-build **there**, not here. This doc
> assumes the signatures above and is written against them; if they land differently, fix this doc's
> call sites, not the invariants.

---

## Config — `GRPOConfig` + `configs/candidate_grpo.json`

**Design: compose, do not mutate.** GRPO wraps an `SFTConfig` rather than adding fields to it, so the
frozen `SFTConfig` and its lock enforcement (`sft/config.py:98-118`) stay byte-identical to SFT. The
locked SFT *identity* (base, quant, `num_new_tokens`, `max_seq_len`, `seed`, paths) is inherited
untouched; the GRPO methodology knobs live in their own dataclass, firewalled from the locked set.

```python
# sft/grpo/config.py
from dataclasses import dataclass, fields
from sft.config import SFTConfig, load_config   # sft/config.py:27,127

@dataclass(frozen=True)
class GRPOConfig:
    sft: SFTConfig                    # locked identity + SFT-tunable knobs (from load_config)
    init_adapter: str = "models/sft_adapters/p6_twostage_d0f9c744_smokefull"  # policy init AND frozen ref
    # -- rollout (methodology) --
    group_size: int = 8              # G samples/prompt (advantage needs G>=2 for a std)
    rollout_temperature: float = 0.7 # where oracle@32=0.42 coverage was measured (oracle_at_n.run); raise toward 1.0 for more group diversity
    rollout_top_p: float = 0.9
    rollout_chunk: int = 16          # -> ceil(G/chunk) .generate calls (sft/generate.py:117)
    prompts_per_round: int = 8       # prompts sampled per rollout round
    # -- optimization (methodology) --
    grpo_lr: float = 5.0e-6          # RL lr << SFT 2e-4 (Doc 03 §7 start)
    warmup_steps: int = 10
    grad_accum: int = 8              # prompt-groups accumulated per optimizer step
    update_epochs: int = 1           # μ inner passes over the round's buffer (μ in Doc 03; μ=1 ⇒ ρ≡1 ⇒ clip inactive)
    clip_eps: float = 0.2            # ε
    kl_beta: float = 0.05            # β
    adv_eps: float = 1.0e-4          # eps_adv (advantage std-divide guard)
    entropy_coef: float = 0.0        # optional rollout-entropy bonus (Doc 03 §7); off by default
    total_steps: int = 500           # optimizer-step budget (NOT sft.epochs — see note)
    # gradient clipping: reuse the SFT-locked cfg.sft.max_grad_norm (sft/config.py:60) — no GRPO override
    #   (locked knob; the loop reads cfg.sft.max_grad_norm directly). A name on BOTH dataclasses that the
    #   loop DOES read off GRPO (ckpt_every) routes to GRPOConfig — see load_grpo_config below.
    # -- checkpoint / eval (methodology) --
    ckpt_every: int = 20             # C: save `latest` every C optimizer steps + on SIGINT
    eval_every: int = 20             # holdout greedy eval cadence (optimizer steps)
    eval_limit: int = 64             # holdout slice size for the periodic eval (matches Phase-1 slice)
    keep_history: bool = False       # also snapshot history/step_NNNNNN/ each eval
    # -- reward shaping (OWNED by Doc 01; carried in the same JSON, passed through) --
    collapse_penalty: float = 0.25   # Doc 01 §2/§5 default (its worked example is pinned to 0.25)
    delta_e_weight: float = 0.0      # eval-only ΔE must never enter reward selection (Invariant 8)

    def __post_init__(self):
        if self.group_size < 2:
            raise ValueError("group_size must be >= 2 (group-relative std needs >=2 samples)")
        if not (0.0 < self.clip_eps < 1.0):     raise ValueError("clip_eps in (0,1)")
        if self.kl_beta < 0:                     raise ValueError("kl_beta must be >= 0")
        if self.ckpt_every < 1 or self.eval_every < 1: raise ValueError("intervals must be >= 1")
        # NOTE: SFTConfig.__post_init__ already enforced the locked identity; do not re-check it here.

def load_grpo_config(path: str) -> GRPOConfig:
    """Flat JSON: a key fills GRPOConfig if it declares that name, else the SFTConfig (same coercion as
    load_config). A name on BOTH dataclasses (e.g. ``ckpt_every``) routes to GRPOConfig — the value the
    loop actually reads via ``gcfg.ckpt_every`` — while SFTConfig keeps its (unused-in-GRPO) default; it
    never double-populates both. Unknown keys are a hard error (guards typos in a methodology knob)."""
    import yaml
    raw = yaml.safe_load(open(path, encoding="utf-8")) or {}
    grpo_names = {f.name for f in fields(GRPOConfig)} - {"sft"}   # GRPO-declared names WIN on a collision
    sft_names  = {f.name for f in fields(SFTConfig)}
    unknown = set(raw) - sft_names - grpo_names
    if unknown:
        raise ValueError(f"candidate_grpo.json: unknown keys {sorted(unknown)}")
    # A shared name goes to GRPO only (exclude grpo_names from the SFT set) so it can never populate both
    # AND so gcfg.ckpt_every reflects the JSON instead of silently keeping the GRPO dataclass default.
    sft_kw  = {k: (tuple(v) if isinstance(v, list) else v)
               for k, v in raw.items() if k in sft_names and k not in grpo_names}
    grpo_kw = {k: v for k, v in raw.items() if k in grpo_names}
    return GRPOConfig(sft=SFTConfig(**sft_kw), **grpo_kw)
```

`configs/candidate_grpo.json` — the SFT half is **identical to `candidate_two_stage.json`** (P6's
locked+tunable knobs, verified against that file), plus the GRPO block:

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

**Locked vs methodology — say it out loud (like `docs/collapse_fix/README.md:149-155` does for Doc 03).**
Every GRPO knob above (`group_size, clip_eps, kl_beta, rollout_*, grad_accum, update_epochs, grpo_lr,
total_steps, ckpt_every, eval_*`, reward-shape weights) is a **methodology knob, OUTSIDE the locked
bilevel search** — flag it exactly like the Phase-3 soft-loss knobs (`sft/config.py:82-91`). The SFT
**locked identity still holds** (base, quant, `num_new_tokens=259`, `max_seq_len=1024`, `seed`, paths);
`GRPOConfig` never overrides them. In particular `total_steps` is GRPO's own step budget — it is **not**
`sft.epochs` (still `=2`, unused by this loop), so the `epochs==2` lock is satisfied by construction,
not by the loop length.

---

## Model loading (three roles, ONE shared 4-bit base)

`sft/loader.py:load_eval_model` (`sft/loader.py:14`) returns a `.eval()` inference model — unusable as
the trainable policy. Build the training stack by mirroring `sft/train.py:97-113` for the base + kbit
prep + LoRA, then attach **two adapters on the one base**:

```python
# 1) shared 4-bit NF4 base (exactly sft/train.py:97-106) + processor
processor = AutoProcessor.from_pretrained(resized_model, min_pixels=cfg.sft.min_pixels,
                                          max_pixels=cfg.sft.max_pixels, trust_remote_code=True)
bnb  = BitsAndBytesConfig(load_in_4bit=cfg.sft.load_in_4bit, bnb_4bit_quant_type=cfg.sft.bnb_4bit_quant_type,
                          bnb_4bit_use_double_quant=cfg.sft.bnb_4bit_use_double_quant,
                          bnb_4bit_compute_dtype=compute_dtype)
base = _ModelCls.from_pretrained(resized_model, quantization_config=bnb, torch_dtype=compute_dtype,
                                 device_map="auto", trust_remote_code=True)
base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=cfg.sft.gradient_checkpointing)

# 2) TRAINABLE policy adapter, initialized from P6 (grounding must-build #3)
policy = PeftModel.from_pretrained(base, cfg.init_adapter, adapter_name="policy", is_trainable=True)
# 3) FROZEN reference adapter, ALSO P6, on the SAME base (grounding must-build #2)
policy.load_adapter(cfg.init_adapter, adapter_name="reference")   # is_trainable=False (default)
```

- **Same LoRA param set as P6** (Invariant): the adapter carries P6's `target_modules` +
  `modules_to_save=["embed_tokens","lm_head"]` (from its `adapter_config.json`). Loading an existing
  adapter reuses that param set exactly — do **not** re-`get_peft_model` with a different set.
- **Reference = frozen P6, not base-only.** The KL reference is `base + P6 adapter`, so
  `model.disable_adapter()` (which yields *base only*) is **wrong** for the reference. Use the named
  `"reference"` adapter instead. `set_adapter("policy")` for training/sampling; `set_adapter("reference")`
  under `torch.no_grad()` for reference logprobs.
- **Compute reference logprobs once per rollout and cache them** (Doc 02 fills them into the buffer).
  After the round, the reference adapter is never touched during the μ update passes — the KL term uses
  the cached `ref_lp`. This is the memory- and compute-optimal path; a second full `load_eval_model`
  instance is a fallback if two adapters on one base misbehave.
- **`use_cache` toggle (gotcha):** generation needs `model.config.use_cache=True`; gradient
  checkpointing (training) needs `use_cache=False`. Flip it at the rollout↔update boundary or PEFT
  will warn/OOM. Set `.train()` for updates, `.eval()` for rollout/eval.
- **Optimizer** mirrors `sft/train.py:122-123`: `AdamW([p for p in policy.parameters() if
  p.requires_grad], lr=cfg.grpo_lr, weight_decay=cfg.sft.weight_decay)` — only the `"policy"` adapter's
  params require grad, so this trains policy and never the reference. **Scheduler:** unlike SFT's
  cosine-to-zero over a fixed epoch budget (`sft/train.py:128-132`), GRPO runs an open-ended
  `total_steps`; use a short linear warmup (`warmup_steps`) to a **flat** `grpo_lr` (declare the
  schedule a methodology choice).

---

## The loop

Define one **GRPO step = one optimizer update.** A **round** = sample `prompts_per_round` prompts, G
rollouts each, reward + advantage → a buffer; then `update_epochs` passes over that buffer, each pass
accumulating `grad_accum` prompt-groups per optimizer step. Checkpoint/eval cadence counts optimizer
steps. This makes "every C steps", "eval every N steps", and `total_steps` unambiguous.

```python
def train(gcfg, resized_model, out_dir, run_id):
    policy, processor, opt = build_stack(gcfg, resized_model)   # two adapters on ONE base: policy + reference
    #   the frozen reference is the "reference" adapter on `policy` (set_adapter), NOT a separate model
    rows  = supported_rows(load_rows(gcfg.sft.active_rows_path), holdout=False) # TRAIN pool (holdout excluded)
    rng   = seed_everything(gcfg.sft.seed)                                      # torch/random/numpy
    state = resume_or_init(out_dir, run_id, policy, opt)   # -> step, best_summary, data_cursor, rng
    install_signal_handlers()                              # SIGINT + SIGTERM -> _INTERRUPTED flag

    while state.step < gcfg.total_steps and not _INTERRUPTED:
        # ---- ROLLOUT ROUND (no grad; use_cache=True; set_adapter('policy')) -------------------
        buffer = []
        for row in next_prompts(rows, state, gcfg.prompts_per_round):          # round-robin + reshuffle
            samples = rollout_row(policy, processor, row, gcfg.sft, G=gcfg.group_size,
                                   sampling={"temperature": gcfg.rollout_temperature,
                                             "top_p": gcfg.rollout_top_p})     # Doc 02: codes+old_lp+ref_lp
            #   rollout_row caches ref_logprobs via set_adapter("reference") on the same `policy` base
            spec = ground_truth_attribute_spec_text(row)                       # canonical spec (Invariant 8)
            rewards = shaped_rewards([s.codes for s in samples], spec, device=policy.device,
                                     collapse_penalty=gcfg.collapse_penalty,
                                     delta_e_weight=gcfg.delta_e_weight)        # Doc 01 (refusal -> 0)
            adv     = group_advantages([r for r, _ in rewards], eps=gcfg.adv_eps)   # Doc 01: (r-mean)/(std+adv_eps)
            buffer.append(RolloutGroup(samples, rewards, adv))                 # per-prompt group wrapper

        # ---- μ (update_epochs) UPDATE PASSES (grad; use_cache=False; policy.train()) ----------
        stop = False                                       # set on total_steps/_INTERRUPTED; checked at BOTH loop levels
        for _ in range(gcfg.update_epochs):
            rng.shuffle(buffer)
            for micro, rg in enumerate(iter_prompt_groups(buffer)):
                cur_lp, sel = code_logprobs(policy, rg.batch)                   # Doc 02/03 (teacher-forced, WITH grad)
                loss, lstats = grpo_loss(cur_lp, rg.old_lp, rg.ref_lp, rg.adv, sel,
                                         clip_eps=gcfg.clip_eps, kl_beta=gcfg.kl_beta)  # Doc 03
                (loss / gcfg.grad_accum).backward()
                if (micro + 1) % gcfg.grad_accum == 0:
                    optim_step(policy, opt, gcfg, state)                        # lr -> clip -> step -> zero
                    if state.step % gcfg.eval_every == 0:  maybe_eval_and_keep_best(...)   # Doc 05
                    if state.step % gcfg.ckpt_every == 0:  save_latest(out_dir, run_id, ...)
                    if state.step >= gcfg.total_steps or _INTERRUPTED:
                        stop = True; break                                      # flag it — don't just break the inner loop
            # trailing sub-grad_accum window: apply it (never mix into the next pass) UNLESS stopping,
            # so a total_steps/SIGINT boundary never fires an extra optimizer step (sft/train.py:197-201)
            if not stop:  flush_partial_accum(policy, opt, gcfg, state)
            if stop:      break                                                 # honor stop at the update-epochs level too

    save_latest(out_dir, run_id, ...)                      # final anytime save
    print_summary_line(...)                                # single {"grpo_summary": {...}} JSON line
```

- `optim_step` reuses `sft/train.py:154-166`'s pattern verbatim: set `lr` on the param groups,
  `clip_grad_norm_(trainable, gcfg.sft.max_grad_norm)`, `opt.step()`, `opt.zero_grad()`, increment
  `state.step`.
- `flush_partial_accum` mirrors `sft/train.py:197-201`: after each update-epoch pass it applies the
  trailing sub-`grad_accum` window (so it is neither silently discarded nor mixed into the next pass's
  first window) — but it is **skipped when the `stop` flag is set**, so hitting `total_steps` or a SIGINT
  never fires an extra optimizer step past the budget.
- **Fail-loud on a no-op** (mirror `sft/train.py:208-213`): if zero rollouts produced any gradable
  sample (every prompt refused / every image path unresolved — the `SLM_ARTIFACT_ROOT` case trap), print
  `[grpo][ABORT]` and return non-zero. Never write an untrained-but-`OK` adapter.

---

## Checkpoint directory layout

Under `out_dir/<run_id>/` (`out_dir` defaults to the locked `models/sft_adapters`). **`latest` is the
resume point, `best` is the deployable** — they are separate directories, never the same bytes.

```
models/sft_adapters/<run_id>/
├── latest/                       # overwritten every C steps + on SIGINT (atomic swap)
│   ├── adapter_config.json       # the "policy" adapter only
│   ├── adapter_model.safetensors # policy.save_pretrained(..., selected_adapters=["policy"])
│   ├── tokenizer* / *.json       # tok.save_pretrained (sft/train.py:217)
│   ├── adapter_manifest.json     # build_adapter_manifest + a "grpo" block (below)
│   ├── trainer_state.pt          # torch.save: {opt_state_dict, step, round, data_cursor,
│   │                             #   best_summary, rng: {torch, cuda, python, numpy}}
│   └── grpo_state.json           # human-readable mirror (step, round, best fidelity, last reward mean)
├── best/                         # BEST holdout greedy fidelity under the anti-hacking gate (Doc 05)
│   └── ... (same layout as latest; trainer_state.pt optional — best is for deploy, not resume)
├── history/                      # optional (keep_history): step_000200/, step_000400/, ...
└── eval_log.jsonl                # one line per periodic eval (see below)
```

- **Atomic writes.** `os.replace` is atomic for a **file or a symlink**, but on Linux/macOS it can only
  replace an *empty* target directory (POSIX `rename(2)` raises `ENOTEMPTY`/`EEXIST` over a populated
  one), so a single `os.replace('latest.tmp', 'latest')` over a live `latest/` crashes on the **2nd**
  save. Use a three-step directory swap: write to `latest.tmp/`, then
  `os.replace(latest, latest.old); os.replace(latest.tmp, latest); shutil.rmtree(latest.old, ignore_errors=True)`
  — each `os.replace` is itself atomic, so an interrupt mid-swap leaves either the old or the new
  `latest/` intact, never a half-written one. (Alternative: keep versioned `history/step_NNNNNN/` dirs and
  atomically re-point a `latest` **symlink** via `os.replace` on the symlink — the single-*file* form the
  repo already relies on, `scripts/materialize_target_tokens.py:171-175`.) Same three-step swap for `best/`.
- **Adapter save** is `policy.save_pretrained(path, selected_adapters=["policy"])` +
  `processor.tokenizer.save_pretrained(path)` — save the trained `"policy"` adapter, never the frozen
  `"reference"`. The manifest reuses `build_adapter_manifest`/`write_manifest`
  (`sft/manifest.py:37,65`) with `cfg=gcfg.sft.to_dict()` and an added `"grpo"` block
  `{init_adapter, group_size, clip_eps, kl_beta, rollout_temperature, rollout_top_p, grpo_lr,
  total_steps, step, best_greedy_fidelity}` for provenance.
- **`eval_log.jsonl`** — one JSON line per periodic eval so a run is falsifiable after the fact and the
  BEST decision is auditable: `{step, behavioral_fidelity_mean, collapse_rate, degenerate_rate,
  code_entropy_norm_mean, decoded_delta_e_mean, kl_to_ref, rollout_entropy, reward_mean, is_best}`.
  `behavioral_fidelity_mean, collapse_rate, degenerate_rate, code_entropy_norm_mean, decoded_delta_e_mean`
  are verbatim `summarize_fidelity` keys (`eval/behavioral_fidelity.py:234-245`; the free-running greedy
  fidelity is `behavioral_fidelity_mean`, `:237`); `kl_to_ref, rollout_entropy, reward_mean` are the
  loop's own stats.

---

## The ANYTIME property (the headline)

### Checkpoint every C steps **and** on SIGINT

Register handlers for **SIGINT (Ctrl-C) and SIGTERM (Colab preemption)** once, before the loop:

```python
_INTERRUPTED = False
def _on_signal(signum, frame):
    global _INTERRUPTED
    if _INTERRUPTED:                      # second signal -> hard exit
        raise KeyboardInterrupt
    _INTERRUPTED = True                   # first signal -> flag; loop saves at the next safe boundary
    print(f"[grpo] signal {signum} received — will checkpoint `latest` and exit cleanly")
import signal
signal.signal(signal.SIGINT, _on_signal); signal.signal(signal.SIGTERM, _on_signal)
```

The handler **only sets a flag** — it never saves from inside the handler (saving mid-`backward()` is
unsafe). The loop checks `_INTERRUPTED` at safe boundaries (after each `optim_step`, and the `while`
condition), and on seeing it: breaks, calls `save_latest(...)`, prints the summary, returns 0 with a
usable adapter. A **second** signal forces an immediate `KeyboardInterrupt` (escape hatch if a save
hangs). Because `latest/` is written atomically every `C` steps regardless, even a hard `kill -9` loses
at most the work since the last periodic save — resume picks up from `latest/`.

### Keep BEST (not just latest)

RL destabilizes and reward-hacks, so the latest checkpoint is frequently **worse** than an earlier one.
After every `eval_every` steps, run the holdout greedy eval and update BEST **only through Doc 05's
anti-hacking-gated selector**:

```python
def maybe_eval_and_keep_best(policy, processor, gcfg, out_dir, run_id, state, loop_stats):
    summ = holdout_greedy_eval(policy, processor, gcfg.sft, limit=gcfg.eval_limit)  # Doc 05
    summ.update(kl_to_ref=loop_stats.kl, rollout_entropy=loop_stats.entropy)
    append_eval_log(out_dir, run_id, state.step, summ)
    if is_best(summ, state.best_summary):        # Doc 05: fidelity UP *and* ΔE/collapse/KL/entropy healthy
        state.best_summary = summ
        atomic_copy(latest_dir, best_dir)        # promote current weights to best/
```

`is_best` (Doc 05) rejects a fidelity gain that arrives with rising `collapse_rate`/`decoded_delta_e` or
collapsing KL/`entropy_norm` — i.e. reward-hacking. `best_summary` lives in `trainer_state.pt`, so
resume preserves the bar across restarts.

**Eval on the holdout is read-only and sacred** — `holdout_greedy_eval` conditions on
`supported_rows(rows, holdout=True)` (`sft/example.py:72`, `sft/holdout.py:61`) and only *scores*; it
never adds a holdout row to the train pool. See Invariant 1.

### Resume from latest

`resume_or_init(out_dir, run_id, policy, opt)`: if `out_dir/<run_id>/latest/` exists, load
`trainer_state.pt` and (a) `policy.load_adapter(latest, adapter_name="policy")` — the **current**
weights, not P6; the frozen `"reference"` stays P6; (b) `opt.load_state_dict(state.opt_state_dict)`;
(c) restore `step, round, data_cursor, best_summary`; (d) restore RNG states. Otherwise start fresh from
`gcfg.init_adapter` with `step=0`. **Rollouts are non-deterministic by design** (sampling + Metal/CUDA
nondeterminism) — restore RNG for reproducible *bookkeeping/data order*, but do not claim bitwise
run-to-run reproducibility of the rollouts (same caveat as `docs/collapse_fix/03_self_distillation.md`).

---

## VRAM budget (40 GB A100, `per_device_batch_size=1` — the locked B=1, `AGENTS.md`)

| Item | Approx | Note |
|---|---|---|
| 4-bit NF4 base (3B), **shared** | ~2.5 GB | one copy for all three roles |
| LoRA deltas ×2 adapters (r=16) | <0.1 GB | policy + reference |
| `modules_to_save` embed+lm_head ×2 adapters | ~2–3 GB | two trainable-shaped copies of the big vocab layers (same cost SFT already pays, ×2 for the ref) |
| AdamW state (policy trainable only) | ~4–5 GB | dominated by the embed/lm_head rows (`modules_to_save`); RL trains the **same** param set as SFT, which already fits |
| Rollout KV cache | ~1–2 GB | G×(prompt+68) tokens, `rollout_chunk=16` bounds it (`sft/generate.py:117`) |
| Activations (updates) | ~2–4 GB | `gradient_checkpointing=True` keeps this small |
| Buffer (logprobs/rewards on CPU) | negligible | float lists, not tensors on GPU |

Working set ≈ **15–20 GB** — comfortable headroom on a 40 GB A100. **If OOM:** lower `rollout_chunk`,
lower `prompts_per_round`, cache reference logprobs and *free the reference adapter for the update pass*,
or (last resort) offload AdamW state. Do **not** raise `max_pixels` above 401408 (truncates the 64
targets — `sft/example.py:212-217` guard) and do **not** disable gradient checkpointing without checking
headroom (`sft/train.py:258-260`).

---

## Invariants (get one wrong and the run is meaningless)

1. **Holdout is sacred.** Train pool = `supported_rows(rows, holdout=False)`; periodic eval slice =
   `supported_rows(rows, holdout=True)`. Never sample a rollout from, or update on, a holdout row
   (`sft/holdout.py:61`, unit-aware on `split_unit_id`).
2. **SFT locked identity holds.** `GRPOConfig` never overrides `base_model_id`, quant scheme,
   `num_new_tokens=259`, `max_seq_len=1024`, `seed`, or paths; they come from `SFTConfig` unchanged
   (`sft/config.py:98-118`).
3. **Same LoRA param set as P6.** The policy is P6's adapter (`target_modules` +
   `modules_to_save=["embed_tokens","lm_head"]`); do not re-init a different set.
4. **GRPO knobs are methodology, outside the locked bilevel search** — flag them like the Phase-3
   soft-loss knobs (`sft/config.py:82-91`); the bilevel loop must never propose them.
5. **Reference = frozen P6 init, never updates.** Not base-only; use the named `"reference"` adapter.
6. **Anytime.** A SIGINT/SIGTERM/crash at any point leaves a usable adapter (atomic `latest/` every C
   steps + on-signal save). `best/` is separate from `latest/`.
7. **Keep BEST under the anti-hacking gate** (Doc 05), not raw latest fidelity.
8. **No target-LUT leakage.** Condition on `input_text_for(row, cfg.sft.input_field)`
   (`sft/example.py:136`); score against `ground_truth_attribute_spec_text(row)`
   (`data_pipeline/attribute_spec.py:286`, `bucketize=False`). ΔE is eval-only and must never drive the
   training reward's selection (`delta_e_weight` shapes magnitude only, per Doc 01's `rerank_key` rule).
9. **Assistant-only 64-code span.** The surrogate loss and both logprob passes cover only the masked
   span, exactly as `build_supervised_example` (`sft/example.py:218-219`).
10. **Grammar-constrained rollouts**, with old-policy logprobs taken from the *same* constrained pass
    (`make_prefix_fn`, `sft/generate.py:39`) so ratios span the legal support only (Doc 02).
11. **Refusal on a supported row ⇒ reward 0**; a `None`-fidelity row is excluded (Doc 01, matching
    `summarize_fidelity`).
12. **Save the `policy` adapter only**; never the frozen reference; never touch the frozen tokenizer or
    `attribute_spec.serialize`/`parse`.

---

## What to build vs reuse

**Reuse (import; do not reimplement):**
- Base/LoRA/optimizer/save patterns from `sft/train.py` (`:106` kbit prep, `:108-113` LoRA+get_peft_model,
  `:122-123` AdamW, `:154-166` clip→step→zero, `:199-200` trailing flush, `:215-231` save+manifest,
  `:208-213` fail-loud) and `sft/loader.py:35-40` (bnb config).
- Rollouts + grammar + logprobs — Doc 02 (`generate_codes_batch` `sft/generate.py:117`, `make_prefix_fn`
  `:39`, `SpecialIds` `:27`).
- Reward + advantage — Doc 01 (wrapper over `eval/fast_reward.py:242`).
- Loss — Doc 03.
- Periodic eval + BEST selector — Doc 05 (`generate_codes(sampling=None)` `sft/generate.py:68` +
  `summarize_fidelity` `eval/behavioral_fidelity.py:221`).
- Data/config — `supported_rows`/`input_text_for`/`load_rows` (`sft/example.py`), `is_holdout_row`
  (`sft/holdout.py:61`), `load_config`/`SFTConfig` (`sft/config.py`), `build_adapter_manifest`/
  `write_manifest` (`sft/manifest.py:37,65`).

**Build here (this doc's deliverables):**
- `sft/grpo_train.py` — the loop orchestration, `optim_step`/`flush_partial_accum`, the round/step
  accounting, `next_prompts` cursor, the `{"grpo_summary": {...}}` line, fail-loud abort.
- The **checkpoint/resume/SIGINT harness**: atomic `latest/`, separate `best/`, `trainer_state.pt`
  (opt + step + rng + best_summary), `resume_or_init`, the SIGINT/SIGTERM flag handler,
  `eval_log.jsonl` writer (grounding must-build #8).
- **Model stack loader** — two adapters (`policy` trainable + `reference` frozen) on one shared 4-bit
  base, `use_cache` toggle, reference-logprob caching (grounding must-build #2, #3).
- The **rollout buffer** type (`RolloutGroup`: wraps Doc 02's `list[RolloutSample]` for one prompt +
  its rewards + group advantages + the teacher-forced `batch`/`mask`) (grounding must-build #4).
- `sft/grpo/config.py` (`GRPOConfig`, `load_grpo_config`) + `configs/candidate_grpo.json`.
- `tests/test_grpo_train.py` — harness logic without a GPU.

---

## Verification (before the overnight run)

- **Smoke:** `python -m sft.grpo_train --config configs/candidate_grpo.json --resized-model
  models/base_resized --run-id grpo_smoke --total-steps 4 --prompts-per-round 2` on a handful of train
  rows. Assert: rollouts produce ≥1 gradable sample (else `[grpo][ABORT]`), loss is finite, `latest/`
  and `eval_log.jsonl` appear, and a `SIGINT` mid-run leaves a `latest/` that **re-loads and resumes**
  (run the smoke, Ctrl-C, restart, confirm `step` continues and `best_summary` is preserved).
- **Anytime test (harness, no GPU):** in `tests/test_grpo_train.py`, drive `save_latest`→`resume_or_init`
  round-trip with a stub optimizer/adapter; assert step/rng/best_summary restore; assert `is_best`
  promotes `best/` only on a gated improvement; assert the SIGINT flag causes a clean save+exit.
- **Reward parity (Invariant 9/11):** on a few rollout rows, assert the loop's shaped base reward equals
  `score_generation` (`eval/behavioral_fidelity.py:202`) on the same codes (guards a shaping bug that
  silently changes the objective) — the same parity contract Doc 01 pins.
- **Success gate (Doc 05):** free-running greedy holdout fidelity climbs toward/past **oracle 0.42**
  (`eval.oracle_at_n.run` `eval/oracle_at_n.py:74`), vs the **best-of-N 0.42** (`eval.best_of_n.evaluate`
  `eval/best_of_n.py:65`) and greedy-0.159 baselines, **without** the anti-hacking diagnostics
  degrading. Re-run `oracle_at_n.run` on the promoted `best/` adapter — greedy should rise toward
  oracle@N without oracle@N regressing.

## Deliverable

`sft/grpo_train.py` + `sft/grpo/config.py` + `configs/candidate_grpo.json` + `tests/test_grpo_train.py`;
a GRPO run whose promoted `best/` adapter's free-running greedy behavioral fidelity on the untouched
holdout beats the 0.159 baseline and moves toward/past oracle 0.42, with `eval_log.jsonl` showing the
anti-hacking diagnostics stayed healthy across the run.
