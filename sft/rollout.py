"""Rollouts + per-token logprobs + the rollout buffer for GRPO (docs/grpo/02_rollout.md).

For each SUPPORTED training row we draw ``G`` free-running, grammar-constrained rollouts and capture,
per rollout, the 64 committed codebook indices AND the OLD-policy per-token logprob over the 64-code
assistant span (and, when a frozen reference adapter is present, the reference logprob for KL). The
grammar helpers (:func:`sft.generate.make_prefix_fn` / :class:`~sft.generate.SpecialIds` /
:func:`~sft.generate.codes_from_output`) are reused verbatim.

The one real build here (``generate_*`` returns codebook indices only — no logprobs) is
:func:`code_logprobs` — the SINGLE grammar-masked teacher-forced per-token logprob extractor, shared
with :mod:`sft.grpo_loss` (Doc 03 §3): called under ``torch.no_grad()`` here for the OLD/REFERENCE
policy and under grad in the update pass for the CURRENT policy. Both passes go through the identical
code path, so on the first inner update after a rollout ``old == new`` and the ratio starts at 1.

Rather than salvage the token ``sequences`` out of ``generate`` (which drops them), we materialize a
teacher-forced example from each SAMPLED completion — the labels-based canonical form (Doc 03 §6 /
IMPLEMENTATION_PROMPT §7): the sampled assistant string ``<lut_bos> <lut_NNN>*64 <lut_eos>`` with the
same ``build_supervised_example`` masking (``labels[:, :n_prompt] = -100``). The old-policy logprob at
rollout time and the current-policy logprob at update time then use the byte-identical batch.

**Dropout is OFF for generation AND both logprob passes** (Doc 02 invariant 6): LoRA dropout would
give the old and current forwards different masks and corrupt the importance ratio. The GRPO loop keeps
the model in ``.eval()`` for every forward (gradients still flow through the ``requires_grad`` LoRA +
``modules_to_save`` params); ``use_cache`` is toggled at the rollout↔update boundary by the loop.

Heavy deps (torch, qwen_vl_utils) are imported lazily so the pure buffer types + this module import
without the ``sft`` extra.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sft.generate import (
    DEFAULT_MAX_NEW_TOKENS,
    TOKEN_COUNT,
    SpecialIds,
    codes_from_output,
    make_prefix_fn,
)

# ---------------------------------------------------------------------------------------------------
# Process-global code-token id maps (the frozen tokenizer is fixed for the whole run, so this is a
# legitimate one-time setup per process, exactly as Doc 03 §3 frames `code_ids` / `id2idx`).
# ---------------------------------------------------------------------------------------------------
_CODE_IDS: list[int] | None = None          # 256 vocab ids in codebook-index order (column k <-> code k)
_TENSOR_CACHE: dict[str, tuple] = {}        # device-str -> (code_ids LongTensor[256], id2idx LongTensor)


def init_code_maps(tokenizer) -> None:
    """Populate the process-global 256-code id map from a (real or fake) tokenizer. Idempotent."""
    global _CODE_IDS
    _CODE_IDS = list(SpecialIds(tokenizer).codes)
    _TENSOR_CACHE.clear()


def _maps_for(model):
    """(code_ids[256], id2idx[max_code_id+1]) tensors on ``model``'s device, cached per device.

    ``id2idx`` is sized ``max(code_ids)+1``; since ``<lut_255>`` is the last resized-vocab token, that
    equals ``len(tokenizer)`` and covers every gold token id the forward can emit (Doc 03 §3 note).
    """
    import torch

    if _CODE_IDS is None:
        raise RuntimeError("sft.rollout.init_code_maps(tokenizer) must be called before code_logprobs")
    dev = getattr(model, "device", None)
    key = str(dev)
    cached = _TENSOR_CACHE.get(key)
    if cached is not None:
        return cached
    code_ids = torch.as_tensor(_CODE_IDS, dtype=torch.long, device=dev)
    # Size id2idx to cover EVERY gold token id, not just the code ids. ``<lut_255>`` is normally the
    # last resized-vocab token (so max_code+1 == len(tok)), but bound by the model's vocab_size too so
    # a stray higher-id token can never index out of bounds (a CUDA device-side assert).
    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
    size = max(int(code_ids.max()) + 1, vocab_size)
    id2idx = torch.zeros(size, dtype=torch.long, device=dev)
    id2idx[code_ids] = torch.arange(len(_CODE_IDS), dtype=torch.long, device=dev)
    _TENSOR_CACHE[key] = (code_ids, id2idx)
    return code_ids, id2idx


def _forward_code_logp(model, batch):
    """Teacher-forced forward -> (logp_full [B,T-1,256] fp32 log-probs over the legal 256-code support,
    gold_idx [B,T-1] codebook index of the emitted token, sel [B,T-1] the 64-code mask).

    Mirrors ``sft/score_tokens.py:247-252`` + Doc 03 §3: logits at position ``t`` predict token ``t+1``;
    restrict to the 256 code columns and ``log_softmax`` there (legal grammar support only, Invariant 8);
    ``sel`` = assistant mask (``labels != -100``) ∩ ``is_code``. ``labels`` is used ONLY for the mask —
    it is not forwarded to the model (avoids the model recomputing a CE loss)."""
    import torch
    import torch.nn.functional as F

    code_ids, id2idx = _maps_for(model)
    model_inputs = {k: v for k, v in batch.items() if k != "labels"}
    logits = model(**model_inputs).logits[:, :-1, :]         # [B, T-1, V]; predicts token t+1
    gold = batch["input_ids"][:, 1:]                          # [B, T-1] emitted/sampled tokens
    labels = batch["labels"][:, 1:]                           # -100 on the prompt half
    sel = (labels != -100) & torch.isin(gold, code_ids)       # the 64 code positions
    code_logits = logits[..., code_ids].float()               # [B, T-1, 256] legal support; fp32 (Doc 03 §5)
    logp_full = F.log_softmax(code_logits, dim=-1)
    gidx = id2idx[gold].clamp_(0, len(_CODE_IDS) - 1)         # codebook index; garbage off-span, masked out
    return logp_full, gidx, sel


def code_logprobs(model, batch):
    """Per-token logprob of the EMITTED code at each of the 64 code positions (the canonical extractor).

    Returns ``(logp [B,T-1], sel [B,T-1])`` — ``logp`` is ``0.0`` off the code span (masked). Used for
    ``logp_new`` (with grad, in the update) and ``logp_old`` / ``logp_ref`` (no grad, at rollout time);
    the identical code path makes ``ρ ≡ 1`` on the first inner update after a rollout."""
    logp_full, gidx, sel = _forward_code_logp(model, batch)
    logp_t = logp_full.gather(-1, gidx[..., None]).squeeze(-1)     # [B, T-1]
    return logp_t.masked_fill(~sel, 0.0), sel


def _mean_code_entropy(logp_full, sel) -> float:
    """Mean per-token entropy (nats) of the grammar-masked 256-code distribution over selected positions.

    The rollout-entropy guard (Doc 05): if this collapses toward 0 the G samples are becoming identical.
    """
    import torch

    ent = -(logp_full.exp() * logp_full).sum(-1)                  # [B, T-1] per-token entropy
    n = sel.sum().clamp(min=1)
    return float((ent * sel).sum() / n)


# ---------------------------------------------------------------------------------------------------
# Teacher-forced example from a SAMPLED completion (mirrors sft.example.build_supervised_example, but
# substitutes the sampled assistant target). Re-spelled locally to keep the module torch-free at import
# (Doc 03 §6 sanction); pinned against scripts.materialize_target_tokens._assistant_target by a test.
# ---------------------------------------------------------------------------------------------------
def assistant_target_from_codes(codes) -> str:
    """``<lut_bos> <lut_NNN>*64 <lut_eos>`` for a sampled 64-code completion (== the materializer form)."""
    return "<lut_bos> " + " ".join(f"<lut_{int(c):03d}>" for c in codes) + " <lut_eos>"


def build_rollout_example(processor, image, cond_text: str, codes, cfg, *, device=None) -> dict:
    """One teacher-forced (input_ids, labels, vision) example for a SAMPLED 64-code completion.

    Conditioning is ``cond_text`` (already resolved via ``input_text_for`` by the caller — training
    parity); the assistant target is the sampled codes. Prompt positions are masked to ``-100`` exactly
    as :func:`sft.example.build_supervised_example`. Returns a batch-of-1 dict."""
    from qwen_vl_utils import process_vision_info

    user = {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": cond_text}]}
    assistant = {"role": "assistant",
                 "content": [{"type": "text", "text": assistant_target_from_codes(codes)}]}
    prompt_text = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template([user, assistant], tokenize=False,
                                              add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info([user])
    full = processor(text=[full_text], images=image_inputs, videos=video_inputs, padding=True,
                     return_tensors="pt", max_length=cfg.max_seq_len, truncation=True)
    prompt = processor(text=[prompt_text], images=image_inputs, videos=video_inputs, return_tensors="pt")
    n_prompt = prompt["input_ids"].shape[1]

    labels = full["input_ids"].clone()
    labels[:, :n_prompt] = -100                              # assistant-only span (build_supervised_example)
    full["labels"] = labels
    full["_n_prompt"] = n_prompt
    if device is not None:
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in full.items()}
    return full


# ---------------------------------------------------------------------------------------------------
# Buffer types
# ---------------------------------------------------------------------------------------------------
@dataclass
class RolloutSample:
    """One (prompt, sample) rollout record. See docs/grpo/02_rollout.md 'The rollout buffer'."""

    row_id: str
    cond_text: str
    spec_text: str
    codes: list[int] | None                 # 64 committed codebook indices; None = refusal
    refused: bool
    valid64: bool
    n_prompt: int = 0
    example: dict | None = None             # teacher-forced batch-of-1 (CPU); None for refusals
    old_logprobs: object = None             # FloatTensor[T-1] (CPU), masked 0 off-span; None for refusals
    ref_logprobs: object = None             # FloatTensor[T-1] (CPU); None if no reference adapter/refusal
    entropy: float | None = None            # mean per-token rollout entropy over the 64 code positions
    reward: float | None = None             # filled by the loop via eval.grpo_reward.shaped_rewards
    advantage: float | None = None          # filled by the loop via eval.grpo_reward.group_advantages


@dataclass
class RolloutGroup:
    """The G rollouts of ONE prompt, plus the rewards/advantages the loop assigns.

    Construct as ``RolloutGroup(samples, rewards, adv)`` where ``rewards`` is the ``shaped_rewards``
    output (``list[(reward|None, record)]``) and ``adv`` is the ``group_advantages`` output — both in
    per-sample order. The constructor writes ``reward``/``advantage`` back onto the samples and records
    group telemetry (reward mean, advantage std, refusal rate, rollout entropy)."""

    samples: list[RolloutSample]
    rewards: list = field(default_factory=list)
    advantages: list = field(default_factory=list)

    def __post_init__(self):
        if self.rewards:
            for s, (r, _rec), a in zip(self.samples, self.rewards, self.advantages):
                s.reward = r
                s.advantage = a
        self.row_id = self.samples[0].row_id if self.samples else None

    # -- telemetry ----------------------------------------------------------------------------------
    @property
    def measurable_rewards(self) -> list[float]:
        return [s.reward for s in self.samples if s.reward is not None]

    @property
    def reward_mean(self) -> float | None:
        import numpy as np
        xs = self.measurable_rewards
        return float(np.mean(xs)) if xs else None

    @property
    def advantage_std(self) -> float | None:
        import numpy as np
        xs = [s.advantage for s in self.samples if s.advantage is not None]
        return float(np.std(xs)) if xs else None

    @property
    def refusal_rate(self) -> float:
        return (sum(1 for s in self.samples if s.refused) / len(self.samples)) if self.samples else 0.0

    @property
    def entropy_mean(self) -> float | None:
        import numpy as np
        xs = [s.entropy for s in self.samples if s.entropy is not None]
        return float(np.mean(xs)) if xs else None

    # -- update-pass tensors ------------------------------------------------------------------------
    def gradable(self) -> list[RolloutSample]:
        """Valid-64 samples with a measurable advantage — the ones that carry a surrogate gradient.

        Refusals (not valid64) and None-fidelity samples (advantage None) are excluded: their effect is
        already baked into the group mean via ``group_advantages`` (Doc 03 §4)."""
        return [s for s in self.samples
                if s.valid64 and s.advantage is not None and s.example is not None]

    def has_grad(self) -> bool:
        return len(self.gradable()) > 0

    def build(self, device):
        """Stack the gradable samples into (batch, old_lp[B,T-1], ref_lp[B,T-1], adv[B,1]) on ``device``.

        All gradable samples of one prompt share the prompt (same image + cond_text) and a fixed 66-token
        completion, so their sequences have identical length — a plain stack, no padding. ``pixel_values``
        for the shared image are concatenated per row (matching how the processor batches B copies)."""
        import torch

        gs = self.gradable()
        exs = [s.example for s in gs]
        batch = {
            "input_ids": torch.cat([e["input_ids"] for e in exs], 0).to(device),
            "attention_mask": torch.cat([e["attention_mask"] for e in exs], 0).to(device),
            "labels": torch.cat([e["labels"] for e in exs], 0).to(device),
        }
        if "pixel_values" in exs[0]:
            batch["pixel_values"] = torch.cat([e["pixel_values"] for e in exs], 0).to(device)
        if "image_grid_thw" in exs[0]:
            batch["image_grid_thw"] = torch.cat([e["image_grid_thw"] for e in exs], 0).to(device)
        old_lp = torch.stack([s.old_logprobs for s in gs]).to(device)               # [B, T-1]
        ref_src = [s.ref_logprobs if s.ref_logprobs is not None else s.old_logprobs for s in gs]
        ref_lp = torch.stack(ref_src).to(device)                                    # [B, T-1]
        adv = torch.tensor([float(s.advantage) for s in gs], dtype=torch.float32,
                           device=device).unsqueeze(1)                              # [B, 1]
        return batch, old_lp, ref_lp, adv


def _has_reference_adapter(model, name: str) -> bool:
    return name in (getattr(model, "peft_config", None) or {})


def rollout_row(model, processor, row, cfg, *, G: int, sampling: dict, chunk: int = 16,
                device=None, reference_adapter: str = "reference",
                max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS) -> list[RolloutSample]:
    """Draw G grammar-constrained rollouts for ONE supported training row -> G buffer records.

    Conditions on ``input_text_for(row, cfg.input_field)`` (training parity), scores against
    ``ground_truth_attribute_spec_text(row)`` (canonical; the reward's spec — Doc 01). Samples the G
    completions with :func:`sft.generate.generate_codes_batch` (grammar-constrained), then for each
    valid-64 completion builds the teacher-forced example and caches the OLD-policy logprob (and the
    REFERENCE logprob, if the model carries a frozen ``reference_adapter``). ``sampling`` MUST enable
    sampling (the group advantage needs diversity)."""
    import torch

    from data_pipeline.attribute_spec import ground_truth_attribute_spec_text
    from sft.example import input_text_for, resolve_image
    from sft.generate import generate_codes_batch

    init_code_maps(processor.tokenizer)
    ids = SpecialIds(processor.tokenizer)
    image = resolve_image(row["image_path"])
    cond = input_text_for(row, cfg.input_field)                       # conditioning (training parity)
    spec = ground_truth_attribute_spec_text(row)                      # scoring spec (canonical)
    row_id = str(row.get("id"))

    if not sampling or not sampling.get("temperature"):
        # Guard the caller: greedy with G>1 yields G identical rows -> std 0 -> no learning signal.
        raise ValueError("rollout_row requires a sampling dict enabling sampling (e.g. {'temperature':0.7})")

    # Draw the G samples (grammar-constrained). generate_codes_batch returns codebook indices | None.
    codes_list = generate_codes_batch(model, processor, image=image, text=cond, n=G,
                                      sampling=sampling, chunk=chunk, max_new_tokens=max_new_tokens,
                                      device=device)

    has_ref = bool(_has_reference_adapter(model, reference_adapter)) and hasattr(model, "set_adapter")
    dev = device if device is not None else getattr(model, "device", None)
    samples: list[RolloutSample] = []
    for codes in codes_list:
        valid64 = codes is not None and len(codes) == TOKEN_COUNT
        if not valid64:
            samples.append(RolloutSample(row_id=row_id, cond_text=cond, spec_text=spec, codes=codes,
                                         refused=codes is None, valid64=False))
            continue
        ex = build_rollout_example(processor, image, cond, codes, cfg, device=dev)
        n_prompt = int(ex.pop("_n_prompt"))
        with torch.no_grad():
            logp_full, gidx, sel = _forward_code_logp(model, ex)
            old_lp = logp_full.gather(-1, gidx[..., None]).squeeze(-1).masked_fill(~sel, 0.0)[0]
            entropy = _mean_code_entropy(logp_full, sel)
            ref_lp = None
            if has_ref:
                model.set_adapter(reference_adapter)
                try:
                    r_lp, _ = code_logprobs(model, ex)
                    ref_lp = r_lp[0]
                finally:
                    model.set_adapter("policy")
        cpu_ex = {k: (v.detach().to("cpu") if hasattr(v, "detach") else v) for k, v in ex.items()}
        samples.append(RolloutSample(
            row_id=row_id, cond_text=cond, spec_text=spec, codes=list(codes), refused=False,
            valid64=True, n_prompt=n_prompt, example=cpu_ex,
            old_logprobs=old_lp.detach().to("cpu"),
            ref_logprobs=(ref_lp.detach().to("cpu") if ref_lp is not None else None),
            entropy=entropy))
    return samples
