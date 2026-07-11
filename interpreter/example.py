"""Text-only supervised example construction for the interpreter (prompt -> attribute_spec_text).

No image, no VQ codes, no ``qwen_vl_utils`` — the one idea carried over from ``sft.example`` is
assistant-span loss masking (labels ``-100`` over the prompt). One prompt-format helper is shared by
train and score so the two never drift (a base/instruct format mismatch would silently wreck scoring).

The pure assembly (``assemble_example``) is separated from tokenization so it is unit-testable
without downloading a tokenizer.
"""

from __future__ import annotations

from typing import Optional

_IGNORE = -100

SYSTEM_PROMPT = (
    "You translate a user's global photo color/tone request into ONE canonical attribute_spec_text "
    "line and nothing else. Format: 'route=<grade|clarify|refuse> | <axis tokens>' (grade), "
    "'route=clarify | ' (too vague to grade), or 'route=refuse | refuse=<out_of_scope|out_of_gamut>' "
    "(not a single global color transform, or out of gamut). Output only the attribute_spec_text."
)


def assemble_example(prompt_ids: list[int], target_ids: list[int], eos_id: int,
                     max_seq_len: int) -> dict:
    """Concatenate prompt + target(+EOS), mask the prompt span, and left-truncate the PROMPT if the
    row is too long (never truncate the target — the model must always see the full spec + EOS)."""
    target = list(target_ids)
    if not target or target[-1] != eos_id:
        target = target + [eos_id]
    budget = max(0, max_seq_len - len(target))
    prompt = prompt_ids[-budget:] if budget else []  # keep the tail (the generation cue) if trimming
    input_ids = prompt + target
    labels = [_IGNORE] * len(prompt) + list(target)
    return {"input_ids": input_ids, "labels": labels}


def build_prompt_ids(tokenizer, text: str) -> list[int]:
    """Tokenize the (system + user) chat prompt with the generation cue appended (shared by train
    and score). Qwen2.5-Instruct ships a chat template; we never blind-call it on a base model.

    Render to a string first, then tokenize — ``apply_chat_template(tokenize=True)`` returns a
    ``BatchEncoding`` (whose elements are ``tokenizers.Encoding``) in transformers 5.x, which breaks
    ``torch.tensor``; the two-step form is an unambiguous ``list[int]`` across versions. The template
    emits the special tokens as text, so ``add_special_tokens=False`` avoids double BOS/EOS while the
    template's own markers still encode to their ids."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": text or ""}]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


def build_supervised_example(tokenizer, row: dict, max_seq_len: int) -> dict:
    """Full training example for one interpreter row: prompt = row['text'], target =
    row['attribute_spec_text']; prompt masked to -100; target terminated with EOS."""
    prompt_ids = build_prompt_ids(tokenizer, row["text"])
    target_ids = tokenizer(row["attribute_spec_text"], add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    return assemble_example(prompt_ids, target_ids, eos_id, max_seq_len)


def resolve_eos_and_pad(tokenizer) -> Optional[int]:
    """Ensure a pad token exists (many base/instruct tokenizers leave pad_token None -> collate
    fails). Point pad at eos; return the eos id used to terminate targets."""
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer.eos_token_id
