"""Shared SFT example construction + row loading (used by :mod:`sft.train` and
:mod:`sft.score_tokens`).

Kept in ONE place so the trainer and the token-accuracy scorer build byte-identical
``(input_ids, labels)`` tensors — the scorer's argmax-vs-labels accuracy is only meaningful if the
assistant-span masking matches training exactly.

Heavy deps (torch, transformers, qwen_vl_utils) are imported lazily inside
:func:`build_supervised_example`, so the pure row helpers (:func:`load_rows`, :func:`supported_rows`)
and this module import cleanly without the ``sft`` extra (unit-test-safe).

Image paths in the corpus are RELATIVE and resolve against ``$SLM_ARTIFACT_ROOT`` (the staged corpus
root on Colab, e.g. ``/content/slm``), falling back to cwd — same rule as the trainer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from functools import lru_cache

from data_pipeline.errors import SFTError
from sft.holdout import is_holdout_row


def artifact_root() -> Path:
    return Path(os.environ.get("SLM_ARTIFACT_ROOT", os.getcwd()))


def resolve_image(path: str) -> str:
    return path if os.path.isabs(path) else str(artifact_root() / path)


def is_supported_materialized(row: dict) -> bool:
    """A supported row with a full 64-token materialized target + the fields the trainer needs."""
    return bool(
        row.get("is_supported")
        and isinstance(row.get("target_tokens"), list)
        and len(row["target_tokens"]) == 64
        and row.get("image_path")
        and row.get("instruction")
        and row.get("assistant_target")
    )


def load_rows(active_rows_path: str) -> list[dict]:
    """Parse the active-rows JSONL into a list of dicts (blank lines skipped)."""
    text = Path(active_rows_path).read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def supported_rows(rows: list[dict], *, holdout: bool | None = None) -> list[dict]:
    """Supported+materialized rows, filtered by holdout membership.

    ``holdout=None`` → all supported rows; ``True`` → only the scored holdout slice;
    ``False`` → training pool (holdout excluded). Holdout membership is **unit-aware**
    (keyed on ``split_unit_id``, ADR 0024) so near-duplicate LUTs cannot straddle the boundary.
    See :mod:`sft.holdout`.
    """
    out: list[dict] = []
    for row in rows:
        if not is_supported_materialized(row):
            continue
        if holdout is None or is_holdout_row(row) == holdout:
            out.append(row)
    return out


# The 256 VQ code-token ids in the resized vocab, cached per tokenizer. Used to assert exact-64
# survival (below) without re-resolving 256 ids on every example build.
@lru_cache(maxsize=4)
def _code_token_ids(tokenizer) -> frozenset:
    from eval.vocab import code_token

    return frozenset(tokenizer.convert_tokens_to_ids(code_token(k)) for k in range(256))


def surviving_code_positions(tokenizer, input_ids, n_prompt: int) -> int:
    """Count VQ code tokens (``<lut_NNN>``) in the assistant span ``input_ids[0][n_prompt:]``.

    The single source of truth for "how many of the 64 target codes survived tokenization +
    end-truncation", shared by the trainer/scorer's exact-64 guard (ADR 0024, closes AUDIT F8).
    """
    code_ids = _code_token_ids(tokenizer)
    span = input_ids[0][n_prompt:].tolist()
    return sum(1 for t in span if t in code_ids)


def build_supervised_example(processor, row: dict, cfg, *, device=None) -> dict:
    """Build one ``(inputs, labels)`` example: assistant target un-masked, prompt masked to ``-100``.

    Mirrors the trainer's supervised construction so teacher-forced scoring aligns with training.
    Raises :class:`SFTError` on a degenerate mask (``n_prompt >= full_len`` after end-truncation,
    which would mask the whole sequence → NaN loss) so callers skip the row rather than corrupt.
    """
    from qwen_vl_utils import process_vision_info

    user = {
        "role": "user",
        "content": [
            {"type": "image", "image": resolve_image(row["image_path"])},
            {"type": "text", "text": row["instruction"]},
        ],
    }
    target = row["assistant_target"] if row.get("is_supported") else "<unsupported>"
    assistant = {"role": "assistant", "content": [{"type": "text", "text": target}]}

    prompt_text = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template([user, assistant], tokenize=False,
                                              add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info([user])
    full = processor(text=[full_text], images=image_inputs, videos=video_inputs, padding=True,
                     return_tensors="pt", max_length=cfg.max_seq_len, truncation=True)
    prompt = processor(text=[prompt_text], images=image_inputs, videos=video_inputs,
                       return_tensors="pt")

    n_prompt = prompt["input_ids"].shape[1]
    full_len = full["input_ids"].shape[1]
    if n_prompt >= full_len:
        raise SFTError(
            f"row {row.get('id')}: prompt ({n_prompt}) >= full ({full_len}) after truncation — the "
            f"assistant target was cut (raise max_seq_len or lower max_pixels)")

    # Exact-64 guard (ADR 0024, closes AUDIT F8): a supported row's assistant span must retain ALL
    # of its target code positions. The complete-loss guard above only catches TOTAL truncation;
    # this catches PARTIAL truncation (some of the 64 <lut_NNN> silently dropped near the seq/pixel
    # limit). Raising here makes the trainer/scorer skip+count the row rather than train/score a
    # truncated target.
    if row.get("is_supported"):
        expected = len(row.get("target_tokens") or [])
        n_code = surviving_code_positions(processor.tokenizer, full["input_ids"], n_prompt)
        if n_code != expected:
            raise SFTError(
                f"row {row.get('id')}: {n_code} code positions survived != expected {expected} "
                f"(partial truncation of the 64 target codes — lower max_pixels or raise max_seq_len)")

    labels = full["input_ids"].clone()
    labels[:, :n_prompt] = -100  # assistant-only loss
    full["labels"] = labels
    if device is not None:
        return {k: v.to(device) for k, v in full.items()}
    return full
