"""Pure loss-masking assembly for interpreter examples (no tokenizer needed)."""

from __future__ import annotations

from interpreter.example import assemble_example


def test_masks_prompt_and_appends_eos():
    ex = assemble_example([1, 2, 3], [4, 5], eos_id=9, max_seq_len=100)
    assert ex["input_ids"] == [1, 2, 3, 4, 5, 9]
    assert ex["labels"] == [-100, -100, -100, 4, 5, 9]  # prompt masked, target (incl EOS) supervised


def test_no_double_eos():
    ex = assemble_example([1, 2], [4, 9], eos_id=9, max_seq_len=100)
    assert ex["input_ids"] == [1, 2, 4, 9] and ex["labels"] == [-100, -100, 4, 9]


def test_left_truncates_prompt_keeps_full_target():
    # target -> [7,9] (len 2); budget = 4-2 = 2; keep the prompt TAIL [5,6] (the generation cue).
    ex = assemble_example([1, 2, 3, 4, 5, 6], [7], eos_id=9, max_seq_len=4)
    assert ex["input_ids"] == [5, 6, 7, 9] and ex["labels"] == [-100, -100, 7, 9]


def test_budget_zero_drops_prompt_entirely():
    ex = assemble_example([1, 2, 3], [7], eos_id=9, max_seq_len=2)
    assert ex["input_ids"] == [7, 9] and ex["labels"] == [7, 9]
