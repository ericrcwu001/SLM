"""Tests for the free-running generation grammar (sft.generate).

The grammar (:func:`make_prefix_fn`) and id-mapping (:func:`codes_from_output`) are pure and
tested without torch/transformers/a model. The ``.generate`` path itself needs the GPU stack
and is exercised on Colab.
"""

from __future__ import annotations

import numpy as np

from sft.generate import SpecialIds, codes_from_output, make_prefix_fn


class _FakeTokenizer:
    """Minimal stand-in: <lut_bos>=1000, <lut_eos>=1001, <unsupported>=1002, codes 2000..2255."""

    eos_token_id = 7

    def convert_tokens_to_ids(self, tok: str) -> int:
        table = {"<lut_bos>": 1000, "<lut_eos>": 1001, "<unsupported>": 1002}
        if tok in table:
            return table[tok]
        return 2000 + int(tok[len("<lut_"):-1])  # <lut_NNN> -> 2000+NNN


def _ids() -> SpecialIds:
    return SpecialIds(_FakeTokenizer())


def _seq(prompt_len, *generated):
    return np.array(list(range(prompt_len)) + list(generated), dtype=np.int64)


def test_grammar_start_allows_bos_or_unsupported():
    ids = _ids()
    fn = make_prefix_fn(prompt_len=5, ids=ids)
    assert fn(0, _seq(5)) == [ids.bos, ids.unsupported]


def test_grammar_codes_then_eos():
    ids = _ids()
    fn = make_prefix_fn(prompt_len=5, ids=ids)
    # after BOS, the 64 code positions must be code ids
    assert fn(0, _seq(5, ids.bos)) == ids.codes
    assert fn(0, _seq(5, ids.bos, *([ids.codes[0]] * 63))) == ids.codes  # 63 emitted -> still codes
    assert fn(0, _seq(5, ids.bos, *([ids.codes[0]] * 64))) == [ids.lut_eos]  # 64 -> LUT_EOS
    assert fn(0, _seq(5, ids.bos, *([ids.codes[0]] * 64), ids.lut_eos)) == [ids.model_eos]


def test_grammar_unsupported_goes_to_eos():
    ids = _ids()
    fn = make_prefix_fn(prompt_len=5, ids=ids)
    assert fn(0, _seq(5, ids.unsupported)) == [ids.model_eos]


def test_codes_from_output_valid_and_refusal():
    ids = _ids()
    valid = _seq(5, ids.bos, *[ids.codes[i] for i in range(64)], ids.lut_eos)
    assert codes_from_output(valid, 5, ids) == list(range(64))
    refusal = _seq(5, ids.unsupported, ids.model_eos)
    assert codes_from_output(refusal, 5, ids) is None
