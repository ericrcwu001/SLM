"""Single-source check for the pinned tokenizer identity constants.

CODEBOOK_SIZE / TOKEN_COUNT are copy-declared as literals in three modules
(tokenizer.config, eval.vocab, data_pipeline.constants) and GRID is imported from
eval.cube_io. They MUST agree: a one-file edit that desyncs them would silently corrupt
the tokenizer identity (token grammar vs codebook vs pipeline). This test fails loudly if
they ever drift, which is the cheap alternative to rewiring all three onto one import.
"""

from __future__ import annotations

from tokenizer import config as tcfg


def test_codebook_and_token_count_agree_across_modules():
    from data_pipeline import constants as dpc
    from eval import vocab

    assert tcfg.CODEBOOK_SIZE == vocab.CODEBOOK_SIZE == dpc.CODEBOOK_SIZE == 256
    assert tcfg.TOKEN_COUNT == vocab.TOKEN_COUNT == dpc.TOKEN_COUNT == 64
    # the special vocab is 3 control tokens + one per codebook entry
    assert vocab.NUM_SPECIAL_TOKENS == tcfg.CODEBOOK_SIZE + 3


def test_grid_single_sourced_and_latent_invariant():
    from eval.cube_io import GRID_SIZE

    assert tcfg.GRID == GRID_SIZE == 17
    assert tcfg.LATENT_GRID ** 3 == tcfg.TOKEN_COUNT
