"""Known-answer tests for the grammar FSM (runtime-constrained decoding)."""

import numpy as np

from eval.baseline_adapters import ids_to_text
from eval.constrained_decoding import (
    GrammarViolation,
    LutGrammarFSM,
    unsupported_sequence,
    valid_lut_sequence,
)
from eval.output_parsers import parse_output
from eval.vocab import DEFAULT_VOCAB

V = DEFAULT_VOCAB
FSM = LutGrammarFSM(V)


def test_valid_lut_sequence_validates():
    seq = valid_lut_sequence(list(range(64)), V)
    assert len(seq) == 66
    assert FSM.validate_sequence(seq) is True


def test_unsupported_validates():
    assert FSM.validate_sequence(unsupported_sequence(V)) is True


def test_63_codes_invalid():
    seq = [V.bos_id] + [V.code_id(0)] * 63 + [V.eos_id]
    assert FSM.validate_sequence(seq) is False


def test_65_codes_invalid():
    seq = [V.bos_id] + [V.code_id(0)] * 65 + [V.eos_id]
    assert FSM.validate_sequence(seq) is False


def test_token_after_eos_invalid():
    seq = valid_lut_sequence(list(range(64)), V) + [V.code_id(0)]
    assert FSM.validate_sequence(seq) is False


def test_bad_first_token_invalid():
    assert FSM.validate_sequence([V.code_id(0)] + [V.code_id(0)] * 63 + [V.eos_id]) is False


def test_step_raises_on_violation():
    st = FSM.start_state()
    try:
        FSM.step(st, V.eos_id)  # eos not allowed at start
        assert False, "expected GrammarViolation"
    except GrammarViolation:
        pass


def test_project_pads_and_is_valid():
    proj = FSM.project([V.bos_id] + [V.code_id(5)] * 10)
    assert FSM.validate_sequence(proj) is True
    assert len(proj) == 66


def test_project_unsupported_branch():
    proj = FSM.project([V.unsupported_id])
    assert proj == [V.unsupported_id]


def test_mask_logits_start_state():
    logits = np.zeros(V.vocab_size)
    masked = FSM.mask_logits(logits, FSM.start_state())
    allowed = {V.bos_id, V.unsupported_id}
    for tid in range(V.vocab_size):
        if tid in allowed:
            assert masked[tid] == 0.0
        else:
            assert masked[tid] == -np.inf


def test_mask_logits_terminal_raises():
    # advance to a terminal state (done_unsupported) then mask -> must raise, not
    # return an all -inf vector that argmaxes to a garbage token.
    st = FSM.step(FSM.start_state(), V.unsupported_id)
    assert FSM.is_terminal(st)
    logits = np.zeros(V.vocab_size)
    try:
        FSM.mask_logits(logits, st)
        assert False, "expected GrammarViolation on terminal-state mask"
    except GrammarViolation:
        pass


def test_constrained_projection_gives_100pct_valid():
    """Any candidate (even garbage) projects to a parser-valid rendered string."""
    candidates = [
        [V.bos_id] + [V.code_id(3)] * 64 + [V.eos_id],   # already valid
        [V.bos_id] + [V.code_id(3)] * 10,                # too few
        [V.bos_id] + [V.code_id(3)] * 200,               # too many
        [V.unsupported_id],                              # refusal
        [V.eos_id, V.eos_id],                            # garbage -> LUT branch default
        [],                                              # empty -> LUT branch default
    ]
    for cand in candidates:
        proj = FSM.project(cand)
        assert FSM.validate_sequence(proj) is True
        text = ids_to_text(proj, V)
        assert parse_output(text).kind in ("lut_tokens", "unsupported")
