"""Tests for baseline adapters: decoder-free behavior, missing-row integrity, gating."""

import pytest

from eval import baseline_adapters as ba
from eval.output_parsers import parse_output
from eval.schemas import EvalRow

SUP = EvalRow(id="s1", instruction="warmer", is_supported=True)
UNSUP = EvalRow(id="u1", instruction="recolor shirt", is_supported=False,
                unsupported_category="semantic_object_recolor")


def test_always_unsupported():
    a = ba.AlwaysUnsupportedAdapter()
    for mode in (ba.FREE_GENERATION, ba.RUNTIME_CONSTRAINED):
        out = a.predict(SUP, mode, 1234)
        assert parse_output(out.text).kind == "unsupported"


def test_always_support_fixed_tokens_never_refuses():
    a = ba.AlwaysSupportFixedTokensAdapter()
    out = a.predict(UNSUP, ba.FREE_GENERATION, 1234)
    assert parse_output(out.text).kind == "lut_tokens"


def test_oracle_boundary_uses_gold():
    a = ba.OracleBoundaryAdapter()
    assert parse_output(a.predict(SUP, ba.FREE_GENERATION, 1).text).kind == "lut_tokens"
    assert parse_output(a.predict(UNSUP, ba.FREE_GENERATION, 1).text).kind == "unsupported"
    assert a.diagnostic is True and a.fair_headline is False


def test_mock_replay_missing_row_is_invalid_not_fabricated():
    # empty fixture: every row is "missing" and must NOT be projected into a valid LUT,
    # in EITHER mode (integrity: absent output must surface as a syntax failure).
    a = ba.MockReplayAdapter({})
    for mode in (ba.FREE_GENERATION, ba.RUNTIME_CONSTRAINED):
        out = a.predict(SUP, mode, 1234)
        assert out.provenance["missing_mock_output"] is True
        assert parse_output(out.text).kind == "invalid"


def test_mock_replay_present_row_constrained_is_valid():
    from eval.output_parsers import format_tokens

    a = ba.MockReplayAdapter({"s1": format_tokens(list(range(64)))})
    out = a.predict(SUP, ba.RUNTIME_CONSTRAINED, 1234)
    assert parse_output(out.text).kind == "lut_tokens"


@pytest.mark.parametrize("adapter,err", [
    (ba.ConstantLutAdapter(), ba.RequiresDecoder),
    (ba.DeterministicRendererAdapter(), ba.RequiresFrozenConfig),
    (ba.QwenVLAdapter(), ba.RequiresModel),
    (ba.PromptedFrontierAdapter(), ba.RequiresModel),
    (ba.CheckpointAdapter(), ba.RequiresModel),
])
def test_gated_adapters_raise(adapter, err):
    with pytest.raises(err):
        adapter.predict(SUP, ba.RUNTIME_CONSTRAINED, 1234)
