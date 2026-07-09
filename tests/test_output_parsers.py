"""Known-answer tests for the strict output parser (L1)."""

from eval.output_parsers import format_tokens, parse_output
from eval.vocab import code_token


def _valid_line(n_codes=64, code=42):
    return "<lut_bos> " + " ".join([code_token(code)] * n_codes) + " <lut_eos>"


def test_valid_64_token_line():
    p = parse_output(_valid_line())
    assert p.kind == "lut_tokens"
    assert p.syntax_pass is True
    assert p.token_count == 64
    assert p.token_ids == [42] * 64
    assert p.parser_errors == []


def test_exact_unsupported_with_whitespace():
    p = parse_output("   <unsupported>\n")
    assert p.kind == "unsupported"
    assert p.syntax_pass is True
    assert p.token_count == 0


def test_63_tokens_invalid():
    p = parse_output(_valid_line(63))
    assert p.kind == "invalid"
    assert any("token_count" in e for e in p.parser_errors)


def test_65_tokens_invalid():
    p = parse_output(_valid_line(65))
    assert p.kind == "invalid"


def test_code_out_of_range_256():
    line = "<lut_bos> " + " ".join(["<lut_256>"] * 64) + " <lut_eos>"
    p = parse_output(line)
    assert p.kind == "invalid"
    assert any("code_out_of_range" in e for e in p.parser_errors)


def test_two_digit_suffix_is_unknown_token():
    line = "<lut_bos> " + " ".join(["<lut_99>"] * 64) + " <lut_eos>"
    p = parse_output(line)
    assert p.kind == "invalid"
    assert any("unknown_token" in e for e in p.parser_errors)


def test_missing_bos():
    line = " ".join([code_token(1)] * 64) + " <lut_eos>"
    assert parse_output(line).kind == "invalid"
    assert "missing_bos" in parse_output(line).parser_errors


def test_missing_eos():
    line = "<lut_bos> " + " ".join([code_token(1)] * 64)
    assert parse_output(line).kind == "invalid"
    assert "missing_eos" in parse_output(line).parser_errors


def test_prose_in_middle():
    line = "<lut_bos> hello " + " ".join([code_token(1)] * 63) + " <lut_eos>"
    p = parse_output(line)
    assert p.kind == "invalid"
    assert any("unknown_token:hello" in e for e in p.parser_errors)


def test_mixed_unsupported_and_tokens():
    line = "<unsupported> " + _valid_line()
    p = parse_output(line)
    assert p.kind == "invalid"
    assert "mixed_unsupported_and_tokens" in p.parser_errors


def test_empty_and_none():
    assert parse_output("").kind == "invalid"
    assert parse_output(None).kind == "invalid"


def test_json_and_extra_content_rejected():
    assert parse_output('{"lut": [1,2,3]}').kind == "invalid"
    assert parse_output(_valid_line() + " extra").kind == "invalid"


def test_unicode_digit_token_rejected():
    # Arabic-Indic digits (U+06F0..) must NOT be accepted as ASCII code tokens.
    line = "<lut_bos> " + " ".join(["<lut_۰۰۵>"] * 64) + " <lut_eos>"
    p = parse_output(line)
    assert p.kind == "invalid"


def test_format_tokens_roundtrip():
    ids = list(range(64))
    p = parse_output(format_tokens(ids))
    assert p.kind == "lut_tokens"
    assert p.token_ids == ids
