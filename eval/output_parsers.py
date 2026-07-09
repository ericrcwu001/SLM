"""Strict output parser (docs/eval_harness_implementation.md "Output Parser").

This is the sole authority for L1 ``syntax_pass`` and for classifying a raw model
output as a refusal / LUT-token sequence / invalid (which feeds the L0 boundary
decision). It never consults gold labels or eval metadata.

Rules (in order):
  1. Strip leading/trailing whitespace only.
  2. If the string is exactly ``<unsupported>``, classify as refusal.
  3. Otherwise the tokenized output must begin with ``<lut_bos>`` and end with
     ``<lut_eos>``.
  4. Count only tokens matching ``^<lut_[0-9]{3}>$``.
  5. Require exactly 64 LUT code tokens.
  6. Require every code integer to be in 0..255.
  7. Reject any unknown token, prose, JSON, or extra content.
  8. Any output mixing ``<unsupported>`` with LUT tokens fails.
"""

from __future__ import annotations

from .schemas import ParsedOutput
from .vocab import CODE_TOKEN_RE, LUT_BOS, LUT_EOS, TOKEN_COUNT, UNSUPPORTED, code_index


def parse_output(text: str | None) -> ParsedOutput:
    if text is None:
        return ParsedOutput(kind="invalid", parser_errors=["null_output"])

    s = text.strip()  # rule 1

    # rule 2: exact refusal
    if s == UNSUPPORTED:
        return ParsedOutput(kind="unsupported", token_count=0, syntax_pass=True)

    if not s:
        return ParsedOutput(kind="invalid", parser_errors=["empty_output"])

    tokens = s.split()  # whitespace tokenization (incl. internal newlines)
    errors: list[str] = []

    # rule 8: any mix of <unsupported> with other content is invalid
    if UNSUPPORTED in tokens:
        return ParsedOutput(kind="invalid", parser_errors=["mixed_unsupported_and_tokens"])

    # rule 3: must be bracketed by BOS ... EOS
    if tokens[0] != LUT_BOS:
        errors.append("missing_bos")
    if tokens[-1] != LUT_EOS:
        errors.append("missing_eos")
    if errors:
        return ParsedOutput(kind="invalid", parser_errors=errors)

    middle = tokens[1:-1]

    # rules 4-7: every middle token is a well-formed code token, exactly 64 of them,
    # each in 0..255, nothing else.
    code_ids: list[int] = []
    for tok in middle:
        if tok in (LUT_BOS, LUT_EOS):
            errors.append(f"stray_control_token:{tok}")
            continue
        if not CODE_TOKEN_RE.match(tok):
            errors.append(f"unknown_token:{tok}")
            continue
        idx = code_index(tok)
        if idx is None:  # 3-digit but out of 0..255 (e.g. <lut_256>..<lut_999>)
            errors.append(f"code_out_of_range:{tok}")
            continue
        code_ids.append(idx)

    if len(middle) != TOKEN_COUNT or len(code_ids) != TOKEN_COUNT:
        errors.append(f"token_count_{len(code_ids)}_not_{TOKEN_COUNT}")

    if errors:
        return ParsedOutput(
            kind="invalid",
            token_ids=code_ids,
            token_count=len(code_ids),
            parser_errors=errors,
        )

    return ParsedOutput(
        kind="lut_tokens",
        token_ids=code_ids,
        token_count=len(code_ids),
        syntax_pass=True,
    )


def is_refusal(parsed: ParsedOutput) -> bool:
    return parsed.kind == "unsupported"


def format_tokens(code_ids: list[int]) -> str:
    """Inverse helper: build a canonical output line from 64 codebook indices."""
    from .vocab import code_token

    body = " ".join(code_token(i) for i in code_ids)
    return f"{LUT_BOS} {body} {LUT_EOS}"
