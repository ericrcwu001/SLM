"""Special LUT vocabulary and its (provisional) token-id map.

The prompt-to-LUT model adds 259 special tokens on top of the base Qwen2.5-VL
tokenizer (model_architecture.md "Output Vocabulary" / "Vocabulary Resize And
Embedding Preflight"):

    <lut_bos>  <lut_eos>  <unsupported>  <lut_000> ... <lut_255>

= 3 control tokens + 256 codebook tokens = 259.

The *absolute* model-vocabulary ids only exist after the base tokenizer is resized
(training Stage 3), which has not happened. Until then this module assigns a
deterministic **provisional** id map so the FSM, CLI, and version manifest have a
concrete, self-consistent contract to operate on. Swap `base_offset` for the real
`len(base_tokenizer)` when the resized tokenizer is available; nothing else changes.

Pinned per the approved plan / cross-doc audit:
  * token_suffix ("000".."255") -> codebook_index  == identity.
  * code-token surface form: ``<lut_%03d>`` with a zero-padded 3-digit suffix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

LUT_BOS = "<lut_bos>"
LUT_EOS = "<lut_eos>"
UNSUPPORTED = "<unsupported>"

CODEBOOK_SIZE = 256
NUM_SPECIAL_TOKENS = CODEBOOK_SIZE + 3  # 259
TOKEN_COUNT = 64  # exactly 64 code tokens per supported output

# Canonical surface form + strict matcher for a code token.
# ASCII digits only: Python's \d is Unicode-aware and would match e.g. Arabic-Indic
# digits, which are NOT in the 259-token ASCII vocabulary. Use [0-9] explicitly.
CODE_TOKEN_RE = re.compile(r"^<lut_([0-9]{3})>$")


def code_token(index: int) -> str:
    """Codebook index (0..255) -> surface token, e.g. 42 -> '<lut_042>'."""
    if not 0 <= index < CODEBOOK_SIZE:
        raise ValueError(f"codebook index out of range [0,{CODEBOOK_SIZE}): {index}")
    return f"<lut_{index:03d}>"


def code_index(token: str) -> int | None:
    """Surface token -> codebook index, or None if not a well-formed code token."""
    m = CODE_TOKEN_RE.match(token)
    if not m:
        return None
    idx = int(m.group(1))
    return idx if 0 <= idx < CODEBOOK_SIZE else None


@dataclass(frozen=True)
class SpecialVocab:
    """The 259 special tokens with a provisional, swappable id map.

    ``base_offset`` is where the special tokens begin in the resized model
    vocabulary. It is 0 in the provisional map used here and becomes
    ``len(base_tokenizer)`` once the real tokenizer is resized.
    """

    base_offset: int = 0

    # Derived maps (built in __post_init__ via object.__setattr__ since frozen).
    token_to_id: dict[str, int] = field(default_factory=dict, compare=False)
    id_to_token: dict[int, str] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        tokens = [LUT_BOS, LUT_EOS, UNSUPPORTED] + [
            code_token(i) for i in range(CODEBOOK_SIZE)
        ]
        t2i: dict[str, int] = {}
        i2t: dict[int, str] = {}
        for local, tok in enumerate(tokens):
            tid = self.base_offset + local
            t2i[tok] = tid
            i2t[tid] = tok
        object.__setattr__(self, "token_to_id", t2i)
        object.__setattr__(self, "id_to_token", i2t)

    # --- control-token ids -------------------------------------------------
    @property
    def bos_id(self) -> int:
        return self.token_to_id[LUT_BOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[LUT_EOS]

    @property
    def unsupported_id(self) -> int:
        return self.token_to_id[UNSUPPORTED]

    # --- codebook-token ids ------------------------------------------------
    def code_id(self, index: int) -> int:
        """Codebook index (0..255) -> model-vocab id."""
        return self.token_to_id[code_token(index)]

    @property
    def code_ids(self) -> list[int]:
        return [self.code_id(i) for i in range(CODEBOOK_SIZE)]

    def id_to_codebook_index(self, token_id: int) -> int | None:
        tok = self.id_to_token.get(token_id)
        return None if tok is None else code_index(tok)

    def is_code_id(self, token_id: int) -> bool:
        return self.id_to_codebook_index(token_id) is not None

    # --- manifest helpers --------------------------------------------------
    @property
    def all_tokens(self) -> list[str]:
        return list(self.token_to_id.keys())

    @property
    def added_special_token_ids(self) -> dict[str, int]:
        """The 259 special tokens -> provisional ids, for the version manifest."""
        return dict(self.token_to_id)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)  # 259

    @staticmethod
    def token_suffix_to_codebook_index() -> str:
        """Pinned mapping identifier (identity)."""
        return "identity"


# A module-level default instance for the provisional (offset-0) map.
DEFAULT_VOCAB = SpecialVocab(base_offset=0)
