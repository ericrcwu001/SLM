"""Token-id grammar FSM for runtime-constrained decoding.

Implements the grammar in docs/eval_harness_implementation.md "Constrained Decoding"
and model_architecture.md "Output Grammar":

    valid first token set: <unsupported> or <lut_bos>
    if <unsupported> is emitted:  only EOS may follow
    if <lut_bos> is emitted:      positions 1-64 allow only <lut_000>..<lut_255>
                                  position 65 allows only <lut_eos>
                                  only EOS may follow <lut_eos>

The mask enforces *syntax only* — it never consults gold support labels, inferred
prompt attributes, or eval metadata, so false-support and over-refusal stay
measurable. In runtime-constrained mode syntax validity must be 100%; any failure is
an implementation bug (spec "Pass Criteria").

Operates over model-vocab ids from :class:`eval.vocab.SpecialVocab`. ``eos_token_id``
is the base model's end-of-text id (unknown until the tokenizer is resized); when it
is None the FSM treats <lut_eos>/<unsupported> completion as terminal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .vocab import DEFAULT_VOCAB, TOKEN_COUNT, SpecialVocab


class GrammarViolation(ValueError):
    """Raised when a token id is not permitted by the grammar at the current state."""


@dataclass(frozen=True)
class FSMState:
    name: str  # start | in_lut | await_eos | done_lut | done_unsupported
    count: int = 0  # number of code tokens emitted so far (in_lut)

    @property
    def is_terminal(self) -> bool:
        return self.name in ("done_lut", "done_unsupported")


class LutGrammarFSM:
    def __init__(self, vocab: SpecialVocab | None = None, eos_token_id: Optional[int] = None):
        self.vocab = vocab or DEFAULT_VOCAB
        self.eos_token_id = eos_token_id
        self._code_ids = frozenset(self.vocab.code_ids)

    # --- state machine -----------------------------------------------------
    def start_state(self) -> FSMState:
        return FSMState("start")

    def allowed_token_ids(self, state: FSMState) -> frozenset[int]:
        v = self.vocab
        if state.name == "start":
            return frozenset({v.bos_id, v.unsupported_id})
        if state.name == "in_lut":
            return self._code_ids
        if state.name == "await_eos":
            return frozenset({v.eos_id})
        # terminal
        return frozenset({self.eos_token_id}) if self.eos_token_id is not None else frozenset()

    def step(self, state: FSMState, token_id: int) -> FSMState:
        v = self.vocab
        if state.name == "start":
            if token_id == v.bos_id:
                return FSMState("in_lut", 0)
            if token_id == v.unsupported_id:
                return FSMState("done_unsupported")
            raise GrammarViolation(f"start: token {token_id} not in {{bos, unsupported}}")
        if state.name == "in_lut":
            if token_id in self._code_ids:
                nxt = state.count + 1
                return FSMState("await_eos") if nxt == TOKEN_COUNT else FSMState("in_lut", nxt)
            raise GrammarViolation(f"in_lut[{state.count}]: token {token_id} is not a code id")
        if state.name == "await_eos":
            if token_id == v.eos_id:
                return FSMState("done_lut")
            raise GrammarViolation(f"await_eos: token {token_id} != lut_eos")
        # terminal: only the model EOS may follow, and it is absorbed
        if self.eos_token_id is not None and token_id == self.eos_token_id:
            return state
        raise GrammarViolation(f"{state.name}: no token may follow (got {token_id})")

    def is_terminal(self, state: FSMState) -> bool:
        return state.is_terminal

    # --- whole-sequence helpers -------------------------------------------
    def validate_sequence(self, token_ids: list[int]) -> bool:
        """True iff the id sequence is grammar-valid and reaches a terminal state.

        A single trailing ``eos_token_id`` (if configured) is permitted.
        """
        state = self.start_state()
        try:
            for tid in token_ids:
                state = self.step(state, tid)
        except GrammarViolation:
            return False
        return state.is_terminal

    def project(self, candidate_token_ids: list[int]) -> list[int]:
        """Project arbitrary candidate ids onto the nearest grammar-valid sequence.

        Stand-in for constrained generation: at each step the candidate token is kept
        if allowed, else a deterministic default (first allowed) is substituted. The
        result is always ``validate_sequence``-true. Used by the constrained eval mode
        so that any adapter's constrained output is syntactically valid by construction.
        """
        v = self.vocab
        first = candidate_token_ids[0] if candidate_token_ids else v.bos_id
        if first == v.unsupported_id:
            return [v.unsupported_id]
        out = [v.bos_id]
        default_code = v.code_id(0)
        codes = [t for t in candidate_token_ids if t in self._code_ids][:TOKEN_COUNT]
        codes += [default_code] * (TOKEN_COUNT - len(codes))
        out.extend(codes)
        out.append(v.eos_id)
        return out

    def mask_logits(self, logits, state: FSMState):
        """Return a copy of ``logits`` with disallowed vocab ids set to -inf.

        ``logits`` is a 1-D array indexed by model-vocab id. Only entries in the
        current state's allowed set (plus any configured EOS in terminal states)
        survive.
        """
        import numpy as np

        masked = np.array(logits, dtype=np.float64, copy=True)
        allowed = self.allowed_token_ids(state)
        if not allowed:
            # terminal state with no explicit EOS configured: nothing may follow.
            # Masking here would produce an all -inf vector (argmax -> garbage), so a
            # decode loop must stop instead. Fail loudly rather than silently.
            raise GrammarViolation(
                "mask_logits called on a terminal state with no allowed tokens; "
                "check is_terminal() and stop generation first"
            )
        keep = np.zeros(masked.shape[0], dtype=bool)
        for tid in allowed:
            if 0 <= tid < masked.shape[0]:
                keep[tid] = True
        masked[~keep] = -np.inf
        return masked


def valid_lut_sequence(code_ids: list[int], vocab: SpecialVocab | None = None) -> list[int]:
    """Build a grammar-valid model-id sequence from 64 codebook indices."""
    v = vocab or DEFAULT_VOCAB
    if len(code_ids) != TOKEN_COUNT:
        raise ValueError(f"need {TOKEN_COUNT} code ids, got {len(code_ids)}")
    return [v.bos_id, *[v.code_id(i) for i in code_ids], v.eos_id]


def unsupported_sequence(vocab: SpecialVocab | None = None) -> list[int]:
    v = vocab or DEFAULT_VOCAB
    return [v.unsupported_id]
