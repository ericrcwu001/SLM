"""Model/baseline invocation wrappers (docs/eval_harness_implementation.md "Baselines").

In this decode-disabled build the harness runs the baselines that need no GPU, no
trained model, and no LUT decoder — they operate purely at the token/boundary level:

  * always-`<unsupported>`            (null baseline #1)
  * always-support-fixed-tokens       (token-level stand-in for identity-all #2)
  * oracle-boundary diagnostic        (#3; uses gold labels, excluded from fair headline)
  * constant-token-sequence           (fixed 64-id line)
  * mock/replay                       (raw outputs from a JSONL fixture)

Everything that needs a decoded LUT or a real model is a GATED adapter that raises a
clear error: LUT-semantic constant (#4/#5), deterministic renderer (#9/#10; also
config-blocked by renderer_baseline.yaml frozen:false), prompted-Qwen/-frontier (#7/#8),
and the warmup/SFT/RS-DPO/GRPO checkpoints (#14-#17).

Each adapter emits a free-generation output; in ``runtime_constrained`` mode the output
is projected onto the grammar via the FSM, so constrained outputs are valid by
construction (spec: constrained syntax validity must be 100%).
"""

from __future__ import annotations

from typing import Optional, Protocol

from .constrained_decoding import LutGrammarFSM
from .output_parsers import format_tokens, parse_output
from .schemas import RawModelOutput
from .vocab import DEFAULT_VOCAB, LUT_BOS, LUT_EOS, SpecialVocab, UNSUPPORTED

FREE_GENERATION = "free_generation"
RUNTIME_CONSTRAINED = "runtime_constrained"

_CANNED_CODE_IDS = [0] * 64  # a fixed, grammar-valid 64-token line (all <lut_000>)


# --- gating errors ---------------------------------------------------------------
class RequiresDecoder(RuntimeError):
    pass


class RequiresModel(RuntimeError):
    pass


class RequiresFrozenConfig(RuntimeError):
    pass


# --- helpers ---------------------------------------------------------------------
def ids_to_text(model_ids: list[int], vocab: SpecialVocab) -> str:
    return " ".join(vocab.id_to_token[i] for i in model_ids)


class BaselineAdapter(Protocol):
    id: str

    def predict(self, row, mode: str, seed: int) -> RawModelOutput: ...  # noqa: ANN001


# --- base implementation ---------------------------------------------------------
class _FreeTextAdapter:
    """Base: subclasses provide ``_free_text``; constrained mode projects via the FSM."""

    id: str = "base"
    diagnostic: bool = False
    fair_headline: bool = True

    def __init__(self, vocab: Optional[SpecialVocab] = None):
        self.vocab = vocab or DEFAULT_VOCAB
        self.fsm = LutGrammarFSM(self.vocab)

    def _free_text(self, row, seed: int) -> Optional[str]:  # noqa: ANN001
        raise NotImplementedError

    def _constrain(self, text: Optional[str]) -> str:
        parsed = parse_output(text)
        if parsed.kind == "unsupported":
            return UNSUPPORTED
        # LUT branch: project whatever code ids we recovered onto a valid 66-token seq
        cand = [self.vocab.bos_id]
        cand += [self.vocab.code_id(i) for i in parsed.token_ids]
        cand += [self.vocab.eos_id]
        proj = self.fsm.project(cand)
        return ids_to_text(proj, self.vocab)

    def predict(self, row, mode: str, seed: int) -> RawModelOutput:  # noqa: ANN001
        text = self._free_text(row, seed)
        if mode == RUNTIME_CONSTRAINED:
            text = self._constrain(text)
        return RawModelOutput(
            row_id=row.id,
            adapter_id=self.id,
            seed=seed,
            mode=mode,
            text=text,
            provenance={"diagnostic": self.diagnostic, "fair_headline": self.fair_headline},
        )


# --- decoder-free baselines ------------------------------------------------------
class AlwaysUnsupportedAdapter(_FreeTextAdapter):
    id = "null_always_unsupported"

    def _free_text(self, row, seed):  # noqa: ANN001
        return UNSUPPORTED


class AlwaysSupportFixedTokensAdapter(_FreeTextAdapter):
    """Emits a fixed valid 64-token line for every prompt (never refuses).

    Token-level stand-in for the identity-all-prompts null baseline (#2): without a
    decoder it cannot render a true identity LUT, but it exercises the boundary/syntax
    stack as an always-support baseline.
    """

    id = "null_always_support_fixed_tokens"

    def _free_text(self, row, seed):  # noqa: ANN001
        return format_tokens(_CANNED_CODE_IDS)


class OracleBoundaryAdapter(_FreeTextAdapter):
    """Diagnostic: refuses iff the row is gold-unsupported (uses gold labels).

    Excluded from fair headline comparisons (#3)."""

    id = "oracle_boundary_diagnostic"
    diagnostic = True
    fair_headline = False

    def _free_text(self, row, seed):  # noqa: ANN001
        return UNSUPPORTED if not row.is_supported else format_tokens(_CANNED_CODE_IDS)


class ConstantTokenSequenceAdapter(_FreeTextAdapter):
    """Emits a fixed, configurable 64-id token line for every prompt."""

    def __init__(self, code_ids: Optional[list[int]] = None, vocab=None, adapter_id="constant_token_sequence"):  # noqa: ANN001
        super().__init__(vocab)
        self.id = adapter_id
        self.code_ids = code_ids if code_ids is not None else _CANNED_CODE_IDS
        if len(self.code_ids) != 64:
            raise ValueError("constant token sequence must have 64 code ids")

    def _free_text(self, row, seed):  # noqa: ANN001
        return format_tokens(self.code_ids)


class MockReplayAdapter(_FreeTextAdapter):
    """Replays raw outputs from a fixture: ``row_id -> text``.

    Missing rows yield an invalid (None) output flagged in provenance, so a partial
    fixture surfaces rather than silently passing.
    """

    id = "mock_replay"

    def __init__(self, outputs_by_row: dict[str, str], vocab=None, adapter_id="mock_replay"):  # noqa: ANN001
        super().__init__(vocab)
        self.id = adapter_id
        self.outputs_by_row = outputs_by_row

    def _free_text(self, row, seed):  # noqa: ANN001
        return self.outputs_by_row.get(row.id)

    def predict(self, row, mode, seed):  # noqa: ANN001
        text = self._free_text(row, seed)
        missing = text is None
        # A missing fixture row must NOT be projected into a valid LUT line: keep it
        # as an invalid (None) output in both modes so a partial fixture surfaces as a
        # syntax failure rather than a fabricated supported emission.
        if mode == RUNTIME_CONSTRAINED and not missing:
            text = self._constrain(text)
        return RawModelOutput(
            row_id=row.id, adapter_id=self.id, seed=seed, mode=mode, text=text,
            provenance={"missing_mock_output": missing},
        )

    @classmethod
    def from_jsonl(cls, path: str, vocab=None, adapter_id="mock_replay") -> "MockReplayAdapter":  # noqa: ANN001
        import json

        mapping: dict[str, str] = {}
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                rid = d.get("row_id") or d.get("id")
                txt = d.get("text", d.get("output"))
                if rid is not None:
                    mapping[rid] = txt
        return cls(mapping, vocab=vocab, adapter_id=adapter_id)


# --- gated adapters (raise until decoder/model/config exist) ---------------------
class _GatedAdapter:
    id = "gated"
    error_type: type = RequiresModel
    message = "gated adapter unavailable in the decode-disabled spine"

    def __init__(self, adapter_id: Optional[str] = None):
        if adapter_id:
            self.id = adapter_id

    def predict(self, row, mode, seed):  # noqa: ANN001
        raise self.error_type(f"{self.id}: {self.message}")


class ConstantLutAdapter(_GatedAdapter):
    id = "constant_lut"
    error_type = RequiresDecoder
    message = "train-mean/dev-optimized constant LUT needs the frozen VQ decoder to map LUT->tokens"


class DeterministicRendererAdapter(_GatedAdapter):
    id = "deterministic_renderer"
    error_type = RequiresFrozenConfig
    message = (
        "deterministic renderer is blocked: renderer_baseline.yaml is frozen:false with "
        "TODO version/code_sha256, and rendering to a LUT needs the decoder"
    )


class QwenVLAdapter(_GatedAdapter):
    id = "qwen_vl"
    error_type = RequiresModel
    message = "real Qwen2.5-VL inference needs a trained checkpoint + GPU"


class PromptedFrontierAdapter(_GatedAdapter):
    id = "prompted_frontier"
    error_type = RequiresModel
    message = "prompted frontier baseline needs configs/model_clients.yaml + API access"


class CheckpointAdapter(_GatedAdapter):
    id = "checkpoint"
    error_type = RequiresModel
    message = "warmup/SFT/RS-DPO/GRPO checkpoints do not exist yet"


# --- registry --------------------------------------------------------------------
def default_decoder_free_adapters() -> list["BaselineAdapter"]:
    return [
        AlwaysUnsupportedAdapter(),
        AlwaysSupportFixedTokensAdapter(),
        OracleBoundaryAdapter(),
        ConstantTokenSequenceAdapter(),
    ]
