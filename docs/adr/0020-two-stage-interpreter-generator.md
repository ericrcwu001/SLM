# Two-Stage Interpreter + Generator Architecture

Status: Accepted. Amends ADR 0008 (staged VLM training scope). Supersedes the one-stage
assumption in `docs/behavior_spec.md` and `docs/detailed_behavior_spec.md`. Related: ADR 0021
(AttributeSpec), ADR 0025 (training sequence v2).

The one-stage design — `(image, natural-language instruction) → Qwen2.5-VL-3B QLoRA → 64 VQ
codes` — cannot map open or metaphorical language ("make it look like Mars / cowboy / underwater")
and fails silently: it emits a conventional muted LUT for requests it should decompose or refuse.
The reconciled audit (`docs/AUDIT_claude_codex_prompt_to_lut.md` §5–§6) traces this to a color
ontology collapsed onto two Lab axes, a closed ~22-tag teacher vocabulary, and a refuse path that
never trains.

Decision: split the system into two stages.

1. **Interpreter** — a small, separately-trained model (see ADR 0021 for its output; ADR 0025 for
   training) that maps any user text to a structured `AttributeSpec` plus a route
   `∈ {grade, clarify, refuse}`. All linguistic diversity and world knowledge live here.
2. **Generator** — the SAME Qwen2.5-VL-3B QLoRA model, conditioned on the serialized
   `attribute_spec_text` (+ image) instead of a free-text instruction, still emitting 64 VQ codes.
   Its output contract and locked knobs (`AGENTS.md:39-44`) are unchanged.

The VQ tokenizer, its decoder, the `.cube`/runtime contract (ADR 0017), and the LUT/image corpus
remain frozen and immutable. The Generator never receives raw open-vocabulary language; the
Interpreter never touches the frozen stack.

Consequences: the language problem is attacked where it is cheap (a small text model, iterable
without the expensive VLM run); the Generator's job narrows to rendering a bounded, backable spec;
and out-of-scope / out-of-gamut requests become explicit refusals rather than silent wrong LUTs.
The interface schema is frozen in `docs/attribute_spec.md`.
