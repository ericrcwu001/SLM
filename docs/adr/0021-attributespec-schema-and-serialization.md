# AttributeSpec Naming, Schema, And Serialization

Status: Accepted. Amends the `CONTEXT.md` glossary. Related: ADR 0020 (two-stage), ADR 0022
(behavior_v2 axes), ADR 0017 (canonical/tokenizer contract, unchanged).

The interpreter↔generator interface needs a name and a frozen schema. "recipe" is rejected: it
collides with four existing uses — the disavowed "JSON recipe" (`CONTEXT.md`), the "recipe window"
calibration term (`docs/detailed_behavior_spec.md`), the "recipe mode" frontier baseline
(`docs/eval_harness_implementation.md`), and `configs/bilevel/recipe.md`.

Decision:

- The interpreter output is **`AttributeSpec`** (instances serialized to **`attribute_spec_text`**).
- It carries a **route enum `{grade, clarify, refuse}`** — `grade` (not `render`) so it does not
  imply the deferred parametric renderer.
- Its axis schema is **`behavior_v2`** (ADR 0022); the same schema the pipeline measures from LUTs
  (`measured_behavior`), so a requested spec and a measured one are directly comparable.
- **Backing rule:** every asserted attribute must be backed by a measurable axis — the
  generalization of `validate_tags_against_behavior` (`data_pipeline/instruction_gen.py:470-497`).
  The input language is unbounded; the asserted axes are the bounded, measurable set. This prevents
  teaching spurious semantic→LUT mappings.
- **Serialization** is deterministic and round-trippable (serialize→parse is identity), with a
  fixed key order and float formatting.

The full field list, control fields (`route`, `confidence`, `out_of_gamut`, `source_text`),
serialization form, and the oracle upper-bound gate are specified in `docs/attribute_spec.md`,
which is the byte-identical source of truth for the schema.
