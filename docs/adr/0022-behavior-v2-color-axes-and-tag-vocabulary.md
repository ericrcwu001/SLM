# behavior_v2 Color Axes And Unified Tag Vocabulary

Status: Accepted. Amends ADR 0004 (teacher prompt quality gates). Supersedes the two-axis color
ontology in `docs/detailed_behavior_spec.md` "Supported Prompt Space". Related: ADR 0021.

The measured behavior vector encodes color only as `temperature_delta_b` (Lab b*) and
`tint_delta_a` (Lab a*). The whole hue circle collapses onto these two axes, so "red / orange /
teal / Mars-ish" have no representation and cannot be labeled or scored
(`docs/AUDIT_claude_codex_prompt_to_lut.md` F1; "M1 — absolute-hue axis" is called the in-scope
unlock). Separately, three divergent tag sets exist (`instruction_gen._TAG_BEHAVIOR`,
`frontier_scoring.TAG_DIRECTIONS`, `docs/eval_harness_implementation.md`), which disagree by
construction.

Decision:

- Extend the behavior vector to **`behavior_v2`** with: global hue-angle + magnitude,
  per-tone-region hue (shadow/mid/highlight), per-hue saturation (by input hue sector),
  contrast-shape (toe/shoulder), and matte as a first-class axis. All are derivable from
  `eval/color_pipeline.py` (`hue_deg`, `chroma`, region masks); no new probe is required. All 27
  `behavior_v1` fields are retained for reproducibility.
- Adopt **one unified tag vocabulary** (the table in `docs/attribute_spec.md` §10) as the single
  source of truth, retiring the divergent aliases (`more_magenta`→`tint_magenta`,
  `higher_contrast`→`more_contrast`, `desaturated`→`muted`, …).
- **Version bump:** `BEHAVIOR_VECTOR_VERSION → "behavior_v2"`. Because the pipeline cache-currency
  check keys on `QUALITY_FILTER_VERSION` (`data_pipeline/run_pipeline.py:182`), NOT on the behavior
  version, the code phase MUST bump `QUALITY_FILTER_VERSION` too, or re-measurement silently will
  not run.

Consequences: hue-named and per-hue-saturation looks become expressible and backable for the first
time; the interpreter, the generator's supervision, and the eval direction-checks all draw from one
axis/tag table. Re-measurement writes a NEW versioned `measured_behavior`; the frozen corpus and
tokenizer are untouched (ADR 0026).
