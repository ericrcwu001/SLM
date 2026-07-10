# AttributeSpec — the interpreter↔generator interface (schema v1)

Status: Design-frozen (spec only; no code in this pass). Governed by ADR 0021 (naming +
schema + serialization + route enum) and ADR 0022 (`behavior_v2` axes + unified tag
vocabulary). Motivated by `docs/AUDIT_claude_codex_prompt_to_lut.md` (§5–§6, §10 "M1/M8").
Related: ADR 0020 (two-stage architecture), ADR 0023 (refuse/clarify boundary), ADR 0017
(frozen canonical/tokenizer contract — unchanged).

## 1. Purpose

`AttributeSpec` is the **structured, high-resolution color-attribute representation** that the
**Interpreter** produces from any natural-language request and that the **Generator** is
conditioned on. It is the single interface between the two stages:

```
user text (+ image) ─▶ Interpreter ─▶ AttributeSpec { route, axes…, confidence, out_of_gamut }
                                          │ route == grade
                                          ▼
              serialize → attribute_spec_text ─(+ image)─▶ Generator ─▶ 64 VQ codes ─▶ frozen decoder ─▶ LUT
```

It replaces the old design in which the **model input was a free-text `instruction`** and the
color language was collapsed onto two Lab axes. See `docs/detailed_behavior_spec.md` "Inputs"
and `docs/master_plan.md` "Stage Artifact Contracts".

## 2. One schema, two provenances

`AttributeSpec` and the pipeline's `measured_behavior` vector share **the same axis schema**
(`behavior_v2`). They differ only in provenance:

| Instance | Provenance | Produced by | Role |
| --- | --- | --- | --- |
| `measured_behavior` | **measured from a LUT** (deterministic probe) | `data_pipeline/behavior_vector.py` | ground truth for dataset construction, captioning, eval |
| `AttributeSpec` (requested) | **inferred from text** | Interpreter | the request handed to the Generator |

This symmetry is deliberate: it lets us (a) validate a requested spec against measurable axes
(the "backing rule", §6), and (b) run the **oracle upper-bound gate** (§8) that compares
`measured_behavior → codes` against `AttributeSpec → codes`.

## 3. Axis schema (`behavior_v2`)

`behavior_v2` extends the frozen `behavior_v1` (27 fields in
`data_pipeline/behavior_vector.py:129-157`) with the color-resolution axes the audit calls the
"in-scope unlock" (M1). Bumping to `behavior_v2` requires bumping BOTH
`BEHAVIOR_VECTOR_VERSION` and `QUALITY_FILTER_VERSION` (see §9). Existing `behavior_v1` fields
are retained unchanged for backward reproducibility.

### 3a. Tonal axes (unchanged from behavior_v1)
`mean_l_delta`, `contrast_l_spread_delta`, `black_point_l_delta`, `shadow_l_delta`,
`highlight_l_delta`.

### 3b. Color axes — legacy 2-axis (retained) + NEW hue resolution
Retained: `temperature_delta_b` (Lab b*), `tint_delta_a` (Lab a*), `chroma_delta`,
`highlight_chroma_delta`, `shadow_chroma_delta`, and the split-tone components
(`split_tone_shadow_a/b`, `split_tone_highlight_a/b`, `split_tone_strength`).

NEW in `behavior_v2` (all derivable from `eval/color_pipeline.py` `hue_deg`/`chroma` + the
existing highlight/shadow masks — no new probe needed):

| Field | Type / range | Meaning |
| --- | --- | --- |
| `global_hue_deg` | float, `[0,360)` | dominant hue angle of the overall cast (absolute hue, not a signed axis) |
| `global_hue_magnitude` | float ≥ 0 | strength of the global cast (mean chroma of the neutral-probe shift) |
| `shadow_hue_deg` / `midtone_hue_deg` / `highlight_hue_deg` | float, `[0,360)` | per-tone-region hue of the cast (split-toning at full hue resolution) |
| `per_hue_saturation` | map: 7 sectors → float | chroma delta binned by INPUT hue sector `{red, orange, yellow, green, cyan, blue, magenta}` (e.g. "crush greens, boost oranges") |
| `contrast_toe_delta` / `contrast_shoulder_delta` | float | contrast *shape* — shadow-toe vs highlight-shoulder, not just global spread |
| `matte_strength` | float ≥ 0 | matte as a first-class axis (black-lift + reduced contrast + slight desat), superseding the "matte recipe window" as a measured quantity |

### 3c. Safety / context axes (unchanged; read-only for the refuse route)
`neutral_drift_deltaE`, `neutral_drift_deltaE_p95`, `skin_locus_deltaE00_mean`,
`skin_locus_deltaE00_p95`, `skin_locus_hue_drift_deg_p95`, `skin_chroma_ratio_min/max`,
`clip_rate`, `smoothness`, `foldover_rate`, `residual_norm`, plus `smoothness_native`
(injected in `run_pipeline.py`).

## 4. AttributeSpec control fields (spec-only, not in `measured_behavior`)

| Field | Type | Meaning |
| --- | --- | --- |
| `attribute_spec_version` | str | `"attribute_spec_v1"` |
| `route` | enum | `grade` \| `clarify` \| `refuse` (§5) |
| `confidence` | map: axis → `[0,1]` (+ `overall`) | interpreter certainty per asserted axis; low overall → `clarify` |
| `out_of_gamut` | bool | the requested look leaves the representable manifold (§5) → `refuse` |
| `refuse_reason` | enum \| null | `out_of_scope` (non-global/local/content) \| `out_of_gamut` (e.g. infrared, pure-primary, hue-rotation) |
| `clarify_options` | list[str] \| null | when `route==clarify`, the concrete supported directions to offer |
| `source_text` | str | the original user prompt (kept as the intent record; never discarded) |

## 5. Route semantics

- **`grade`** — the request maps to a determinate, representable point in axis space. Serialize
  and send to the Generator. (Named `grade`, not `render`, to avoid implying the deferred
  parametric renderer — see ADR 0023.)
- **`clarify`** — the request has a color intent but is under-specified (flat confidence over
  axes), e.g. "make it better", "make it pop" with no direction. Emit `clarify_options`; do not
  fabricate a grade.
- **`refuse`** — emit `<unsupported>`. Two sub-cases via `refuse_reason`:
  - `out_of_scope`: not a global color transform (local/semantic/content/relighting/geometry/
    texture/reference — the existing taxonomy in `data_pipeline/unsupported_gen.py`).
  - `out_of_gamut`: a global color look the **frozen tokenizer cannot represent** — infrared /
    channel-swap, pure single-primary casts, large hue rotations. The boundary **mirrors the
    materialization admission gate** (`scripts/materialize_target_tokens.py:50`, mean ΔE00 ≤ 3.0
    / p95 ≤ 6.0): a look whose nearest representable LUT exceeds it is `out_of_gamut`. The refuse
    route only *reads* this boundary; it never modifies the tokenizer. (AUDIT F3.)

Handling "Mars"-style prompts: partly-representable concepts are **clamped to the nearest
representable point and graded**, not refused; only looks with no acceptable in-gamut match are
refused. Which concepts are supported is defined empirically by which corpus LUTs materialize —
see `docs/data_collection_plan.md` "Captioning".

## 6. Backing rule

Every asserted attribute in a `grade`/`clarify` spec must be **backed by a measurable axis** —
the generalization of `validate_tags_against_behavior`
(`data_pipeline/instruction_gen.py:470-497`). The interpreter may not assert an axis the schema
cannot measure. This is what prevents teaching spurious semantic→LUT mappings (AUDIT §T-D): the
*language* is unbounded, but the *asserted axes* are always the bounded, measurable set.

## 7. Canonical serialization (`attribute_spec_text`)

Deterministic and round-trippable (serialize→parse is identity). One canonical key ordering, fixed
float formatting, omitted-when-zero axes, explicit `route`. Illustrative form:

```
route=grade | warmer=+2.0 muted=+3.0 more_contrast=+1.0 matte=+2.5 \
  shadow_hue=210 highlight_hue=45 split_strength=6.0 | conf=0.82
```

The exact grammar is fixed in ADR 0021 and (later) `data_pipeline/attribute_spec.py`. The
Generator consumes `attribute_spec_text` at the input position currently holding `instruction`
(`sft/example.py:80`); the target side (64 VQ codes) is unchanged.

## 8. Oracle upper-bound gate (hard go/no-go for the build)

Before any interpreter/generator retrain spend: verify that feeding the **ground-truth**
`measured_behavior` (as `attribute_spec_text`) to the Generator reproduces the target codes at or
above the current one-stage token accuracy. If a perfect spec cannot drive the Generator to the
target (the seam is lossy — many LUTs share a summary, AUDIT §9), the two-stage design is
abandoned before P5/P6. This gate lives in the deferred roadmap (P4) but is specified here because
it validates this schema.

## 9. Versioning & consistency

- `attribute_spec_version = "attribute_spec_v1"`; axis schema `behavior_v2`.
- The `behavior_v2` field list in §3 is the **byte-identical source of truth**; it MUST match the
  "Measured Behavior Vector" section of `docs/data_collection_plan.md` and the Supported Prompt
  Attribute entry in `CONTEXT.md`.
- The tag vocabulary in §10 is the single source of truth reconciling the three previously
  divergent sets; it MUST match the direction-check table in
  `docs/eval_harness_implementation.md`.

## 10. Unified tag vocabulary (single source of truth)

Canonical tag → axis mapping. Retired aliases (previously in `frontier_scoring.TAG_DIRECTIONS`
and `eval_harness_implementation.md`) are listed so they are removed everywhere in the code phase.

| Canonical tag(s) | Axis | Retired alias(es) |
| --- | --- | --- |
| `warmer` / `cooler` | `temperature_delta_b` (b*) | — |
| `tint_magenta` / `tint_green` | `tint_delta_a` (a*) | `more_magenta` / `more_green` |
| `brighter` / `darker` | `mean_l_delta` | — |
| `more_contrast` / `less_contrast` | `contrast_l_spread_delta` | `higher_contrast` / `softer_contrast` |
| `more_saturated` / `muted` | `chroma_delta` | `desaturated` (→ `muted`) |
| `lifted_blacks` / `crushed_blacks` | `black_point_l_delta` | — |
| `lifted_shadows` | `shadow_l_delta` | — |
| `brighter_highlights` / `softer_highlights` | `highlight_l_delta` | — |
| `cooler_shadows` / `warmer_shadows` | `shadow_hue_deg` (region) | — |
| `cooler_highlights` / `warmer_highlights` | `highlight_hue_deg` (region) | — |
| `hue_cast_{red,orange,yellow,green,cyan,blue,magenta}` | `global_hue_deg` + `global_hue_magnitude` | NEW (behavior_v2) |
| `sat_{sector}_up` / `sat_{sector}_down` | `per_hue_saturation[sector]` | NEW (behavior_v2) |
| style bundles: `matte`, `faded`, `filmic`, `cinematic`, `teal-orange`, `sepia`, `bleach bypass`, `natural` | measured composites (see `eval/configs/calibration_manifest.json` windows; `matte` also a first-class `matte_strength` axis) | — |

Note: the hue-cast and per-hue-saturation tags are the axes that make "red / orange / teal /
Mars-ish / underwater" **expressible and backable** for the first time. Looks that require angles
or magnitudes outside the representable manifold remain `refuse:out_of_gamut` (§5).
