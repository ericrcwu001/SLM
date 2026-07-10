# Eval Harness Implementation Document

## Purpose

The eval harness is built before training and is the source of truth for whether
the fine-tuned model actually learned the target behavior. It evaluates the same
frozen rows across baselines, warmup checkpoints, SFT checkpoints, RS/DPO
checkpoints, and GRPO checkpoints.

The primary metric is `supported_prompt_to_lut_pass_rate` on headline-eligible
rows. A supported case passes only when the boundary (non-refusal), grammar,
decoder, color-direction, target-fidelity, LUT safety, skin-locus, and style
checks all pass — i.e. the full L0-L7 deterministic stack. The L8 judge score is
recorded but cannot override an L0-L7 result. An unsupported case passes only
when the output is exactly `<unsupported>`.

## Eval Unit

One supported eval row is:

```json
{
  "id": "eval_seen_000001",
  "image_path": "images/eval_seen/000001.jpg",
  "image_sha256": "...",
  "instruction": "Make the image warmer with softer contrast.",
  "is_supported": true,
  "support_label": "supported",
  "gold_tags": ["warmer", "less_contrast"],
  "style_bundle": null,
  "style_primary": null,
  "target_lut_path": "luts/eval/000001.npy",
  "target_image_path": "targets/eval/000001.png",
  "target_tokens": [42, 17, 200, 5, "... 64 code ids total ...", 128],
  "acceptance_mode": "exact_target",
  "reference_lut_paths": [],
  "reference_tokens": [],
  "behavior_window": null,
  "source_lut_id": "ppr10k_group_001_expert_a_lut_002",
  "source_family": "ppr10k",
  "canonical_domain_id": "slm_lut_v1_srgb_display_encoded_17_trilinear",
  "canonical_absolute_lut_hash": "...",
  "canonical_residual_lut_hash": "...",
  "tokenizer_version": "...",
  "vq_codebook_sha256": "...",
  "vq_decoder_sha256": "...",
  "representability_tier": "gold",
  "headline_eligible": true,
  "procedural_filler": false,
  "usage_weight": 1.0,
  "split": "eval_usage_weighted_headline",
  "measured_behavior": {
    "temperature_delta_b": 2.3,
    "contrast_l_spread_delta": -3.1
  },
  "derived_lut_quality": {
    "representability_status": "accepted",
    "fit_deltaE00_mean": 1.4,
    "fit_deltaE00_p95": 4.8,
    "supported_cell_rate": 0.99
  },
  "metadata": {
    "has_people": true,
    "scene_cluster": 42,
    "source_hash": "...",
    "prompt_template_family": "explicit_compound",
    "prompt_generation_batch_id": "...",
    "teacher_model_version": "...",
    "split_unit_id": "..."
  }
}
```

Unsupported and mixed rows use:

```json
{
  "is_supported": false,
  "support_label": "unsupported",
  "unsupported_category": "mixed_partial_supported_plus_content_generation",
  "unsupported_components": ["content_removal"],
  "supported_components": ["warmer", "less_contrast"],
  "mixed_prompt": true,
  "boundary_pair_id": "mixed_boundary_001",
  "boundary_pair_role": "unsupported_mixed",
  "target_lut_path": null,
  "target_tokens": [],
  "gold_tags": []
}
```

Gold tags are frozen at eval construction time. They are never inferred from the
model output during scoring.

## Harness Modules

Recommended file/module layout:

```text
eval/
  run_eval.py
  schemas.py
  output_parsers.py
  constrained_decoding.py
  lut_decoder.py
  cube_io.py
  color_pipeline.py
  deterministic_checks.py
  target_fidelity.py
  unsupported_metrics.py
  judge_client.py
  baseline_adapters.py
  stats.py
  report.py
  configs/
    eval_default.yaml
    calibration_manifest.json
    gating_slice_registry.yaml
```

Responsibilities:

- `schemas.py`: row schema, model-output schema, metric schema, manifest schema.
- `output_parsers.py`: strict token/refusal parser.
- `constrained_decoding.py`: token-id grammar mask/FSM for runtime mode.
- `lut_decoder.py`: maps 64 token ids to residual LUT through frozen VQ decoder.
- `cube_io.py`: validates and writes canonical `.cube` files with pinned
  serialization.
- `color_pipeline.py`: ICC-aware sRGB/Lab/CIEDE2000 conversions.
- `deterministic_checks.py`: direction, style, skin-locus, and safety checks.
- `target_fidelity.py`: target image/chart DeltaE checks.
- `unsupported_metrics.py`: refusal, over-refusal, coverage, boundary F1.
- `judge_client.py`: LLM/VLM-as-judge scoring.
- `baseline_adapters.py`: model invocation wrappers for baselines and tuned modes.
- `stats.py`: Wilson intervals, paired bootstrap, paired tests, seed summaries.
- `report.py`: CSV/JSON/Markdown result tables.

## Evaluation Layers

| Layer | Name | Pass Rule |
| --- | --- | --- |
| L0 | Boundary | Gold unsupported passes only with exact `<unsupported>`; gold supported fails if refused |
| L1 | Syntax | Supported output has only BOS, 64 valid LUT tokens, EOS |
| L2 | Decode/export | Tokens decode to finite canonical 17x17x17 residual LUT and export valid `.cube` |
| L3 | Tokenizer gate | Frozen tokenizer passes mean, tail, per-family, and per-target reconstruction gates |
| L4 | Direction | Every gold tag moves in the correct measured direction and minimum magnitude window |
| L5 | Target fidelity | Acceptance_mode selects the fidelity gate: exact single decoded-target DeltaE, any of K decoded reference LUTs, or a predeclared behavior window |
| L6 | LUT safety | Clip, out-of-range, smoothness, foldover, neutral drift, and skin-locus gates pass |
| L7 | Style recipe | Style rows pass recipe windows and discriminability checks; underspecified style rows may accept fidelity via multi-reference or behavior-window instead of exact target |
| L8 | Judge | LLM/VLM judge score recorded; cannot override deterministic failure |

The final pass/fail for supported rows is:

```text
supported_pass =
  boundary_pass
  and syntax_pass
  and decode_pass
  and direction_pass
  and fidelity_pass
  and safety_pass
  and style_recipe_pass

fidelity_pass is selected by acceptance_mode:
  exact_target                       -> target_fidelity_pass
  multi_reference                    -> multi_reference_pass
  behavior_window                    -> behavior_window_pass
  multi_reference | behavior_window  -> multi_reference_pass or behavior_window_pass

safety_pass, direction_pass, and style_recipe_pass are required in every acceptance_mode.
Rows that omit acceptance_mode default to exact_target.
```

where `boundary_pass` for a supported row is the L0 boundary check: the output
is not a refusal (not `<unsupported>`). A refused supported row also fails
`syntax_pass`, so this term makes the L0 non-refusal requirement explicit and
adds no new machinery.

The final pass/fail for unsupported rows is:

```text
unsupported_pass = exact_output == "<unsupported>"
```

## Constrained Decoding

CLI/product decoding must use grammar-constrained token-id decoding.

```text
valid first token set:
  <unsupported> or <lut_bos>

if <unsupported> is emitted:
  only EOS may follow

if <lut_bos> is emitted:
  positions 1-64 allow only <lut_000> through <lut_255>
  position 65 allows only <lut_eos>
  only EOS may follow <lut_eos>
```

The grammar mask must not use gold support labels, inferred prompt attributes, or
eval metadata. It only enforces output syntax, so false support and over-refusal
remain measurable.

Two eval modes are required:

```text
free_generation_eval:
  do_sample=false
  num_beams=1
  no grammar mask
  strict parser scores learned syntax validity

runtime_constrained_eval:
  do_sample=false
  num_beams=1
  grammar mask enabled
  syntax validity should be 100%; failures are implementation bugs
```

The SFT `valid_token_rate` gate is measured in free-generation eval only.
Runtime constrained eval is the product path.

## Output Parser

Strict parser rules:

- Strip leading/trailing whitespace only.
- If the string is exactly `<unsupported>`, classify as refusal.
- Otherwise require tokenized output to begin with `<lut_bos>` and end with
  `<lut_eos>`.
- Count only tokens matching `^<lut_[0-9]{3}>$`.
- Require exactly 64 LUT code tokens.
- Require every code token integer to be between 0 and 255.
- Reject any unknown token, prose, JSON, or extra content.

## Color Pipeline

The deterministic evaluator uses:

```text
canonical LUT domain:
  display-referred IEC 61966-2-1 sRGB
  encoded RGB [0,1]
  D65
  17x17x17
  trilinear interpolation

metric pipeline:
  ICC-aware image conversion to canonical sRGB
  linearization where needed
  CIE Lab D65
  CIEDE2000 for DeltaE reporting
```

L2 fails on mismatched canonical domain, interpolation, grid size, axis order,
token flatten order, tokenizer version, `vq_codebook_sha256`,
`vq_decoder_sha256`, ICC conversion config, or `.cube` serialization version.

Direction and safety checks are run on:

- sampled image pixels;
- a fixed synthetic RGB chart;
- neutral-axis samples;
- fixed skin-locus samples;
- optional face/skin masks as qualitative diagnostics only.

The same interpolation method must be used in target generation, scoring,
`graded.png`, and `.cube` roundtrip tests. Use trilinear interpolation for v1.

## Direction Checks

Tags map to canonical `behavior_v2` axes from the unified tag vocabulary, which is
the single source of truth (`docs/attribute_spec.md` §10; ADR 0022). This table
MUST match that mapping. The divergent aliases `more_magenta`/`more_green`
(-> `tint_magenta`/`tint_green`), `higher_contrast`/`softer_contrast`
(-> `more_contrast`/`less_contrast`), and `desaturated` (-> `muted`) are retired.

| Tag | Axis (`behavior_v2`) | Expected Direction |
| --- | --- | --- |
| `warmer` | `temperature_delta_b` (mean Lab b*) | increase |
| `cooler` | `temperature_delta_b` (mean Lab b*) | decrease |
| `tint_magenta` | `tint_delta_a` (mean Lab a*) | increase |
| `tint_green` | `tint_delta_a` (mean Lab a*) | decrease |
| `brighter` | `mean_l_delta` (mean L*) | increase |
| `darker` | `mean_l_delta` (mean L*) | decrease |
| `more_contrast` | `contrast_l_spread_delta` (p95(L*) - p5(L*)) | increase |
| `less_contrast` | `contrast_l_spread_delta` (p95(L*) - p5(L*)) | decrease |
| `more_saturated` | `chroma_delta` (chroma) | increase |
| `muted` | `chroma_delta` (chroma) | decrease |
| `lifted_blacks` | `black_point_l_delta` (p5(L*)) | increase |
| `crushed_blacks` | `black_point_l_delta` (p5(L*)) | decrease |
| `lifted_shadows` | `shadow_l_delta` (low-mask L*) | increase |
| `brighter_highlights` | `highlight_l_delta` (high-mask L*) | increase |
| `softer_highlights` | `highlight_l_delta` (high-mask L* + clip gate) | decrease or roll off |
| `cooler_shadows` | `shadow_hue_deg` (low-mask region hue) | shift toward blue/cyan (teal) |
| `warmer_shadows` | `shadow_hue_deg` (low-mask region hue) | shift toward orange/warm |
| `cooler_highlights` | `highlight_hue_deg` (high-mask region hue) | shift toward blue/cyan |
| `warmer_highlights` | `highlight_hue_deg` (high-mask region hue) | shift toward orange/yellow |
| `hue_cast_{red,orange,yellow,green,cyan,blue,magenta}` | `global_hue_deg` + `global_hue_magnitude` (NEW) | `global_hue_deg` rotates toward the named sector; `global_hue_magnitude` increases |
| `sat_{sector}_up` | `per_hue_saturation[sector]` (NEW) | increase in the named INPUT-hue sector |
| `sat_{sector}_down` | `per_hue_saturation[sector]` (NEW) | decrease in the named INPUT-hue sector |

Style bundles (`matte`, `faded`, `filmic`, `cinematic`, `teal-orange`, `sepia`,
`bleach bypass`, `natural`) are measured composites, not single-axis directions;
they are scored by the L7 style-recipe/discriminability gate against the
`calibration_manifest.json` windows (`matte` additionally has a first-class
`matte_strength` axis). See "Style Metrics".

Final eval minimum detectable movement:

- Temperature/tint: at least 1.5 Lab channel units.
- Exposure/shadows/highlights/black point: at least 2.0 L*.
- Saturation: at least 2.0 chroma.
- Contrast: at least 2.5 L* spread.

Rows whose target LUT does not meet the minimum movement for its gold tag should
not enter the final headline eval set.

## Target Fidelity

Target fidelity prevents a direction-only LUT from passing.

```text
target_fidelity_pass =
  image_mean_deltaE00_to_target <= 3.0
  and image_p95_deltaE00_to_target <= 8.0
  and chart_mean_deltaE00_to_target <= 3.0
  and chart_p95_deltaE00_to_target <= 8.0
```

Use the tokenizer-decoded target LUT as the scoring target for model output, and
store canonical-target reconstruction separately:

```text
canonical_to_decoded_mean_deltaE00
canonical_to_decoded_p95_deltaE00
```

Eval rows are headline-eligible only if target tokenization is acceptable:

```text
canonical_to_decoded_mean_deltaE00 <= 2.0
canonical_to_decoded_p95_deltaE00 <= 6.0
representability_tier == "gold"
headline_eligible == true
```

### Acceptance Modes for Underspecified Rows

Exact-target scoring assumes one correct LUT. Underspecified style prompts
("warm matte", "cinematic") admit many valid LUTs, so ANDing single decoded-target
`target_fidelity_pass` into `supported_pass` would fail correct outputs. Such rows
set `acceptance_mode` to `multi_reference`, `behavior_window`, or both; direction
(L4) and style-recipe/discriminability (L7) windows still apply on top.

```text
multi_reference_pass =
  exists r in reference set:
    image_mean_deltaE00_to_r <= 3.0
    and image_p95_deltaE00_to_r <= 8.0
    and chart_mean_deltaE00_to_r <= 3.0
    and chart_p95_deltaE00_to_r <= 8.0

behavior_window_pass =
  for every dimension d in behavior_window:
    behavior_window[d].min <= measured_behavior[d] <= behavior_window[d].max
```

The reference set is `reference_tokens` (K per row, a list of 64-token lists)
decoded through the frozen VQ decoder; `reference_lut_paths` holds the canonical
reference LUTs for provenance. References use the same DeltaE gate as exact_target,
and each must individually satisfy the tokenization-acceptability gate above to keep
the row headline-eligible. `behavior_window` keys are a subset of the measured
behavior vector, bounds are in LUT-domain measured-behavior units, and windows are
frozen from deterministic spec/config thresholds before final eval:

```text
behavior_window = {
  "temperature_delta_b":     {"min": 1.5, "max": 6.0},
  "contrast_l_spread_delta": {"min": -8.0, "max": -2.5}
}
```

Reference tokens and behavior windows are frozen at construction time and never
inferred from model output during scoring. Report `supported_prompt_to_lut_pass_rate`
separately for exact_target rows (e.g. `eval_expert_holdout`,
`eval_cross_source_expert`) and for multi-reference/behavior-window rows, so
expert-mimicry fidelity stays measurable and is not diluted by the looser windows.

## Safety Checks

Provisional gates:

| Safety Check | Threshold |
| --- | --- |
| Clip rate | <= 0.5% sampled channels clipped |
| Pre-clamp out-of-range | max violation <= 0.03 |
| Foldover/grid monotonicity | <= 0.1% severe grid-cell violations |
| Smoothness | p99 second-difference <= 0.06 |
| Neutral drift | DeltaE00 <= 3.0 unless prompt explicitly requests tint |
| Skin locus | fixed LUT-domain `skin_locus_v1` gate passes |

Skin-locus metrics:

```text
skin_locus_deltaE00_p95
skin_locus_hue_drift_deg_p95
skin_locus_luma_drift_abs_p95
skin_locus_chroma_ratio_min
skin_locus_chroma_ratio_max
skin_locus_clip_rate
skin_locus_lightness_order_violations
```

Provisional skin-locus gate:

```text
skin_locus_clip_rate == 0
skin_locus_hue_drift_deg_p95 <= 8
skin_locus_deltaE00_p95 <= 12
skin_locus_chroma_ratio_min >= 0.75
skin_locus_chroma_ratio_max <= 1.35
skin_locus_lightness_order_violations == 0
```

Safety failure rate is:

```text
safety_failures / supported_non_refusal_outputs
```

Report safety failures by type, not only as one aggregate number.

## Unsupported Metrics

| Metric | Formula |
| --- | --- |
| Unsupported recall | correct refusals on gold unsupported / all gold unsupported |
| Unsupported precision | correct refusals / all model refusals |
| False-support rate | LUT output on gold unsupported / all gold unsupported |
| Over-refusal rate | `<unsupported>` on gold supported / all gold supported |
| Supported coverage | non-refusal on gold supported / all gold supported |
| Selective risk | deterministic failures / supported non-refusals |
| Boundary accuracy | correct refusal plus correct non-refusal / all rows |
| Boundary F1 | F1 on supported-vs-unsupported decision |
| Mixed unsupported recall | correct refusals on mixed unsupported rows / all mixed unsupported rows |
| Near-boundary pair accuracy | correct decision on both rows in a boundary pair |

Unsupported recall alone is not enough. A model that refuses too often can look
safe but be useless.

## Style Metrics

Add these fields to every measured behavior vector where applicable:

```text
highlight_delta_a
highlight_delta_b
highlight_hue_delta_deg
highlight_chroma_delta
shadow_delta_a
shadow_hue_delta_deg
shadow_chroma_delta
split_tone_strength
split_tone_high_hue_quadrant
split_tone_shadow_hue_quadrant
style_multi_match_count
style_margin_to_nearest_neighbor
```

Report a style confusion matrix and pairwise overlap rates. Single-style
headline rows must pass the style recipe and style-discriminability gate.

## Eval Splits

Minimum final reporting:

```text
500 supported eval cases
100 unsupported eval cases
50-100 qualitative demo cases
```

Target:

```text
800 supported eval cases
200 unsupported eval cases
100 qualitative demo cases
```

The minimum is a reporting floor, not automatic evidence that every fine-grained
gate is statistically powered.

> **v1 provisional rebind.** The `800 / 200 / 100` targets are the aspirational
> full-supply sizes. The realized v1 eval reserve (~382 headline supported rows)
> cannot power the original +5pp headline claim (which needs ~1356 rows under
> Holm over the `sft_ship` family). For v1 the headline quality gates are rebound
> to a **+10pp claim at a uniform min_N = 350** floor. See "Pass Criteria" and
> `eval/configs/gating_slice_registry.yaml` (`frozen: false`) for the exact
> bindings; restore these targets when the direct-LUT supply lever lands.

Composition of the frozen eval budget:

The `800 / 200 / 100` target counts are the usage-weighted **headline** slice
(`eval_usage_weighted_headline` supported and unsupported, plus
`qualitative_demo`), not the whole frozen budget. The binding registry slices
below relate to the headline pool as follows.

- **Within** the headline pool (subsets, no extra rows; a headline row may carry
  more than one slice tag): `eval_subtle_control` (>= 150) and
  `eval_style_discriminability` (>= 150) are drawn inside the 800 supported
  headline rows; `eval_unsupported_mixed` (>= 100) is drawn inside the 200
  unsupported headline rows; `eval_coverage_macro` is a reporting view over the
  existing headline rows (macro coverage across source/style/attribute buckets)
  and carries no extra rows.
- **Additive** (contrastive, grouped, or robustness rows that are not
  usage-weighted and would over-subscribe the headline pool if drawn within):
  `eval_boundary_pairs` (>= 100 complete pairs = 100 supported + 100 unsupported
  additive rows), `eval_image_sensitivity` (>= 300 supported additive rows,
  grouped), and `eval_real_world_cli_inputs` (>= 100 supported additive rows,
  already reported separately from curated headline rows).

Reconciled frozen total = **1300 supported** (800 headline + 100
boundary + 300 image-sensitivity + 100 real-world) + **300 unsupported** (200
headline + 100 boundary) + **100 qualitative** = **1700 rows**. Within-pool draws
must never exceed the 800/200 headline counts, and any gated slice added later
must declare its within/additive status here at eval freeze.

Splits:

| Split | Purpose |
| --- | --- |
| `dev_calibration` | tune thresholds and catch harness bugs; never final |
| `eval_usage_weighted_headline` | headline supported eval weighted by rough expected usage |
| `eval_coverage_macro` | macro coverage across source/style/attribute buckets |
| `eval_image_sensitivity` | same-prompt/different-image rows where the correct safe LUT must differ across source images; drives the image-conditioning gate |
| `eval_real_world_cli_inputs` | real CLI-style inputs: phone JPEG/HEIC exports, screenshots, heavy JPEG compression, odd white balance, embedded/wide-gamut ICC, small/large images; reported separately from curated headline rows |
| `eval_subtle_control` | common low-magnitude but visible adjustments |
| `eval_style_discriminability` | single-style rows with neighbor-exclusion checks |
| `eval_expert_holdout` | held-out PPR10K/FiveK expert ids absent from SFT |
| `eval_cross_source_expert` | train mostly on filter/public/HaldCLUT, eval on expert-derived LUTs |
| `eval_unseen_family` | held-out LUT/style/source families absent from training |
| `eval_unsupported` | local, semantic, generative, geometry/detail, relighting, reference-transfer, selective-preservation prompts |
| `eval_unsupported_mixed` | supported global request plus unsupported component |
| `eval_boundary_pairs` | contrastive near-boundary supported/unsupported pairs |
| `eval_procedural_diagnostic` | procedural filler diagnostics; excluded from headline gates |
| `qualitative_demo` | hand-reviewed demos with before/after artifacts |

Only headline-eligible rows can drive headline pass claims and ship gates.

`eval_image_sensitivity` is accepted only as grouped evidence. Each group has an
`image_conditioning_group_id`, uses identical instruction text across at least
two source images, stores target-difference evidence showing that the correct
decoded safe LUTs differ by a predeclared behavior-vector or chart DeltaE
threshold, and fails construction if a single prompt-only/common LUT can pass
every row in the group on `dev_calibration`.

Leakage prevention:

- Split PPR10K by group id, not only image id.
- Split FiveK by source photo id.
- Split public LUTs by normalized LUT hash and source pack.
- Split prompts by template family and teacher generation batch.
- Keep near-duplicate images in the same split using perceptual hash and
  embedding nearest-neighbor checks.
- No same `derived_lut_id`, `source_pair_id`, `support_map_hash`, generic paired
  input image, or split unit id may cross train/eval.

## Eval-Honesty Slices And Interpreter Metrics (ADR 0024)

The two-stage design (ADR 0020) adds an Interpreter that maps user text to an
AttributeSpec plus a route, so the harness gains decoder-free honesty slices and
interpreter-level metrics on top of the L0-L8 LUT stack. These are contractual
under ADR 0024 (eval-honesty).

Honesty invariants:

- **Unit-aware holdout.** Held-out assignment keys on `split_unit_id`, never the
  row id, so near-duplicate LUT units cannot straddle the train/holdout boundary.
  Expect and accept a headline drop that quantifies prior per-row-split inflation.
- **Score all held-out rows.** Drop any default `--limit`; score every held-out
  row and report macro per-slice and per-family accuracy with group-bootstrap CIs.
- **Exact-64 assertion.** Every scored supported row must have exactly 64 surviving
  code positions (consistent with L1/L2; closes the partial-truncation blind spot).
- **Frozen decoder stays disabled** for these slices (`eval/lut_decoder.py`): they
  test the Interpreter and routing with decoder-free proxies, not the frozen VQ
  decode path.

Three-way route at the L0 boundary. The route decision is
`grade` / `clarify` / `refuse` (ADR 0023), extending the L0 boundary beyond the
2-way non-refusal-vs-`<unsupported>` split. `refuse` covers BOTH out-of-scope and
out-of-gamut requests. At L0: a gold `grade` row fails if it is clarified or
refused; a gold `refuse` row passes only on the correct refusal (`<unsupported>`);
`clarify` is scored as its own class. `grade` still carries the full L1-L8
supported stack; `refuse` uses the exact-`<unsupported>` rule.

New decoder-free slices:

| Slice | What it tests |
| --- | --- |
| `unseen-wording` | supported request phrased with wording absent from training |
| `named-concept` | a named style/look concept the Interpreter should map to known axes |
| `nonce-concept` | a made-up/nonce concept that must route to `clarify` or `refuse`, not hallucinate axes |
| `counterfactual-ranking` | the correct AttributeSpec must rank above counterfactual edits |
| `paraphrase-consistency` | paraphrases of one request must yield a consistent AttributeSpec/route |
| `refuse (out-of-scope + out-of-gamut)` | both refusal families must route to `refuse` |
| `in-distribution regression` | held-in supported distribution does not regress |

These slices are declared in `eval/configs/gating_slice_registry.yaml` alongside
the existing binding registry entries.

Interpreter metrics (reported next to the LUT-stack metrics):

```text
interpreter_attribute_f1   # requested-vs-measured axis F1 against the gold AttributeSpec
route_accuracy             # 3-way grade/clarify/refuse accuracy
over_refusal_rate          # gold grade routed to clarify/refuse / all gold grade
```

These interpreter-facing metrics complement but do not replace the L0-L8
supported/unsupported pass criteria; the frozen decoder remains disabled here.

## Baselines

Evaluate the same frozen rows against:

1. Always-unsupported null baseline.
2. Identity-all-prompts null baseline.
3. Oracle-boundary identity diagnostic baseline, excluded from fair headline
   comparisons.
4. Train-mean constant LUT.
5. Dev-optimized single constant LUT.
6. Dev-optimized style-blind constant LUT.
7. Token baseline: Qwen2.5-VL-3B with LUT vocabulary but no LUT SFT.
8. Prompted Qwen raw mode: prompted to emit `.cube` or LUT tokens.
9. Prompted Qwen recipe mode: prompted to emit compact JSON recipe, then
   deterministic renderer converts recipe to LUT.
10. Deterministic slider/recipe baseline: hand-written parser and renderer for
    supported attributes.
11. Prompt-only/image-blind SFT baseline.
12. Blank-image and shuffled-image ablations.
13. Frontier prompted baseline if available.
14. Generative-warmup checkpoint.
15. SFT checkpoint.
16. RS/DPO checkpoint if trained.
17. GRPO checkpoint if trained.

> **"Recipe mode" is a baseline term, distinct from the production AttributeSpec
> path (ADR 0020, 0021).** The "recipe mode" baselines (9 and 10) — a model or
> parser emitting a compact JSON *recipe* that a deterministic renderer converts to
> a LUT — are a BASELINE construct only. They are NOT the production two-stage path:
> there the Interpreter emits an **AttributeSpec** (serialized to
> `attribute_spec_text`), NOT a "recipe", which conditions the Generator, whose 64
> VQ codes go through the frozen decoder. Keep "recipe mode" reserved for these
> deterministic-renderer baselines so the baseline question (does fine-tuning beat
> prompt+renderer?) stays separate from the production interpreter path. Schema
> source of truth: `docs/attribute_spec.md`.

The project model uses only its native path:

```text
64 LUT tokens -> tokenizer decoder -> residual LUT -> full LUT
```

Prompted recipe baselines are allowed to use deterministic rendering because the
baseline question is whether fine-tuning is necessary for reliable behavior.

If SFT does not beat the best prompted frontier recipe/raw baseline by the
predeclared gate, the project must not claim that fine-tuning is required. It may
still claim local/offline/cost/artifact advantages if those are true.

### Deterministic Renderer Baseline Pinning

The deterministic renderer used by baselines 9 and 10 is a ship-gate reference,
so it is pinned with the same rigor as the model and the judge. Because the +5pp
headline gate (see Pass Criteria) is measured against it, an under-specified or
tunable renderer would make that gate gameable.

`baseline_adapters.py` deterministic-renderer mode is blocked until
`configs/renderer_baseline.yaml` pins:

- `renderer.version` and `renderer.code_sha256`: the frozen renderer + parser
  build. Aliases such as `latest` are not allowed.
- `renderer.canonical_domain_id` equal to the eval canonical LUT domain
  (`slm_lut_v1_srgb_display_encoded_17_trilinear`); a mismatch fails like any L2
  domain mismatch.
- `parser.supported_attributes`: the exact allowed input-attribute list, which
  must equal the behavior-spec supported-attribute taxonomy (Temperature, Tint,
  Exposure, Contrast, Black point, Highlights, Shadows, Saturation, Neutral
  safety, Global skin safety; see detailed_behavior_spec.md "Supported Prompt
  Space"). Any attribute outside this list must map to `<unsupported>`; the
  parser may not silently extend its scope to win rows.
- `parser.style_bundles`: the supported style words, equal to the behavior-spec
  "Supported Style Bundles" taxonomy.
- `dev_calibration_budget`: a bounded tuning budget for the baseline's
  thresholds and slider magnitudes, analogous to the dev-optimized constant
  baselines. Tuning is allowed only on `dev_calibration`, logs every trial in the
  config, and is frozen before final eval; no tuning on any `eval_*` split.

Once frozen, the renderer version, parser scope, style scope, and calibrated
thresholds are recorded in each `eval_runs/{run_id}/config.yaml` alongside
`configs/model_clients.yaml`, so every deterministic-renderer comparison is
reproducible.

## Statistics

Rate formula:

```text
rate(model, slice) = sum_i pass_i / N
single-rate CI = Wilson 95% CI
```

Paired delta formula:

```text
paired_delta(A, B, metric, slice) =
  mean_i(metric_i_A - metric_i_B)
  over the same frozen row ids

paired_delta_CI =
  stratified paired bootstrap over row ids, B >= 10,000
```

Use paired tests because base/SFT/RS/DPO/GRPO run on the same frozen rows. For
binary pass/fail metrics, report paired bootstrap CI plus McNemar or exact paired
permutation test. For continuous metrics like DeltaE, use paired bootstrap on
mean/median delta.

Seed protocol:

```text
SFT final reporting: run 3 seeds when making final claims.
RS/DPO/GRPO final reporting: run at least 3 seeds from a predeclared SFT checkpoint.
Smoke/dev runs may be single-seed but must be labeled exploratory.
Never choose the final model by final-eval performance; select on dev_calibration only.
Report every seed plus mean/std/min/median/max across seeds.
```

Small eval slice handling:

```text
50 supported / 20 unsupported smoke eval:
  pipeline sanity only; no pass/fail gate

attribute/category slices with N < 100:
  report raw count, rate, Wilson CI, and examples; no gate

unsupported categories:
  aggregate for the main refusal gate; category rows are diagnostic unless each category is sufficiently powered
```

Every ship-gated metric must declare a gating-slice registry entry before final
eval freeze:

```text
split
metric
min_N or min_paired_N
strata
MDE_pp
CI method
underpowered behavior
```

A gate with N below its declared minimum is not evaluable and cannot silently
pass or fail. `N < 100` never gates unless the metric is aggregated into a
predeclared sufficiently powered slice.

Multiplicity policy:

```text
ship_gate_family = all ship-gated SFT/RS-DPO/GRPO pass/fail tests evaluated for
  one ship decision on one frozen eval set
family_alpha = 0.05
method = Holm-Bonferroni over p-values for tests inside the family, plus
  simultaneous paired-bootstrap confidence bounds for reported gate deltas
OR groups = declared composite families; every member test in the OR group is
  included in the same multiplicity adjustment before the OR is evaluated
diagnostic metrics = not part of ship_gate_family and cannot ship a model
```

The report may include unadjusted exploratory intervals, but pass/fail decisions
must use the multiplicity-adjusted family.

Initial binding registry:

```text
eval_usage_weighted_headline: supported N >= 800; unsupported N >= 200   # v1: supported floor rebound to 350 (+10pp claim); see Pass Criteria
eval_unsupported_mixed: N >= 100
eval_boundary_pairs: >= 100 complete pairs
eval_image_sensitivity: N >= 300 rows, >= 100 same-prompt image groups, MDE +10pp vs prompt-only/image-blind
eval_subtle_control: N >= 150
eval_style_discriminability: N >= 150 single-style rows, >= 30/style for per-style gates
eval_real_world_cli_inputs: N >= 100, product/robustness report slice, diagnostic unless a target-quality gate is predeclared
```

The registry above is materialized as `eval/configs/gating_slice_registry.yaml`
with version key `gating_slice_registry_version` and is a Stage 9 output (see
master_plan.md). The `min_N`/`min_paired_N` values are the provisional bindings;
`strata`, `MDE_pp`, CI method, multiplicity family, and underpowered behavior are
declared per entry and frozen with the eval sets before final eval.

These N's are minimum evaluable sizes, not a partition of the headline budget.
`eval_subtle_control`, `eval_style_discriminability`, and `eval_unsupported_mixed`
are counted **within** the 800/200 headline pools; `eval_boundary_pairs`,
`eval_image_sensitivity`, and `eval_real_world_cli_inputs` are **additive**. See
"Eval Splits" for the reconciled frozen total (1300 supported / 300 unsupported /
100 qualitative).

## LLM/VLM Judge

The judge is required by the project spec but is not the primary authority for
color behavior.

Judge dimensions:

| Dimension | 0 | 1 | 2 |
| --- | --- | --- | --- |
| Spec adherence | violates output/support contract | partial | fully follows contract |
| Robustness | breaks under messy prompt | wobbles | stable |
| Task quality | wrong/useless | acceptable | clean controlled grade |
| Consistency | inconsistent across similar inputs | mostly stable | reliable |

Judge prompt must include the compact behavior spec, the source instruction, the
model output, deterministic metrics, and before/after thumbnails when available.
The judge may explain likely causes of failures, but cannot convert a
deterministic fail into a pass.

`judge_client.py` is blocked until `configs/model_clients.yaml` pins
`judge_primary.provider`, `judge_primary.model_id`, endpoint/base-url env var,
API-key env var, `judge_primary.prompt_version`, and `judge_primary.batch_id`.
Model aliases such as `latest` are not allowed, and secrets are referenced only
by env var name.

## Reports

Required outputs:

```text
eval_runs/{run_id}/
  config.yaml
  rows.jsonl
  raw_model_outputs.jsonl
  parsed_outputs.jsonl
  metrics_by_row.parquet
  overall_results.csv
  attribute_results.csv
  target_fidelity_results.csv
  style_results.csv
  safety_results.csv
  unsupported_results.csv
  baseline_delta.csv
  seed_summary.csv
  failure_manifest.jsonl
  qualitative/
    row_id_input.png
    row_id_graded.png
    row_id_side_by_side.png
```

Required tables:

| Table | Required Columns |
| --- | --- |
| Overall | model, checkpoint_id, seed, mode, split, N, pass_n, pass_rate, pass_ci_low, pass_ci_high, valid-token rate, decode-valid rate, target-fidelity pass, safety fail, judge means |
| Baseline delta | model pair, seed policy, metric, N_paired, delta_pp, paired_boot_ci_low_pp, paired_boot_ci_high_pp, paired_test_p, gate threshold, gate result |
| Seed summary | model_stage, seed_count, metric, mean, std, min, median, max, seed_mean_ci_low, seed_mean_ci_high |
| Attribute | model, attribute, N, direction pass, mean measured delta |
| Target fidelity | model, split, acceptance_mode, image mean/p95 DeltaE00, chart mean/p95 DeltaE00, reference match count, behavior-window pass, pass |
| Style | model, style, style pass, multi-match count, margin, confusion |
| Safety | model, clip fail, smoothness fail, foldover fail, neutral drift fail, skin-locus fail |
| Unsupported | model, category, recall, precision, false-support, over-refusal, coverage, boundary F1, mixed recall |
| Error analysis | failure layer, count, representative row ids, likely data/eval/model cause |

## Pass Criteria

SFT passes only if:

```text
free_generation_valid_token_rate Wilson 95% lower bound >= 85%
unsupported_recall Wilson 95% lower bound >= 80%
unsupported_precision Wilson 95% lower bound >= 80%
boundary_f1 Wilson 95% lower bound >= 80%
mixed_unsupported_recall Wilson 95% lower bound >= 80%
near_boundary_pair_accuracy Wilson 95% lower bound >= 85%
over_refusal_rate Wilson 95% upper bound <= 10%
safety_failure_rate Wilson 95% upper bound <= 5%   # v1 absolute safety ship-gate (min_N 350; certifies true <=1% unsafe)
supported_prompt_to_lut_pass_rate Wilson 95% lower bound >= 60%
paired-bootstrap 95% lower bound for supported_prompt_to_lut_pass_rate vs best null >= +30pp
paired-bootstrap 95% lower bound for supported_prompt_to_lut_pass_rate vs best constant >= +20pp
paired-bootstrap 95% lower bound for supported_prompt_to_lut_pass_rate vs deterministic renderer on eval_usage_weighted_headline >= +10pp   # v1 rebind (was +5pp)
paired-bootstrap 95% lower bound for supported_prompt_to_lut_pass_rate vs deterministic renderer on eval_subtle_control >= 0pp
paired-bootstrap 95% lower bound for supported_prompt_to_lut_pass_rate vs deterministic renderer on eval_style_discriminability >= 0pp
paired-bootstrap 95% lower bound vs deterministic renderer on at least one of eval_subtle_control or eval_style_discriminability >= +5pp
paired-bootstrap 95% lower bound for supported_prompt_to_lut_pass_rate vs prompt-only/image-blind SFT baseline on eval_image_sensitivity >= +10pp (provisional; calibratable)
paired-bootstrap 95% lower bound for supported_prompt_to_lut_pass_rate vs blank-image ablation on eval_image_sensitivity > 0
paired-bootstrap 95% lower bound for supported_prompt_to_lut_pass_rate vs shuffled-image ablation on eval_image_sensitivity > 0
over_refusal_rate <= deterministic_renderer_over_refusal + 2pp        # v1: DIAGNOSTIC only (needs ~2400-7600 rows; not ship-gated)
safety_failure_rate <= deterministic_renderer_safety_failure + 2pp    # v1: DIAGNOSTIC only (see v1 note; prefer an absolute bound)
```

> **v1 provisional rebind** (headline supported slice; `gating_slice_registry.yaml`, `frozen: false`).
> Thresholds above are the full/aspirational bar. For v1, because the eval reserve is ~382 rows:
> - Headline quality gates (`supported_prompt_to_lut_pass_rate`, `free_generation_valid_token_rate`,
>   `vs_best_null`, `vs_best_constant`, `vs_deterministic_renderer`) are rebound to a **+10pp claim**
>   at a uniform **min_N = 350** floor. Absolute thresholds (>= 60%, >= 85%) are unchanged; the
>   "+10pp" is a power-design statement (the model must truly sit ~10pp over the bar to certify with
>   350 rows). The renderer comparison **threshold** moves +5pp -> +10pp.
> - `over_refusal_rate` is coarsened (MDE 5 -> 6, min_N 350): a true <= 4% over-refusal clears the
>   10% ceiling within the slice.
> - The two paired `*_vs_deterministic_renderer` @ +/-2pp guardrails are **demoted to diagnostic**
>   (they need ~2400-7600 rows). Safety instead stays ship-gated by an **absolute
>   `safety_failure_rate` Wilson upper bound <= 5%** (min_N 350) — cheap because it is not
>   differenced against the renderer. It needs 303 rows to certify a true <= 1% unsafe rate at 80%
>   power, so a passing model is genuinely ~<= 1% unsafe even though the nominal bar is 5%.
> - `free_generation_valid_token_rate` was min_N 1000 -> 350 (own +10pp requirement ~143).
>
> Restore the +5pp / min_N ~1356 bindings and the paired renderer guardrails when supply grows.

These SFT criteria are self-contained: the prompt-only/image-blind SFT baseline and the blank-image and shuffled-image ablation runs on `eval_image_sensitivity` are trained and scored as part of the SFT evaluation gate, so the gate is computed before it is evaluated and does not depend on any later baselines/reporting stage.

If the image-conditioning gate fails while the non-image-dependent gates pass,
the v1 claim narrows to prompt-to-LUT reliability on the supported synthetic
prompt distribution. Do not claim that the VLM uses image evidence unless the
`eval_image_sensitivity` gate passes.

In every gate above, `deterministic renderer` and the `deterministic_renderer_*`
quantities refer to the single frozen, config-pinned baseline defined under
"Deterministic Renderer Baseline Pinning" (`configs/renderer_baseline.yaml`). The
+5pp headline gate on `eval_usage_weighted_headline` is measured only against that
reproducible baseline; a re-tuned or re-scoped renderer invalidates the
comparison.

If a prompted frontier baseline is run:

```text
SFT must beat the best prompted frontier recipe/raw baseline by >= 5pp outside paired CI
to claim fine-tuning is required for behavior reliability.
```

Runtime constrained eval must report `syntax_valid_rate == 100%`; any syntax
failure in constrained mode is an implementation bug and blocks release.

GRPO ships only if:

```text
ships = (A) AND (B)

(A) improvement group — at least one holds:
      paired-bootstrap 95% lower bound for pass_rate(GRPO - best prior tuned stage) >= +5pp
      OR paired-bootstrap 95% upper bound for safety_failure_rate(GRPO - best prior tuned stage) <= -5pp

(B) guardrail group — all hold:
      paired-bootstrap 95% upper bound for over_refusal_rate(GRPO - best prior tuned stage) <= +2pp
      AND over_refusal_rate still satisfies the absolute SFT ceiling
      AND mixed_unsupported_recall does not drop by more than 2pp
      AND near_boundary_pair_accuracy does not drop by more than 2pp
      AND the multi-seed summary shows the effect is not from one lucky seed
```

If the point estimate clears a threshold but the CI does not, label it:

```text
directional improvement; statistically inconclusive; do not ship over the prior stage on this evidence
```

The GRPO ship gate above presumes GRPO was run at all. GRPO is run only when the
best prior tuned stage has plateaued and reward correctness is proven; both
preconditions are defined operationally in `docs/training_plan_colab.md` Stage 15
(conditional GRPO under ADR 0025 training sequence v2).
