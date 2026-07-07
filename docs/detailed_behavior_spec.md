# Detailed Behavior Spec

## Status

This document expands the compact behavior contract in `docs/behavior_spec.md`.
The compact spec remains the shortest pass/fail definition. This document adds
the operational rules needed to build data, training, evaluation, and the first
CLI demo.

The selected direction is Track B with caveats:

- Build the real prompt-to-LUT architecture, not only a toy course slice.
- Keep the first product surface CLI-first; the child-facing workbench comes
  later.
- Collect the broad source corpus, then cull to a smaller, diverse active
  dataset.
- Treat data rights as non-blocking for the personal project, while still
  recording provenance and source metadata.
- Use Google Colab as the primary training environment, with A100/L4 preferred
  and T4 treated as smoke-test only.

## Compact Behavior Contract

Given a source image and a supported global color-grading instruction, the model
must output exactly 64 valid LUT code tokens enclosed by `<lut_bos>` and
`<lut_eos>`. Those tokens must decode into one canonical 17x17x17 residual
global LUT which, after adding identity and applying to the image, moves every
explicit gold prompt attribute in the correct measured direction, matches the
target grade within the target-fidelity gate, and passes LUT safety gates.

Given an unsupported instruction requiring local, semantic, generative,
geometry/detail, relighting, reference-transfer, or impossible
selective-preservation behavior, the model must output exactly `<unsupported>`.

A prompt is supported only if every required edit can be represented by one
global LUT. If any required component is unsupported, the entire prompt is
unsupported in v1. The model must not satisfy only the supported subset of a
mixed prompt.

## Inputs

The model receives:

```text
source_image: RGB image
instruction: natural-language prompt from the user
```

The model does not receive:

```text
target graded image
target LUT
gold tags
structured recipe
explanation text
```

Gold tags, target images, target LUTs, measured behavior vectors, and
target-fidelity measurements are used for dataset construction, filtering,
reward computation, and evaluation.

## Outputs

The model output must be exactly one of the following forms:

```text
<lut_bos> <lut_000> ... 64 total LUT code tokens ... <lut_eos>
```

or:

```text
<unsupported>
```

Rules:

- The 64 code tokens exclude `<lut_bos>` and `<lut_eos>`.
- LUT code tokens must be in the closed range `<lut_000>` through `<lut_255>`.
- Dataset `target_tokens` for supported rows contain exactly 64 integer code
  ids in `0..255` and exclude `<lut_bos>` / `<lut_eos>`. Unsupported rows use an
  empty `target_tokens` list and `assistant_target = "<unsupported>"`.
- No prose, JSON, markdown, raw `.cube` values, XML, recipe text, apology, or
  explanation is allowed in the model output.
- Identity LUT is not a refusal. It is valid only when the instruction is a
  supported "leave unchanged" or near-identity prompt with matching gold tags.
- Any output that mixes `<unsupported>` with LUT tokens fails.
- Any output with fewer or more than 64 LUT code tokens fails.

The CLI and later workbench may produce explanatory artifacts from deterministic
metrics. The model itself is not responsible for explanation in v1.

## Supported Prompt Space

V1 supports global color-grading instructions that one global LUT can represent.
A single global LUT maps the same input RGB value to the same output RGB value
everywhere in the image. Therefore supported behavior is limited to global color
and tone changes.

| Prompt Attribute | Supported Examples | Measured Behavior |
| --- | --- | --- |
| Temperature | "make it warmer", "cool it down" | Lab b* direction on sampled pixels/chart |
| Tint | "more magenta", "less green" | Lab a* direction |
| Exposure | "brighter", "darker" | mean L* direction |
| Contrast | "more punch", "softer contrast" | L* p95-p5 spread direction |
| Black point | "lift the blacks", "crush the blacks" | low-percentile L* direction |
| Highlights | "soften highlights", "brighter highlights" | high-luminance L*, a*, b*, hue, chroma, and clipping gates |
| Shadows | "lift shadows", "cooler shadows" | low-luminance L*, a*, b*, hue, and chroma direction |
| Saturation | "more saturated", "muted", "desaturated" | chroma direction |
| Neutral safety | "keep neutrals clean" | neutral-axis drift gate |
| Global skin safety | "keep skin natural" | intrinsic skin-locus LUT-domain gate |

Important boundary: "keep skin natural" is a global safety constraint, not a
promise that the model can isolate skin semantically while changing everything
else. A prompt such as "make the background blue but keep the face unchanged" is
unsupported.

## Supported Style Bundles

Style words are supported only when decomposed into measurable color behavior.
They are not aesthetic catchalls. Provisional windows are calibrated on
`dev_human_calibration`, then frozen before final eval.

| Style | Provisional Recipe Window |
| --- | --- |
| matte | black point L* +2..+8, contrast spread -1..-7, chroma -1..-5, no strong split tone |
| faded | black point L* +2..+10, chroma -4..-12, highlight L* -1..-7, contrast -2..-10 |
| filmic | highlight L* -2..-8, chroma -1..-6, contrast -2..+4, weak/no split tone |
| cinematic | teal/cyan shadows plus orange/yellow highlights, split strength 2..8, chroma -1..-6 |
| teal-orange | teal/cyan shadows plus orange/yellow highlights, split strength 5..14, neutral DeltaE00 <= 2 |
| sepia | global b* +2..+12, global a* +0.5..+6, chroma -1..-8, no teal shadows |
| bleach bypass | chroma -5..-16, contrast +3..+15, black point L* -1..-8 |
| natural | small deltas only, neutral drift improved or <= 1.5, must not match another style |

For single-style eval rows:

```text
style_discriminability_pass =
  style_recipe_pass
  and style_multi_match_count == 1
  and style_margin_to_nearest_neighbor >= calibrated_margin
```

Composite prompts such as "warm faded film look" are labeled as composite rows
and excluded from single-style discriminability headlines unless a primary style
is declared.

## Ambiguous Child-Language Policy

Natural language is valuable because children may describe intent before they
know editing terms. Ambiguous terms are handled by mapping them to global color
behavior when possible and refusing them when they require unsupported behavior.

| Prompt | Supported Interpretation | Refusal Case |
| --- | --- | --- |
| "make it pop" | higher contrast and/or saturation | if user asks one object to pop while background stays unchanged |
| "make it sharper" | slightly higher contrast or clarity-like global tone | if prompt asks for texture/detail sharpening |
| "make it moody" | darker exposure, lower saturation, cooler shadows | if prompt asks to add fog, rain, lights, or scene content |
| "make the sky prettier" | unsupported by default because it is local/semantic | supported only if rephrased globally, e.g. cooler overall |
| "make her shirt red" | none | semantic object recoloring, unsupported |

Global tone-range requests such as "cool the shadows" are supported. Semantic or
region-specific requests such as "make only the sky bluer" are unsupported
unless rewritten as a global color/tone instruction.

The CLI should return `<unsupported>` for unsupported prompts. The later
workbench should show a visible boundary message and suggest a global rewrite.

## Unsupported Prompt Space

The model must refuse prompts that require:

- Local region edits.
- Semantic object recoloring.
- Subject-only or background-only edits.
- Inpainting, removal, replacement, or new image content.
- Relighting, shadows cast by objects, or changed light direction.
- Geometry changes, camera changes, crop, pose, or perspective changes.
- Texture/detail edits such as denoise, sharpen, deblur, skin smoothing, hair
  cleanup, or object cleanup.
- Reference-image style transfer.
- Multiple region-specific looks.
- Selective preservation that one global LUT cannot represent.
- Mixed prompts that combine a supported global request with any unsupported
  component.

Examples that must produce `<unsupported>`:

```text
make only the sky bluer
change the shirt to red
remove the person in the background
make the face brighter but leave everything else dark
copy the colors from this reference image
make it look like sunset light is coming from the left
sharpen the details in the hair
blur the background
make it warmer and remove the background
give it a cinematic look and make the shirt red
```

## LUT Representation Contract

The model predicts tokens, not raw LUT floats.

```text
source image + instruction
        ->
Qwen2.5-VL-3B-Instruct with LUT vocabulary
        ->
<lut_bos> 64 LUT code tokens <lut_eos>
        ->
VQ tokenizer decoder
        ->
canonical 17x17x17 residual LUT
        ->
identity LUT + residual LUT
        ->
canonical absolute 17x17x17 global LUT
        ->
.cube export and graded image
```

## Canonical LUT Domain

Canonical LUT domain for v1 is display-referred IEC 61966-2-1 sRGB,
transfer-encoded R'G'B' values in `[0,1]`, D65 white, 17x17x17 grid, full-range
RGB, and trilinear interpolation. The canonical LUT maps encoded sRGB input
values to encoded sRGB output values. Residual LUT equals canonical absolute LUT
minus the canonical encoded-sRGB identity grid at the same nodes.

Tokenizer inputs, decoded runtime LUTs, exported `.cube` files, behavior
vectors, reconstruction metrics, and eval target LUTs all use this canonical
domain.

Raw source pipelines may be ProPhoto, RAW/ACR, Display P3, HaldCLUT-native, or
unknown, but accepted derived LUT artifacts must be converted into the canonical
domain before hashing, residual conversion, tokenizer encoding, prompt tagging,
export, or evaluation. Eval rows do not declare alternate active LUT domains in
v1.

Requirements:

- Grid size: 17x17x17.
- Representation during tokenizer training: canonical residual LUT.
- Runtime export: full canonical absolute LUT after identity addition.
- `.cube` export uses `LUT_3D_SIZE 17`, `DOMAIN_MIN 0 0 0`,
  `DOMAIN_MAX 1 1 1`, RGB table order with R changing fastest, fixed
  10-decimal float formatting, LF line endings, UTF-8, and no timestamps.
- Interpolation: trilinear only for v1; changing interpolation requires a new
  canonical domain/tokenizer version.
- Axis order, `.cube` table order, latent flatten order, and token suffix to
  codebook-index mapping are pinned in the tokenizer manifest.
- Output values may be clamped for rendering, but pre-clamp out-of-range
  magnitude is still evaluated.

## Color Pipeline

All deterministic evaluation uses the canonical LUT domain and a fixed
display-referred color pipeline:

```text
input image -> ICC-aware conversion to sRGB [0,1] -> linear RGB where needed -> CIE Lab D65
```

Metrics use sampled image pixels plus fixed color charts sampled through the
LUT. CIEDE2000 is used for reconstruction, target fidelity, and reporting.
Wide-gamut inputs such as Display P3, AdobeRGB, and ProPhoto are converted with
the pinned color-management module, relative-colorimetric intent, black-point
compensation enabled, deterministic gamut clipping to `[0,1]`, and float32
working precision. Preview artifacts are written as canonical sRGB PNG unless a
manifested export profile is explicitly requested.

Contrast spread uses one canonical key and formula:

```text
contrast_l_spread_delta =
  (p95(L*_out) - p5(L*_out)) - (p95(L*_in) - p5(L*_in))
```

The RAW and ProPhoto details of PPR10K/FiveK derivation must be recorded in the
data pipeline, but active instruction and eval artifacts are canonical sRGB LUTs.

## Safety Gates

These thresholds are starting gates, not immutable science. They should be
calibrated on the dev split before final reporting, then frozen for the final
eval.

| Gate | Provisional Threshold |
| --- | --- |
| Tokenizer reconstruction | mean DeltaE00 <= 2.0 on held-out LUTs, plus tail/per-family gates |
| Tokenizer reconstruction | PSNR >= 35 dB on LUT grid or rendered chart |
| Target fidelity | image/chart mean and p95 DeltaE00 to target within eval gate |
| Clip rate | <= 0.5% of sampled output channels at 0 or 1 due to clipping |
| Pre-clamp out-of-range | max violation <= 0.03 |
| Foldover/grid monotonicity | <= 0.1% severe grid-cell violations |
| Smoothness | p99 second-difference <= 0.06 |
| Neutral drift | neutral-axis DeltaE00 <= 3.0 unless explicitly tinted |
| Direction magnitude | >= 1.5 Lab units for tint/temperature tags in final eval |
| Exposure/tonal magnitude | >= 2.0 L* for exposure, shadows, highlights, black point in final eval |
| Saturation magnitude | >= 2.0 chroma for saturation tags in final eval |
| Contrast magnitude | >= 2.5 L* spread change for contrast tags in final eval |

## Skin-Locus Safety

Skin preservation is evaluated intrinsically in the LUT domain. Every decoded LUT
is sampled on a fixed `skin_locus_v1` chart, independent of whether the eval
image contains people.

Starter canonical sRGB8 anchors:

```text
cc_dark_skin:   115,  82,  68
cc_light_skin:  194, 150, 130
deep_anchor:     74,  48,  38
medium_anchor:  144,  98,  75
tan_anchor:     173, 123,  96
fair_anchor:    231, 195, 170
```

Metrics:

```text
skin_locus_deltaE00_p95
skin_locus_hue_drift_deg_p95
skin_locus_luma_drift_abs_p95
skin_locus_chroma_ratio_min
skin_locus_chroma_ratio_max
skin_locus_clip_rate
skin_locus_lightness_order_violations
```

Provisional gate:

```text
skin_locus_clip_rate == 0
skin_locus_hue_drift_deg_p95 <= 8
skin_locus_deltaE00_p95 <= 12
skin_locus_chroma_ratio_min >= 0.75
skin_locus_chroma_ratio_max <= 1.35
skin_locus_lightness_order_violations == 0
```

For exposure/contrast prompts, L* movement is allowed but excess skin-locus
movement relative to the midtone chart remains capped. Face/skin masks and
manual review are qualitative diagnostics, not the deterministic safety gate.

## Human Calibration

Before final eval freeze, build a `dev_human_calibration` set with 30-50
candidates per style, 15-20 hard negatives per style, identity/constant
baselines, and balanced people/non-people plus source families.

Blind raters see source plus graded image and provide:

- forced-choice style label or `none/unclear`;
- attribute-direction checks;
- magnitude rating;
- skin-naturalness acceptability.

Metric windows are calibrated against those labels, then frozen. Provisional
calibration gates are style precision/recall >= 70%, nearest-neighbor style
confusion <= 20%, attribute-direction agreement >= 80%, and unacceptable skin
rate <= 5%.

## CLI-First Behavior

The first demo is a CLI:

```text
prompt_to_lut --image input.jpg --prompt "give it a warm faded film look" --out outputs/run_001
```

Required output artifacts:

```text
outputs/run_001/
  input.png
  graded.png
  preview_side_by_side.png
  output.cube
  output_tokens.txt
  metrics.json
  version_manifest.json
```

For supported prompts, `metrics.json` reports measured deltas and safety metrics.
It does not invent gold tags. Direction checks run only when expected attributes
are supplied by an eval row or explicit CLI option.

Minimum `metrics.json` schema:

```json
{
  "schema_version": "1.0",
  "run_id": "...",
  "input": {
    "image_path": "...",
    "image_sha256": "...",
    "prompt": "..."
  },
  "output": {
    "kind": "lut_tokens | unsupported | invalid",
    "syntax_pass": true,
    "token_count": 64,
    "token_ids": [],
    "parser_errors": []
  },
  "decoding": {
    "mode": "runtime_constrained | free_generation",
    "grammar_mask": true,
    "fsm_version": "1.0",
    "do_sample": false,
    "num_beams": 1,
    "seed": 1234,
    "max_new_tokens": 67,
    "precision": "merged_fp16 | qlora_4bit_bf16"
  },
  "lut": {
    "canonical_domain_id": "slm_lut_v1_srgb_display_encoded_17_trilinear",
    "grid_size": [17, 17, 17],
    "latent_shape": [4, 4, 4],
    "codebook_size": 256,
    "flatten_order": "pinned_in_manifest",
    "interpolation": "trilinear",
    "vq_codebook_sha256": "...",
    "vq_decoder_sha256": "...",
    "cube_serialization_version": "cube_v1_size17_domain01_rgb_rfast_f10_lf",
    "icc_conversion_config": "srgb_relcol_bpc_float32_v1"
  },
  "measured_behavior": {
    "temperature_delta_b": null,
    "tint_delta_a": null,
    "mean_l_delta": null,
    "contrast_l_spread_delta": null,
    "highlight_l_delta": null,
    "shadow_l_delta": null,
    "chroma_delta": null,
    "neutral_drift_deltaE": null,
    "skin_locus_deltaE00_p95": null,
    "clip_rate": null,
    "smoothness": null,
    "foldover_rate": null
  },
  "direction_checks": {
    "expected_attributes_source": "none | eval_row | cli_supplied",
    "checks": []
  },
  "status": {
    "blocked": false,
    "block_reason": null
  },
  "version_manifest_sha256": "..."
}
```

For unsupported prompts, the CLI writes `output_tokens.txt` containing
`<unsupported>` and `metrics.json` with `output.kind = "unsupported"`. It should
not apply an identity LUT silently.

## Workbench-Later Behavior

The later child-facing workbench inherits the same model behavior but adds:

- Preview.
- Compare original vs version A/B.
- Undo.
- Revise.
- Name the look.
- Visible refusal and suggested global rewrite.
- Descriptive feedback generated from metrics.

The workbench must not rank children, praise traits, infer emotions, profile
style identity, or claim general creativity improvement. The defensible learning
claim is narrow: the tool may support closer noticing, better vocabulary,
prediction, explanation, and revision inside global color editing.

## Success Criteria

The model succeeds when final, CI-gated eval shows:

- tokenizer held-out reconstruction passes mean, tail, per-family, and
  per-target reconstruction gates;
- runtime constrained decoding reaches 100% syntax validity and free generation
  measures learned validity separately;
- SFT clears boundary gates, including unsupported recall/precision, boundary
  F1, mixed-prompt recall, and over-refusal ceiling;
- SFT beats required null, constant, and deterministic-renderer baselines;
- supported rows pass direction, target fidelity, safety, style, and skin-locus
  gates;
- GRPO, if shipped, beats the best prior tuned stage outside paired confidence
  intervals without increasing over-refusal beyond the allowed ceiling.

The central claim is behavior reliability, not general image-editing ability and
not child learning efficacy.

## Failure Taxonomy

| Failure | Meaning |
| --- | --- |
| Boundary false support | Model outputs LUT tokens for unsupported prompt |
| Mixed-prompt partial support | Model outputs a LUT for a prompt containing any unsupported component |
| Over-refusal | Model outputs `<unsupported>` for supported prompt |
| Syntax failure | Wrong token count, invalid token, missing BOS/EOS, extra text |
| Decode failure | Token sequence cannot decode to finite LUT |
| Direction failure | Explicit prompt tag moves in wrong direction |
| Target mismatch | LUT is safe and directional but fails target-fidelity gate |
| Safety failure | LUT clips, folds, drifts neutrals, distorts skin locus, or is too rough |
| Style failure | Style word recipe tags or discriminability checks fail |
| Explanation/UI failure | Later workbench describes the change inaccurately |
