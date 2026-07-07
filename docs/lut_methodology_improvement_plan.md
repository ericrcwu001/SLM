# Methodology Plan for V1 Prompt-to-LUT Image Grading

## 1. Executive Summary

This plan defines the v1 methodology for an instruction-guided prompt-to-LUT system. Given a source image and a global color-grading instruction, the model predicts a compact sequence of LUT tokens. A learned tokenizer decoder converts those tokens into a full 17x17x17 global 3D LUT, which is then applied to the image.

The v1 behavior target is intentionally narrow:

```text
source image + global color-grading prompt
        ->
64 LUT code tokens
        ->
17x17x17 residual global LUT
        ->
full .cube LUT and graded image
```

The model must output `<unsupported>` when the prompt asks for an edit that one global LUT cannot represent, such as local, object-specific, generative, geometry/detail, or impossible selective-preservation edits.

This replaces the earlier broad research roadmap with a project-shaped plan:

- Instruction-guided grading only.
- One global LUT in v1.
- Local LUT mixtures and reference-style transfer deferred.
- Clean implementation from scratch, with AceTone used as an architectural reference only.
- Real/expert LUT corpus first.
- VQ tokenizer over residual 17x17x17 LUTs.
- Qwen2.5-VL-3B-Instruct as the primary VLM.
- Supervised fine-tuning first, then a small GRPO stage.
- Prompt-to-LUT pass rate as the primary metric.

The detailed falsifiable behavior spec lives in [behavior_spec.md](./behavior_spec.md).

## 2. V1 Scope

### 2.1 In Scope

V1 supports image-conditioned instruction-guided global color grading:

```text
input:
  source image
  natural-language global grading instruction

output:
  exactly 64 valid LUT code tokens
  or <unsupported>
```

Supported prompt attributes include:

```text
temperature: warmer, cooler
tint: more magenta, more green
exposure: brighter, darker
contrast: higher contrast, softer contrast
black point: lifted blacks, crushed blacks
highlights: softer highlights, brighter highlights
shadows: lifted shadows, darker shadows, cooler shadows, warmer shadows
saturation: more saturated, muted, desaturated
style bundles: matte, faded, filmic, cinematic, teal-orange, sepia, bleach bypass, natural
safety constraints: preserve neutrals, avoid clipping, keep skin natural
```

Style bundles must be defined as measurable combinations of supported attributes. For example, `cinematic` should not be an unbounded aesthetic label; it should map to a consistent bundle such as cooler shadows, warmer highlights, mild contrast, restrained saturation, and highlight rolloff.

### 2.2 Out of Scope for V1

V1 does not support:

```text
local region edits
semantic object recoloring
background-only or subject-only changes
content generation
relighting
texture/detail edits
geometry changes
reference-image style transfer
K-LUT local mixtures
learned gating maps
```

These are extension paths. They require masks, segmentation, multiple LUTs, or spatial gating, which are separate architecture and evaluation problems.

## 3. Core Architecture

### 3.1 Overall Pipeline

The v1 system uses:

```text
source image + instruction
        ->
Qwen2.5-VL-3B-Instruct
        ->
<lut_bos> 64 LUT code tokens <lut_eos>
        ->
VQ tokenizer decoder
        ->
17x17x17 residual LUT
        ->
identity LUT + residual LUT
        ->
full global .cube LUT
```

The VLM does not emit raw RGB values. A 17x17x17 LUT contains 4,913 grid points and 14,739 channel values, which is brittle as direct language output. The model instead emits a fixed-length discrete token sequence.

### 3.2 LUT Tokenizer

The tokenizer is a VQ-style autoencoder trained on residual LUTs:

```text
absolute source LUT
        ->
resample to 17x17x17
        ->
residual = source LUT - identity LUT
        ->
VQ encoder
        ->
4x4x4 latent grid
        ->
64 codebook tokens
        ->
VQ decoder
        ->
reconstructed residual LUT
```

Tokenizer decisions:

```text
LUT grid: 17x17x17
representation: residual LUT
token count: 64
codebook size: 256
deployed output: full absolute LUT after adding identity
```

Residual LUTs are used because most photographic grades are controlled changes around the identity transform. This makes reconstruction easier, provides a safer default, and reduces the risk of catastrophic decoded mappings.

### 3.3 VLM Training Scope

The primary model is Qwen2.5-VL-3B-Instruct.

Initial training scope:

```text
freeze vision encoder
apply LoRA to the language model
train or LoRA the multimodal projector/connector
add LUT vocabulary tokens
train output behavior for LUT token generation
```

Full language-model fine-tuning or full-model fine-tuning is reserved for later scale-up after the tokenizer, SFT, evaluation, and small GRPO loop show proof of concept.

### 3.4 LUT Vocabulary

Add dedicated special tokens:

```text
<lut_bos>
<lut_eos>
<unsupported>
<lut_000>
<lut_001>
...
<lut_255>
```

The model should emit either:

```text
<lut_bos> <lut_###> ... 64 total code tokens ... <lut_eos>
```

or:

```text
<unsupported>
```

The unsupported output is explicit. Identity LUT is not used as a refusal, because it is ambiguous between "do nothing" and "cannot satisfy this prompt."

## 4. LUT Corpus

The LUT corpus is real/expert-first rather than synthetic-first. Synthetic or procedural LUTs may still be useful for diagnostics, edge cases, or controlled eval, but they are not the primary source.

### 4.1 Source Priority

Target source order:

```text
1. PPR10K-derived expert LUTs
2. FiveK-derived expert LUTs
3. Fresh LUTs CC0 LUTs
4. G'MIC / RawTherapee HaldCLUT collections
5. Smaller public LUT packs
```

Explicitly excluded from v1:

```text
DPED
HDR+ / ISP pipeline transforms
general camera-pipeline datasets
```

Those sources are more about camera rendering or ISP reconstruction than photographic color grading.

### 4.2 PPR10K-Derived Expert LUTs

PPR10K provides portrait retouching targets from three experts. The target count is approximately:

```text
11,161 images x 3 experts = 33,483 expert targets
```

Extraction strategy:

```text
expert XMP target
        ->
apply edit to identity Hald/grid image
        ->
read transformed grid as LUT
        ->
resample to 17x17x17
        ->
convert to residual LUT
```

Pair fitting can be used as a validation or fallback path when preset-to-grid rendering is incomplete.

### 4.3 FiveK-Derived Expert LUTs

MIT-Adobe FiveK contains source photos and expert-retouched outputs. Since the expert edits are naturally represented as before/after image pairs, FiveK-derived LUTs should be created through pair fitting:

```text
source image + expert target image
        ->
optimize a global 17x17x17 LUT
        ->
apply fitted LUT to source
        ->
measure fit quality against target
```

FiveK can contribute up to:

```text
5,000 images x 5 experts = 25,000 expert targets
```

Only fitted LUTs with acceptable quality scores should enter the tokenizer or instruction corpus.

### 4.4 Public LUT Sources

Fresh LUTs contributes roughly 736 visible LUT records and appears useful as a CC0 source. The downloader should use an authenticated account/session and avoid guessing hidden file URLs.

G'MIC and RawTherapee HaldCLUT collections provide broad film and creative look diversity. HaldCLUTs should be converted to normal 3D LUT tensors, resampled to 17x17x17, and converted to residual form.

Smaller public packs can be used for additional style coverage and parser tests, after basic provenance and format checks.

### 4.5 Derived LUT Quality Filters

Derived LUTs can be poor approximations if the source edit included local masks, sharpening, denoising, healing, geometry, exposure recovery, or other operations not representable by one global LUT.

Each derived LUT should store a quality vector:

```text
fit_error_deltaE
LUT smoothness
clip_rate
foldover / monotonicity violations
neutral_drift
residual_magnitude
source family
expert id or LUT id
```

Bad derived LUTs should be filtered or downweighted. Filtering should be explicit and auditable rather than silent.

## 5. Instruction Data

### 5.1 Instruction Example Format

One supervised training example contains:

```text
model input:
  source image
  natural-language instruction

model target:
  <lut_bos> 64 LUT code tokens <lut_eos>
  or <unsupported>

metadata:
  source LUT id
  LUT source family
  gold prompt tags
  measured behavior vector
  derived LUT quality scores
  teacher scores
  judge scores
```

The graded target image is not part of the model input for v1 instruction-guided grading. It is used for prompt generation, filtering, reward computation, and evaluation.

### 5.2 Image Corpus

Instruction-pair images should mix:

```text
70% Unsplash Lite-style photographic images
30% COCO-style diverse scene images
```

The point is to train on visually useful photo-grading inputs while preserving enough scene diversity that the model does not only work on polished portraits or landscapes.

PPR10K and FiveK remain important sources for deriving expert LUTs, but they are not the main SFT image mix unless needed by the derivation pipeline.

### 5.3 Instruction Corpus Scale

The baseline SFT target is:

```text
50,000 instruction examples
```

After the 50k proof of concept works, scale to:

```text
100,000 instruction examples
```

Use two prompts per accepted image-LUT pair:

```text
1 concise prompt
1 more natural or creative prompt
```

Example:

```text
concise:
  "Make the image warmer with softer contrast."

natural:
  "Give it a gentle warm film look with slightly muted contrast."
```

### 5.4 Source-Balanced Sampling

Do not sample LUTs purely by source count, or PPR10K/FiveK will dominate. Use target proportions:

```text
30% PPR10K-derived expert LUTs
25% FiveK-derived expert LUTs
20% Fresh LUTs
15% G'MIC / RawTherapee HaldCLUTs
10% smaller public LUT packs
```

This keeps the instruction corpus balanced between realistic expert retouching and more expressive creative styles.

### 5.5 Prompt Difficulty Mix

Target mix:

```text
50% simple explicit prompts
  "make it warmer"
  "increase contrast"
  "mute the colors"

30% compound explicit prompts
  "make it warmer and more contrasty with lifted shadows"

15% style-bundle prompts
  "give it a cinematic teal-orange look"

5% unsupported/refusal prompts
  "make only the sky bluer"
  "change the shirt to red"
```

Unsupported prompts should include local region edits, semantic object edits, content generation, geometry/detail changes, and impossible selective-preservation requests.

## 6. Teacher Prompt Generation and Filtering

### 6.1 Teacher Contract

For each image-LUT pair, the teacher should produce:

```text
structured tags:
  ["warmer", "muted", "lifted_blacks", "matte"]

natural prompt:
  "Give the image a warmer matte look with muted colors and lifted blacks."
```

The structured tags make generation auditable. The natural prompt is what the model sees during SFT.

### 6.2 Deterministic Validation

Measured LUT behavior is authoritative for explicit attributes:

```text
warmer -> Lab b* should increase
cooler -> Lab b* should decrease
more saturated -> chroma should increase
muted/desaturated -> chroma should decrease
higher contrast -> luminance spread should increase
softer contrast -> luminance spread should decrease
lifted blacks -> low-percentile luminance should rise
crushed blacks -> low-percentile luminance should fall
```

If tags contradict measured behavior, reject or regenerate the prompt.

### 6.3 Judge Quality Gate

An LLM/VLM judge should check language and semantic quality:

```text
prompt is concise and human-like
prompt matches structured tags
prompt does not mention unsupported local edits
prompt does not leak irrelevant scene content
prompt does not overclaim impossible preservation
style labels are plausible given the tags
```

The judge is a quality gate, not the only source of truth. Deterministic color checks remain authoritative for measurable claims.

## 7. Tokenizer Training

### 7.1 Inputs

Tokenizer inputs are 17x17x17 residual LUT tensors from the approved LUT corpus:

```text
source LUT
        ->
normalize/resample to 17x17x17
        ->
residual = LUT - identity
        ->
VQ tokenizer
```

### 7.2 Losses

Use a reconstruction objective plus VQ commitment/codebook loss:

```text
L_recon: LUT-grid reconstruction loss
L_deltaE: perceptual color error on sampled RGB grid or rendered chart
L_smooth: smoothness regularization
L_clip: out-of-range penalty before final clamp
L_neutral: neutral-axis preservation penalty where applicable
L_commit: VQ commitment/codebook loss
```

### 7.3 Acceptance Gate

Do not begin VLM SFT until the tokenizer passes:

```text
mean reconstruction DeltaE <= 2.0 on held-out LUTs
PSNR >= 35 dB on LUT grid or rendered chart
valid decoded range after bounds/clamp
low smoothness failure rate
no severe codebook collapse
qualitative original vs reconstructed previews are nearly identical
```

The tokenizer is a prerequisite. If tokenizer reconstruction is poor, the VLM can predict correct tokens and still produce poor LUTs.

## 8. VLM Supervised Fine-Tuning

### 8.1 Objective

SFT trains the VLM to predict LUT code tokens:

```text
p(token_t | token_<t, source_image, instruction)
```

Training examples are:

```text
input:
  image + instruction

target:
  <lut_bos> 64 code tokens <lut_eos>
```

Unsupported examples target:

```text
<unsupported>
```

### 8.2 Initial Training Scope

Start with:

```text
freeze vision encoder
LoRA on language model
train or LoRA projector/connector
train new LUT token embeddings/output behavior
```

Scale to fuller fine-tuning only after the 50k SFT loop shows a real pass-rate improvement.

## 9. GRPO Stage

### 9.1 When to Run GRPO

GRPO starts only after SFT has:

```text
high valid-token rate
basic direction-following ability
reasonable safety behavior
nontrivial improvement over baselines
```

### 9.2 First GRPO Size

Start small:

```text
1,000 to 3,000 prompts
4 sampled completions per prompt
```

Scale only after reward curves and qualitative outputs look sane.

### 9.3 Reward Priority

GRPO should optimize:

```text
1. valid 64-token sequence or valid <unsupported>
2. correct unsupported refusal
3. correct direction for explicit prompt attributes
4. LUT safety: clipping, smoothness, foldover, neutral drift
5. similarity to target graded image
6. small aesthetic/style reward
```

Aesthetics must not compensate for wrong direction, invalid tokens, unsafe LUTs, or failure to refuse unsupported prompts.

Reward sketch:

```text
R = R_valid
  + R_refusal
  + R_direction
  + R_safety
  + R_target_similarity
  + small_weight * R_aesthetic
  - hard_penalties
```

## 10. Evaluation Plan

### 10.1 Primary Unit

The primary eval unit is:

```text
source image + instruction
        ->
model output
        ->
decoded LUT or <unsupported>
        ->
measured pass/fail
```

The headline metric is:

```text
prompt-to-LUT pass rate
```

A supported case passes only if:

```text
exactly 64 valid LUT code tokens are emitted
the decoder produces a valid 17x17x17 global LUT
every explicit gold prompt attribute changes in the correct measured direction
clipping, smoothness, foldover, and neutral drift stay within bounds
```

An unsupported case passes only if:

```text
the model emits <unsupported>
```

### 10.2 Held-Out Eval Splits

Use three held-out sets:

```text
seen-family / unseen-pair eval:
  held-out LUTs, images, and prompts from source families seen in training

unseen-family eval:
  held-out LUT source or style families not used in the training split

unsupported-prompt eval:
  local, semantic, generative, geometry/detail, and impossible preservation prompts
```

Minimum eval sizes:

```text
500 supported eval cases
100 unsupported eval cases
50-100 qualitative demo cases
```

### 10.3 Gold Prompt Tags

Eval prompts must have frozen gold tags. They are created during eval-set construction:

```text
choose target LUT
measure LUT behavior
teacher proposes structured tags and natural prompt
deterministic validator checks tags against measured behavior
judge checks prompt/tag language quality
manual review for ambiguous or important eval cases
freeze row
```

Gold tags are not predicted by the model being evaluated.

### 10.4 Metrics

Report:

```text
prompt-to-LUT pass rate
valid-token output rate
unsupported refusal accuracy
direction accuracy by attribute
clip-rate failure
smoothness failure
foldover failure
neutral-drift failure
mean DeltaE to target where target exists
LLM/VLM judge score against behavior spec
```

The judge score is required as part of the project evaluation, but the primary pass/fail behavior is measured with deterministic LUT checks.

### 10.5 Baselines

Compare against:

```text
token baseline:
  Qwen2.5-VL-3B with LUT vocabulary added but no LUT SFT

prompted Qwen baseline:
  normal Qwen2.5-VL-3B prompted to produce a LUT or compact recipe

frontier prompted baseline:
  top multimodal model prompted to produce a LUT or compact recipe

SFT checkpoint:
  trained prompt-to-LUT model

GRPO checkpoint:
  SFT model after small GRPO stage
```

Prompted baselines should be evaluated in two modes:

```text
raw LUT mode:
  model outputs .cube directly

recipe mode:
  model outputs compact JSON recipe
  deterministic renderer converts recipe to LUT
```

The project model uses its native path:

```text
64 LUT tokens -> tokenizer decoder -> full LUT
```

## 11. Success Criteria

The v1 project succeeds if:

```text
tokenizer mean reconstruction DeltaE <= 2.0 on held-out LUTs
SFT improves prompt-to-LUT pass rate by >= 30 percentage points over prompted Qwen baseline
SFT reaches >= 85% valid-token output rate
SFT reaches >= 80% unsupported-refusal accuracy
GRPO improves either prompt-to-LUT pass rate or safety failure rate by >= 5 percentage points over SFT
```

The central claim is not that the model beats frontier systems at general editing. The claim is that a small tuned image-conditioned model reliably performs a constrained prompt-to-LUT behavior that prompting alone does not perform consistently.

## 12. Training and Implementation Order

Implement in this order:

```text
1. Download, parse, and derive LUT corpus.
2. Normalize all accepted LUTs to 17x17x17.
3. Convert LUTs to residual representation.
4. Train VQ tokenizer.
5. Check tokenizer acceptance gate.
6. Generate 50k instruction examples.
7. Train Qwen2.5-VL SFT.
8. Run base-vs-SFT eval.
9. Run small GRPO stage.
10. Run base-vs-SFT-vs-GRPO eval.
11. Package CLI demo and results.
```

No VLM SFT before tokenizer quality is acceptable. No GRPO before SFT beats baseline on basic validity and direction-following.

## 13. CLI Demo

The first demo is a CLI, not an application UI.

Command shape:

```text
prompt_to_lut --image input.jpg --prompt "give it a warm faded film look" --out outputs/run_001
```

Default output artifacts:

```text
outputs/run_001/
  input.png
  graded.png
  preview_side_by_side.png
  output.cube
  output_tokens.txt
  metrics.json
```

`metrics.json` should include:

```text
prompt attributes detected or gold tags when available
valid token count
direction checks
clip rate
smoothness score
foldover score
neutral drift
unsupported flag
decoder/tokenizer metadata
```

A Gradio or application layer can be built later once the core CLI, model, decoder, and evaluation pipeline work.

## 14. Extension Path

The main v1 extension paths are:

```text
100k instruction corpus after 50k proof of concept
larger or fuller VLM fine-tuning after LoRA proof of concept
larger GRPO stage after reward validation
reference-image style transfer
K-LUT local mixtures with masks or learned gates
application/UI layer around the CLI
```

Local edits should be revisited only after the global prompt-to-LUT system is measurable and stable. A local system would require:

```text
multiple decoded LUTs
mask or gating prediction
region-specific eval
halo/smoothness checks
new failure cases
```

That is a separate methodology phase, not part of v1.
