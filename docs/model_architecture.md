# Model Architecture Document

## Objective

Build an image-conditioned, instruction-guided model that predicts one canonical
global color LUT or refuses unsupported requests. The model is not a general
image editor. It is a constrained prompt-to-LUT generator.

The first runnable surface is a CLI. The later child-facing workbench wraps the
same model with preview, compare, undo, revise, and naming workflows.

## System Overview

```text
source image + natural-language instruction
        ->
Qwen2.5-VL-3B-Instruct with added LUT vocabulary
        ->
<lut_bos> 64 LUT code tokens <lut_eos>
        ->
VQ LUT tokenizer decoder
        ->
canonical 17x17x17 residual LUT
        ->
identity LUT + residual LUT
        ->
canonical absolute global LUT
        ->
.cube export + graded image + metrics + version manifest
```

Unsupported request path:

```text
source image + unsupported instruction
        ->
Qwen2.5-VL-3B-Instruct with added LUT vocabulary
        ->
<unsupported>
        ->
visible refusal in CLI/workbench
```

## Major Components

| Component | Responsibility |
| --- | --- |
| LUT corpus builder | ingest, parse, derive, canonicalize, normalize, and filter LUTs |
| VQ LUT tokenizer | compress canonical residual 17x17x17 LUTs into 64 code tokens |
| Instruction corpus builder | pair images, prompts, gold tags, and LUT token targets |
| VLM | predict token sequence or `<unsupported>` from image + instruction |
| Decoder/runtime | constrained decoding, decode tokens, add identity, export `.cube`, apply LUT |
| Evaluator | score syntax, boundary, color direction, target fidelity, safety, and baselines |
| CLI | run inference and produce versioned artifacts |
| Workbench | later UI for child-facing comparison and revision |

## Base Model

Use:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

Reasons:

- Small enough for Colab QLoRA experimentation.
- Multimodal input supports image-conditioned behavior.
- Instruct tuning gives natural-language instruction handling.
- 3B scale keeps the project aligned with "small tuned behavior" rather than
  general frontier capability.

## Output Vocabulary

Add special tokens:

```text
<lut_bos>
<lut_eos>
<unsupported>
<lut_000>
<lut_001>
...
<lut_255>
```

The tokenizer resize operation must initialize and train:

- new input embedding rows;
- new LM head rows;
- any tied embedding/head weights consistently.

The training loss is applied only to the assistant target tokens.

## Output Grammar

Supported output:

```text
<lut_bos> <lut_###> x64 <lut_eos>
```

Unsupported output:

```text
<unsupported>
```

No other output format is valid. The runtime parser is strict because loose
parsing would hide model failures.

CLI/product decoding must use grammar-constrained token-id decoding:

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

The grammar mask must not use gold support labels or inferred prompt attributes.
It only enforces syntax. Free-generation eval keeps the grammar mask disabled to
measure learned syntax validity.

## Canonical LUT Domain

Canonical LUT domain for v1 is:

```text
display-referred IEC 61966-2-1 sRGB
encoded RGB [0,1]
D65
17x17x17 grid
full-range RGB
trilinear interpolation
```

The canonical LUT maps encoded sRGB input values to encoded sRGB output values.
Residual LUT equals canonical absolute LUT minus the canonical encoded-sRGB
identity grid at the same nodes.

Raw/source LUTs must be color-managed into this domain before hashing, residual
conversion, tokenizer encoding, prompt tagging, export, or evaluation.

ICC conversion is part of the canonical domain contract. Embedded Display P3,
AdobeRGB, ProPhoto, RAW-derived, or other tagged sources are converted to
canonical sRGB with the pinned color-management module, relative-colorimetric
intent, black-point compensation enabled, deterministic gamut clipping to
`[0,1]`, and float32 working precision unless a later manifest explicitly
changes those settings. Unknown profiles are recorded as assumed sRGB.

Canonical `.cube` export is Adobe/Resolve-compatible and deterministic:

```text
LUT_3D_SIZE 17
DOMAIN_MIN 0 0 0
DOMAIN_MAX 1 1 1
table values are full canonical absolute RGB, not residuals
axis convention is RGB with R changing fastest, then G, then B
float format is fixed decimal with 10 digits after the decimal point
line endings are LF, encoding is UTF-8, and no timestamps or comments vary
cube_serialization_version = cube_v1_size17_domain01_rgb_rfast_f10_lf
```

The `.cube` hash is over those canonical bytes. The runtime must not use a
different table order or formatting and then call the artifact equivalent.

## LUT Tokenizer

The VQ tokenizer is trained before VLM SFT.

Input:

```text
raw/source LUT
        ->
color-managed canonical absolute sRGB 17x17x17 LUT
        ->
residual = canonical LUT - encoded-sRGB identity
        ->
VQ tokenizer
```

Architecture target:

```text
17x17x17x3 residual LUT
        ->
encoder
        ->
4x4x4 latent grid
        ->
64 discrete codebook indices
        ->
decoder
        ->
17x17x17x3 reconstructed residual LUT
```

Configuration:

| Parameter | Value |
| --- | --- |
| LUT grid | 17x17x17 |
| Representation | canonical residual LUT |
| Latent grid | 4x4x4 |
| Token count | 64 |
| Codebook size | 256 |
| VQ type | single-stage VQ v1 |
| Output at runtime | full canonical absolute LUT after identity addition |

Frozen tokenizer manifest fields:

```text
lut_grid_size = 17x17x17
representation = residual_after_identity
canonical_domain_id = slm_lut_v1_srgb_display_encoded_17_trilinear
interpolation = trilinear
latent_grid = 4x4x4
token_count = 64
codebook_size = 256
tensor_axis_order
cube_table_order
latent_flatten_order
token_suffix_to_codebook_index mapping
code_id_to_codebook_row
vq_codebook_sha256
vq_decoder_sha256
encoder/decoder layer table
lut_corpus_hash
tokenizer_weights_hash
color_pipeline_version
cube_serialization_version
```

Recommended exact 17 -> 4 geometry:

```text
Encoder:
  17 -> 9  via Conv3d k=3 s=2 p=1
   9 -> 5  via Conv3d k=3 s=2 p=1
   5 -> 4  via Conv3d k=2 s=1 p=0

Decoder:
   4 -> 5  via ConvTranspose3d k=2 s=1 p=0
   5 -> 9  via ConvTranspose3d k=3 s=2 p=1 output_padding=0
   9 -> 17 via ConvTranspose3d k=3 s=2 p=1 output_padding=0
```

Losses:

```text
L_recon: LUT-grid reconstruction
L_deltaE: perceptual color error on chart samples
L_smooth: smoothness regularization
L_clip: pre-clamp out-of-range penalty
L_neutral: neutral-axis preservation penalty
L_commit: VQ commitment/codebook loss
```

Acceptance gates:

- mean reconstruction DeltaE00 <= 2.0 on held-out LUTs;
- p95 DeltaE00 <= 4.0;
- p99 DeltaE00 <= 6.0;
- max DeltaE00 <= 10.0 or explicitly reviewed exception;
- PSNR >= 35 dB mean, p5 PSNR >= 30 dB;
- per-family mean DeltaE00 <= 2.5 when the family has enough rows;
- per-family p95 DeltaE00 <= 5.0;
- per-target SFT admission mean DeltaE00 <= 3.0 and p95 <= 6.0;
- valid finite decoded range;
- low smoothness failure rate;
- no severe codebook collapse;
- alert if active codes < 70% or perplexity < 64;
- identity/ramp/flatten and `.cube` roundtrip tests pass;
- qualitative reconstructed previews are nearly identical to source LUT previews.

Use EMA VQ, dead-code revival, family-balanced batches, and LUT augmentation if
codebook usage or tail errors fail. Stay with single-stage VQ for v1 unless these
mechanics still fail the gates; switching to RVQ requires a new token grammar and
updated docs.

VLM SFT does not start until the tokenizer passes its gate. If the tokenizer is
poor, perfect token prediction still produces bad LUTs.

## VLM Fine-Tuning Architecture

Training mode:

```text
4-bit QLoRA base model + LoRA adapters
```

Default module policy:

| Module | V1 Policy |
| --- | --- |
| Vision encoder | frozen |
| Language model attention/MLP projections | LoRA |
| Multimodal projector/connector | LoRA or small full-trainable module |
| New token embeddings | train row-selectively |
| New LM head rows | train row-selectively |
| Full LM weights | frozen in QLoRA |

LoRA target modules:

```text
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

Starting LoRA settings:

```text
r = 16
alpha = 32
dropout = 0.05
```

Quantization:

```text
4-bit NF4
double quantization enabled
bf16 compute when available, otherwise fp16
```

The projector/connector choice must be explicit in run config. If Colab memory
allows, train the projector fully with a lower learning rate. If not, apply
adapters or freeze it and report that limitation.

## Vocabulary Resize And Embedding Preflight

Before any full SFT run, assert:

- `len(tokenizer) == input_embedding.num_embeddings == lm_head.out_features`.
- `<lut_000>` through `<lut_255>` are contiguous or explicitly mapped.
- Token suffix to codebook index is asserted.
- `<lut_bos>`, `<lut_eos>`, and `<unsupported>` are unique and do not collide
  with LUT code ids.
- Codebook size is 256 and flatten order is recorded.
- New token rows are mean/stat-initialized from existing token embeddings.
- Tied embedding/head status is detected and logged.
- Only intended rows/modules are trainable: LoRA, projector policy, and new
  token rows.
- Old embedding/head rows remain unchanged after a smoke optimizer step.
- Save/load roundtrip preserves tokenizer ids, resized rows, adapter weights,
  and first-batch logits within tolerance.

Prefer a row-selective embedding/head adapter or gradient mask so only the 259
new rows train. Do not rely on `modules_to_save=["embed_tokens","lm_head"]`
unless accepting full-matrix training, memory cost, and old-vocab drift. If
embeddings are tied, handle input/head as one shared matrix; if untied,
initialize and save both consistently.

## Dataset Interface

The instruction dataset consumed by SFT uses JSONL or Parquet.

Supported row:

```json
{
  "id": "train_000001",
  "image_path": "images/train/000001.jpg",
  "image_sha256": "...",
  "instruction": "Give it a warm matte look with muted colors.",
  "assistant_target": "<lut_bos> <lut_042> ... <lut_eos>",
  "target_tokens": [42, 17, 200, 5, "... 64 code ids total ...", 128],
  "is_supported": true,
  "source_lut_id": "freshluts_123",
  "source_family": "freshluts",
  "gold_tags": ["warmer", "matte", "muted"],
  "measured_behavior": {
    "temperature_delta_b": 2.3,
    "contrast_l_spread_delta": -3.1,
    "chroma_delta": -2.0,
    "skin_locus_deltaE00_p95": 4.2
  },
  "derived_lut_quality": {
    "representability_tier": "gold",
    "fit_deltaE00_mean": null,
    "fit_deltaE00_p95": null,
    "supported_cell_rate": null
  },
  "canonical_domain_id": "slm_lut_v1_srgb_display_encoded_17_trilinear",
  "canonical_absolute_lut_hash": "...",
  "canonical_residual_lut_hash": "...",
  "tokenizer_version": "...",
  "vq_codebook_sha256": "...",
  "vq_decoder_sha256": "...",
  "prompt_template_family": "style_bundle_v1",
  "prompt_generation_batch_id": "teacher_batch_2026_07_01",
  "split_unit_id": "...",
  "headline_eligible": true,
  "split": "train"
}
```

Unsupported row:

```json
{
  "id": "train_unsupported_000001",
  "image_path": "images/train/000002.jpg",
  "instruction": "Make only the shirt red.",
  "assistant_target": "<unsupported>",
  "target_tokens": [],
  "is_supported": false,
  "support_label": "unsupported",
  "unsupported_category": "semantic_object_recolor",
  "unsupported_components": ["semantic_object_recolor"],
  "supported_components": [],
  "mixed_prompt": false,
  "split": "train"
}
```

Mixed unsupported row:

```json
{
  "instruction": "Make the whole photo warmer and remove the background.",
  "assistant_target": "<unsupported>",
  "is_supported": false,
  "unsupported_category": "mixed_partial_supported_plus_content_generation",
  "unsupported_components": ["content_removal"],
  "supported_components": ["warmer"],
  "mixed_prompt": true,
  "boundary_pair_id": "mixed_001"
}
```

## Runtime Inference

CLI command:

```text
prompt_to_lut --image input.jpg --prompt "make it warmer and softer" --out outputs/run_001
```

Runtime steps:

Before step 1, process startup runs the manifest self-check described in "Version Manifest And Startup Assertions"; inference does not begin unless those startup assertions pass.

1. Load image and instruction.
2. Read embedded ICC profile; convert image to canonical sRGB for LUT
   application and metrics using the pinned CMM, rendering intent,
   black-point-compensation, gamut-mapping, and float-precision settings.
   Unknown profile is recorded as assumed sRGB.
3. Format Qwen2.5-VL chat input.
4. Generate through grammar-constrained token-id decoding.
5. Parse output strictly.
6. If `<unsupported>`, write refusal artifacts and stop.
7. If token sequence, decode 64 token ids through frozen VQ decoder.
8. Add identity LUT.
9. Run safety and measured-behavior checks.
10. Validate version manifest compatibility.
11. Export `.cube`.
12. Apply LUT to image using the same in-memory canonical LUT and trilinear
    interpolation.
13. Write previews, `metrics.json`, and `version_manifest.json`.

Invalid decoded LUTs must be blocked. The runtime must not silently replace an
invalid LUT with identity.

Determinism requirements:

```text
do_sample=false
num_beams=1
fixed seed values
eval mode / dropout disabled
inference artifact type recorded: 4-bit adapter or merged fp16/bf16
compute dtype recorded
quantization config recorded
image preprocessing recorded
interpolation method recorded
color pipeline recorded
ICC conversion config recorded
cube serialization version recorded
hardware/CUDA/cuDNN/kernel determinism scope recorded
```

Bit-identical `output_tokens.txt` and `.cube` hashes are required only under the
same manifest, hardware class, CUDA/cuDNN/kernel determinism flags, library
versions, quantization/dtype settings, ICC conversion config, and `.cube`
serialization version. Across materially different environments, compare LUTs
with tolerance-based metrics rather than byte hashes.

## Version Manifest And Startup Assertions

Every CLI/eval artifact set includes `version_manifest.json` binding:

```text
base_model_id, base_model_revision
adapter_id, adapter_sha256, adapter_step
text_tokenizer_revision
added_special_token_ids
vocab_size_after_resize
embedding_rows
lm_head_rows
tied_embedding_status
vq_codebook_sha256
vq_decoder_sha256
codebook_size = 256
token_count = 64
latent_shape = 4x4x4
token_suffix_to_codebook_index mapping
flatten_order
lut_grid = 17x17x17
canonical_domain_id
color_pipeline_version
icc_conversion_config
cube_serialization_version
interpolation
parser_version
fsm_version
safety_threshold_version
eval_config_version
active_set_version
eval_set_version
determinism_scope
library versions
```

Startup fails if vocab size, special-token ids, codebook size,
`vq_codebook_sha256`, `vq_decoder_sha256`, flatten order, canonical domain,
color pipeline, ICC conversion config, or `.cube` serialization version differ
from the manifest.
Retraining the VQ tokenizer changes decoded token meaning and therefore requires
a new manifest, regenerated targets, and re-evaluation.

## Safety Architecture

Safety gates run outside the VLM:

- strict output parser;
- grammar-constrained runtime decoder;
- tokenizer decoder finite-value check;
- canonical-domain manifest check;
- output range check;
- clipping and pre-clamp violation;
- smoothness;
- foldover/monotonicity;
- neutral-axis drift;
- intrinsic skin-locus LUT-domain gate;
- unsupported boundary evaluator.

This keeps reward training and inference honest. The model does not get
aesthetic credit for invalid syntax, wrong color direction, poor target fidelity,
unsafe LUTs, or failure to refuse unsupported prompts.

## Rollout Optimization Architecture

GRPO is no longer the first post-SFT optimization step.

Order:

1. Run rejection sampling over SFT: sample 4-8 completions per prompt, score with
   deterministic gates, and SFT on winners.
2. Run DPO from winner/loser pairs if rejection sampling improves quality but not
   enough.
3. Escalate to GRPO only if RS/DPO plateaus, invalid rollout rate is low, reward
   hacking checks pass, and improvements are outside confidence intervals.

Reward priority for RS/DPO/GRPO:

1. valid 64-token sequence or valid `<unsupported>`;
2. correct support/refusal boundary;
3. correct prompt direction;
4. LUT safety;
5. target fidelity;
6. style discriminability;
7. small style/aesthetic score.

Hard failures zero out or heavily penalize later reward terms. Aesthetic reward
cannot compensate for invalid tokens, wrong direction, target mismatch, unsafe
LUTs, or boundary failure.

GRPO-specific config must pin KL/reference policy, rollout budget, generation
backend, seeds, and CI-based pass/fail. GRPO ships only if it beats RS/DPO, not
just SFT, under the eval harness gates.

## Colab Constraints

Colab is the expected environment, but not a guarantee of full Track B
throughput.

Assumptions:

- A100 or L4 preferred for final SFT.
- T4 can run smoke tests, tokenizer experiments, data validation, and small
  batches.
- Image resolution should be capped for training.
- Heavy scraping, RAW conversion, and LUT fitting should be done as resumable
  preprocessing jobs, not inside one fragile notebook session.
- All artifacts should checkpoint to Google Drive or Hugging Face Hub.

Memory fallback order:

1. reduce image pixel budget;
2. reduce per-device batch to 1;
3. increase gradient accumulation;
4. freeze projector;
5. reduce LoRA target modules;
6. shorten max sequence length;
7. run SFT without rollout optimization.

## Workbench Extension

The workbench is not part of the first implementation gate. It wraps the same
model/runtime and adds:

- version history;
- side-by-side compare;
- undo and revise;
- suggested global rewrites for unsupported prompts;
- "what changed" labels derived from measured metrics;
- child-safe feedback rules.

No separate companion personality or tutor-like grading is part of the model
architecture.
