# Prompt-to-LUT Color Grading

This context defines the project language for a small model that turns color-grading prompts into 3D lookup tables. It keeps the boundaries between global color transforms and later local-edit extensions explicit.

## Language

**Global LUT**:
A single 3D lookup table that maps input RGB colors to output RGB colors uniformly across the whole image.
_Avoid_: Local LUT, masked LUT

**Local Edit**:
A spatially constrained color change that applies differently to different image regions or semantic objects.
_Avoid_: Global edit

**Local LUT Extension**:
A future architecture that combines multiple LUTs with masks or gating weights to support local edits.
_Avoid_: V1 LUT, global LUT

**Image-Conditioned Prompt-to-LUT**:
A color-grading task where the system receives both the source image and user text: the interpreter produces an AttributeSpec, and the generator, conditioned on that AttributeSpec plus the image, produces a global LUT.
_Avoid_: Prompt-only LUT generation

**Instruction-Guided Grading**:
A prompt-to-LUT task where the system receives a source image and a text instruction describing the desired global color grade; the interpreter maps the instruction to an AttributeSpec that conditions the generator.
_Avoid_: Reference-style transfer

**Reference-Style Transfer**:
A color-grading task where the model receives a source image and a separate reference image whose color style should be copied.
_Avoid_: V1 task, instruction-guided grading

**Interpreter**:
The small model mapping any user text to an AttributeSpec plus a route.
_Avoid_: teacher model, generator

**AttributeSpec**:
The structured, high-resolution color-attribute representation the Interpreter produces from user text and the Generator is conditioned on; shares the behavior_v2 axis schema with the measured behavior vector (see docs/attribute_spec.md).
_Avoid_: recipe, JSON recipe, structured recipe

**Route (grade / clarify / refuse)**:
The interpreter's three-way decision.
_Avoid_: render (implies the deferred parametric renderer)

**Generator**:
The Qwen2.5-VL-3B QLoRA model conditioned on attribute_spec_text (+image) that emits the 64 LUT code tokens.
_Avoid_: interpreter, teacher model

**LUT Token Sequence**:
A fixed-length sequence of discrete tokens that represents one global LUT before it is decoded back into a full LUT tensor. The interpreter's upstream output is the AttributeSpec, not a "recipe".
_Avoid_: JSON recipe, raw LUT floats

**LUT Codebook**:
The finite set of discrete visual-color codes used by the LUT tokenizer to represent parts of a global LUT.
_Avoid_: Prompt vocabulary, text tokens

**VQ Codebook Value Hash**:
The `vq_codebook_sha256` value that binds LUT token ids to the frozen codebook vectors.
_Avoid_: Codebook size

**VQ Decoder Hash**:
The `vq_decoder_sha256` value that binds LUT token ids to the frozen decoder artifact.
_Avoid_: Decoder name

**Identity LUT**:
A global LUT that leaves every input RGB color unchanged.
_Avoid_: Neutral preset

**Residual LUT**:
A representation of a color grade as the difference between a target global LUT and the identity LUT.
_Avoid_: Absolute LUT

**Absolute LUT**:
A full global LUT that directly stores final output RGB values at each grid point.
_Avoid_: Residual LUT

**Canonical LUT Domain**:
The v1 artifact contract for all accepted LUT tensors: display-referred IEC 61966-2-1 sRGB, encoded RGB values in [0,1], D65, 17x17x17 grid, trilinear interpolation, and ICC-converted source material with pinned wide-gamut conversion behavior.
_Avoid_: Raw source LUT domain

**Canonical Cube Serialization**:
The deterministic `.cube` byte format for exported LUTs: `LUT_3D_SIZE 17`, `DOMAIN_MIN/MAX` of 0/1, RGB axis order with R fastest, fixed float formatting, LF line endings, UTF-8, and no timestamps.
_Avoid_: Tool-default LUT export

**Canonical Absolute LUT**:
An absolute global LUT after conversion into the canonical LUT domain.
_Avoid_: RAW/ProPhoto/Display P3 source LUT

**Canonical Residual LUT**:
The canonical absolute LUT minus the encoded-sRGB identity grid at the same 17x17x17 nodes.
_Avoid_: Linear RGB residual

**Expert LUT**:
A global LUT derived from a professional retouching adjustment, such as applying an expert preset to an identity color grid.
_Avoid_: Synthetic LUT

**PPR10K-Derived Expert LUT**:
An expert LUT produced from PPR10K portrait retouching targets by converting expert adjustment metadata into a global LUT.
_Avoid_: Generic LUT pack

**Derived LUT**:
A global LUT extracted from an edit preset or before/after expert edit. For headline training or eval, a derived LUT must pass representability gates, spatial residual checks, and support-map checks; a best-fit approximation alone is not enough.
_Avoid_: Source LUT, hand-authored LUT

**FiveK-Derived Expert LUT**:
A derived LUT fitted from a MIT-Adobe FiveK source image and one expert-retouched target image.
_Avoid_: ISP transform, camera-pipeline transform

**Derived LUT Quality Score**:
A set of measurements describing how well a derived LUT behaves as a global color transform, including fit error, held-out fit error, spatial residual structure, support-map coverage, smoothness, clipping, foldover, neutral drift, skin-locus shift, and residual magnitude.
_Avoid_: Aesthetic score

**Supported Prompt Attribute**:
A color-grading intent that v1 is expected to convert into a global LUT and evaluate with measurable color statistics. Supported attributes now include the behavior_v2 hue axes — global hue cast (angle + magnitude), per-tone-region hue (shadow/midtone/highlight), per-hue saturation, contrast shape (toe/shoulder), and matte — not just the two Lab axes (temperature/tint).
_Avoid_: Arbitrary edit request

**Style Bundle**:
A named grading style defined as a measurable combination of supported prompt attributes.
_Avoid_: Vague style label

**Instruction Example**:
A supervised training example whose model input is a source image and natural-language grading instruction, and whose target is the LUT token sequence for the intended global LUT.
_Avoid_: Before/after pair as model input

**Prompt-to-LUT Pass Rate**:
The primary evaluation metric: the fraction of headline-eligible image-and-instruction cases where the model outputs valid LUT tokens that decode to a safe canonical global LUT matching all explicit prompt attributes and the target-fidelity gate, or correctly refuses unsupported prompts.
_Avoid_: Aesthetic score as primary metric

**Unsupported Output**:
A dedicated interpreter output on the refuse route indicating that a prompt cannot be represented by one global LUT, covering both out_of_scope and out_of_gamut intents (e.g. infrared, pure-primary, hue-rotation).
_Avoid_: Identity LUT as refusal

**V1 Base Model**:
The primary vision-language model used for instruction-guided prompt-to-LUT training.
_Avoid_: Teacher model, judge model

**LUT Vocabulary**:
The model's special output tokens for LUT generation, consisting of LUT control tokens and one token for each LUT codebook entry.
_Avoid_: Natural-language vocabulary

**Tokenizer Acceptance Gate**:
The quality threshold a LUT tokenizer must pass before its tokens are used as supervised targets for VLM training.
_Avoid_: Final model evaluation

**Reward Config Version**:
The versioned reward specification for RS/DPO/GRPO, including lexicographic priority, hard penalties, margins, and adversarial reward-hacking test set.
_Avoid_: Reward version prose

**Determinism Scope**:
The manifest-bound environment in which identical prompts, images, model artifacts, ICC config, and `.cube` serialization must produce byte-identical tokens and LUT files.
_Avoid_: Cross-hardware reproducibility guarantee

**CLI Demo Artifact**:
The set of files produced by one CLI run, including the decoded `.cube` LUT, graded image, side-by-side preview, output token sequence, and evaluation metrics.
_Avoid_: Application UI

**Held-Out Eval Set**:
A reserved multi-slice evaluation set including usage-weighted headline, coverage macro, image sensitivity, subtle control, style discriminability, expert holdout, cross-source expert, unseen-family, unsupported, mixed unsupported, boundary pairs, procedural diagnostic, real-world CLI inputs, and qualitative demo slices.
_Avoid_: Training split

**Gold Prompt Tags**:
Frozen structured labels for an instruction, created during dataset construction and used by evaluation to check whether the decoded LUT matches the requested attributes; tags map to the unified behavior_v2 vocabulary.
_Avoid_: Model-predicted tags
