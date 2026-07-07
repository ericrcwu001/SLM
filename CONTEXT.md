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
A color-grading task where the model receives both the source image and a natural-language grading prompt before producing a global LUT.
_Avoid_: Prompt-only LUT generation

**Instruction-Guided Grading**:
A prompt-to-LUT task where the model receives a source image and a text instruction describing the desired global color grade.
_Avoid_: Reference-style transfer

**Reference-Style Transfer**:
A color-grading task where the model receives a source image and a separate reference image whose color style should be copied.
_Avoid_: V1 task, instruction-guided grading

**LUT Token Sequence**:
A fixed-length sequence of discrete tokens that represents one global LUT before it is decoded back into a full LUT tensor.
_Avoid_: JSON recipe, raw LUT floats

**LUT Codebook**:
The finite set of discrete visual-color codes used by the LUT tokenizer to represent parts of a global LUT.
_Avoid_: Prompt vocabulary, text tokens

**Identity LUT**:
A global LUT that leaves every input RGB color unchanged.
_Avoid_: Neutral preset

**Residual LUT**:
A representation of a color grade as the difference between a target global LUT and the identity LUT.
_Avoid_: Absolute LUT

**Absolute LUT**:
A full global LUT that directly stores final output RGB values at each grid point.
_Avoid_: Residual LUT

**Expert LUT**:
A global LUT derived from a professional retouching adjustment, such as applying an expert preset to an identity color grid.
_Avoid_: Synthetic LUT

**PPR10K-Derived Expert LUT**:
An expert LUT produced from PPR10K portrait retouching targets by converting expert adjustment metadata into a global LUT.
_Avoid_: Generic LUT pack

**Derived LUT**:
A global LUT extracted from an edit preset or before/after expert edit, including cases where the LUT is the best global approximation of a richer retouching operation.
_Avoid_: Source LUT, hand-authored LUT

**FiveK-Derived Expert LUT**:
A derived LUT fitted from a MIT-Adobe FiveK source image and one expert-retouched target image.
_Avoid_: ISP transform, camera-pipeline transform

**Derived LUT Quality Score**:
A set of measurements describing how well a derived LUT behaves as a global color transform, including fit error, smoothness, clipping, foldover, neutral drift, and residual magnitude.
_Avoid_: Aesthetic score

**Supported Prompt Attribute**:
A color-grading intent that v1 is expected to convert into a global LUT and evaluate with measurable color statistics.
_Avoid_: Arbitrary edit request

**Style Bundle**:
A named grading style defined as a measurable combination of supported prompt attributes.
_Avoid_: Vague style label

**Instruction Example**:
A supervised training example whose model input is a source image and natural-language grading instruction, and whose target is the LUT token sequence for the intended global LUT.
_Avoid_: Before/after pair as model input

**Prompt-to-LUT Pass Rate**:
The primary evaluation metric: the fraction of image-and-instruction cases where the model outputs valid LUT tokens that decode to a safe global LUT matching all explicit prompt attributes.
_Avoid_: Aesthetic score as primary metric

**Unsupported Output**:
A dedicated model output indicating that a prompt cannot be represented by one global LUT.
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

**CLI Demo Artifact**:
The set of files produced by one CLI run, including the decoded `.cube` LUT, graded image, side-by-side preview, output token sequence, and evaluation metrics.
_Avoid_: Application UI

**Held-Out Eval Set**:
A reserved evaluation set split into seen-family supported prompts, unseen-family supported prompts, and unsupported prompts.
_Avoid_: Training split

**Gold Prompt Tags**:
Frozen structured labels for an instruction, created during dataset construction and used by evaluation to check whether the decoded LUT matches the requested attributes.
_Avoid_: Model-predicted tags
