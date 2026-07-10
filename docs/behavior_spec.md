# Prompt-to-LUT Behavior Spec

The system receives a source image and free-form user text. A small distilled
interpreter LM maps the user text to an AttributeSpec plus a route -- `grade`,
`clarify`, or `refuse` (schema: `docs/attribute_spec.md`, behavior_v2). On
`grade`, the generator (the same Qwen2.5-VL-3B QLoRA), conditioned on the
interpreter's `attribute_spec_text` and the image, must output exactly 64 valid
LUT code tokens enclosed by `<lut_bos>` and `<lut_eos>`. Those tokens must decode
through the frozen VQ decoder into one canonical 17x17x17 residual global LUT
which, after adding identity and applying to the image, changes every explicit
gold prompt attribute in the correct measured direction, matches the target grade
within the target-fidelity gate, and passes LUT safety gates. Otherwise the whole
system emits exactly `<unsupported>` (route `refuse`), for either `out_of_scope`
or `out_of_gamut` instructions. See ADRs 0020-0023.

Given an unsupported instruction requiring local, semantic, generative,
geometry/detail, relighting, reference-transfer, or impossible
selective-preservation behavior, the model must output exactly `<unsupported>`
instead of LUT tokens.

Mixed prompts are unsupported in v1 when any required component is unsupported.
For example, "make it warmer and remove the background" must produce
`<unsupported>`, not a partial warm LUT.

## Decision Rationale

The v1 system is limited to one global LUT because a single 3D LUT applies the
same RGB mapping everywhere in an image. Local, object-specific, relighting,
reference-transfer, or selective-preservation edits require masks,
segmentation, additional conditioning, or multiple LUTs with spatial gating, so
they are treated as extension work rather than part of the first behavior
target.

The model outputs discrete LUT code tokens instead of raw LUT floats because a
17x17x17 LUT contains 14,739 numeric channel values, which is brittle for
language-model generation. A fixed 64-token sequence gives the VLM a compact
generation target, supports supervised token training, and makes rollout-based
optimization practical by sampling, decoding, and scoring candidate token
sequences.

The tokenizer represents residual LUTs rather than absolute LUTs because most
color grades are controlled changes around the identity transform. Predicting
the residual gives the system a safer default, makes reconstruction easier on a
smaller dataset, and still exports a normal full LUT after the residual is added
back to the identity LUT.

The first task is instruction-guided grading only. Reference-style transfer and
local LUT mixtures are valuable extensions, but they add separate conditioning
and evaluation problems. A focused image-plus-text-to-LUT task is easier to
evaluate against a falsifiable behavior spec within the project scope.

The LUT corpus is real/expert-first because the model should learn from
photographic grades rather than only procedural color transforms.
PPR10K-derived expert LUTs, FiveK-derived expert LUTs, Fresh LUTs,
G'MIC/RawTherapee HaldCLUTs, and smaller public LUT packs provide a mixture of
professional retouching and creative styles, with derived-LUT representability
gates preventing poor global approximations from entering headline training or
evaluation.

The primary metric is prompt-to-LUT pass rate because the project goal is
reliable constrained behavior, not general image aesthetics. Aesthetic
preference can be used as a small secondary reward, but it must not compensate
for invalid tokens, wrong prompt direction, poor target fidelity, unsupported
local edits, clipping, foldover, unsafe neutral drift, or skin-locus safety
failure.

## Success Thresholds

Final thresholds are CI-gated. Point estimates alone are not sufficient for
ship/no-ship claims.

- Tokenizer reconstruction must pass mean, tail, per-family, and per-target
  gates before its tokens are used for SFT targets.
- Runtime/CLI decoding must use grammar-constrained token-id decoding; free
  generation is still evaluated separately to measure learned syntax validity.
- The SFT checkpoint must clear boundary gates: unsupported recall,
  unsupported precision, boundary F1, over-refusal ceiling, and mixed-prompt
  recall.
- The SFT checkpoint must beat the best null/constant baselines and the
  deterministic renderer gate, not only a prompted-Qwen baseline.
- The SFT checkpoint must pass target-fidelity, safety, skin-locus, and
  style-discriminability gates on headline-eligible eval rows.
- GRPO may ship only after simpler RS/DPO stages plateau and GRPO improves over
  the best prior tuned checkpoint outside paired confidence intervals without
  violating over-refusal or safety gates.
