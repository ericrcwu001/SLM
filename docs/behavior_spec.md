# Prompt-to-LUT Behavior Spec

Given a source image and a supported global color-grading instruction, the model must output exactly 64 valid LUT code tokens that decode into a 17x17x17 residual global LUT. After adding the identity LUT and applying the result, every explicit gold prompt attribute must change in the correct measured direction while clipping, smoothness, foldover, and neutral-drift metrics remain within bounds.

Given an unsupported instruction requiring local, semantic, generative, geometry/detail, or impossible selective-preservation edits, the model must output `<unsupported>` instead of LUT tokens.

## Decision Rationale

The v1 system is limited to one global LUT because a single 3D LUT applies the same RGB mapping everywhere in an image. Local, object-specific, or selective-preservation edits require masks, segmentation, or multiple LUTs with spatial gating, so they are treated as extension work rather than part of the first behavior target.

The model outputs discrete LUT code tokens instead of raw LUT floats because a 17x17x17 LUT contains 14,739 numeric channel values, which is brittle for language-model generation. A fixed 64-token sequence gives the VLM a compact generation target, supports supervised token training, and makes GRPO practical by sampling, decoding, and scoring candidate token sequences.

The tokenizer represents residual LUTs rather than absolute LUTs because most color grades are controlled changes around the identity transform. Predicting the residual gives the system a safer default, makes reconstruction easier on a smaller dataset, and still exports a normal full LUT after the residual is added back to the identity LUT.

The first task is instruction-guided grading only. Reference-style transfer and local LUT mixtures are valuable extensions, but they add separate conditioning and evaluation problems. A focused image-plus-text-to-LUT task is easier to evaluate against a falsifiable behavior spec within the project scope.

The LUT corpus is real/expert-first because the model should learn from photographic grades rather than only procedural color transforms. PPR10K-derived expert LUTs, FiveK-derived expert LUTs, Fresh LUTs, G'MIC/RawTherapee HaldCLUTs, and smaller public LUT packs provide a mixture of professional retouching and creative styles, with derived-LUT quality filters preventing poor global approximations from entering training.

The primary metric is prompt-to-LUT pass rate because the project goal is reliable constrained behavior, not general image aesthetics. Aesthetic preference can be used in GRPO, but it must not compensate for invalid tokens, wrong prompt direction, unsupported local edits, clipping, foldover, or unsafe neutral drift.

## Success Thresholds

- Tokenizer mean reconstruction DeltaE must be less than or equal to 2.0 on held-out LUTs.
- The SFT checkpoint must improve prompt-to-LUT pass rate by at least 30 percentage points over the prompted Qwen baseline.
- The SFT checkpoint must reach at least 85% valid-token output rate.
- The SFT checkpoint must reach at least 80% unsupported-refusal accuracy.
- The GRPO checkpoint must improve either prompt-to-LUT pass rate or safety failure rate by at least 5 percentage points over SFT.
