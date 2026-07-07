# Master Plan

## Project Thesis

Prompt-to-LUT Color Playground trains a small image-conditioned model to turn a
natural-language global color-grading instruction into one canonical global LUT,
or refuse when the request exceeds what a global LUT can do.

The project should be judged by reliable constrained behavior, not by general
image-editing ability or aesthetic preference.

## Selected Direction

Use Track B with caveats:

- Build the real prompt-to-LUT architecture.
- Keep the first delivery CLI-first.
- Build the child-facing workbench later.
- Collect/scrape the broad source corpus for personal research use.
- Do not let raw source size determine training size.
- Cull the active SFT set to about 10k-15k examples, default 12k.
- Add a cheap generative LUT-token warmup before instruction SFT.
- Record provenance even though data rights are not a blocker for this personal
  project.
- Use Colab QLoRA for SFT.
- Use RS/DPO before GRPO; run GRPO only after simpler tuned stages plateau and
  reward correctness is proven.

## Document Set

| Document | Purpose |
| --- | --- |
| `docs/behavior_spec.md` | compact falsifiable behavior contract |
| `docs/detailed_behavior_spec.md` | full behavioral contract, boundaries, style recipes, thresholds, CLI/workbench behavior |
| `docs/eval_harness_implementation.md` | eval layers, parser, constrained decoding modes, metrics, splits, baselines, reports |
| `docs/model_architecture.md` | model, tokenizer, vocabulary, runtime, safety, version manifest, rollout architecture |
| `docs/data_collection_plan.md` | scraping, derivation, provenance, representability gates, usage-aware culling, active dataset |
| `docs/training_plan_colab.md` | Colab notebooks, tokenizer training, warmup, QLoRA SFT, RS/DPO, optional GRPO |
| `docs/master_plan.md` | consolidated project direction |

`docs/lut_methodology_improvement_plan.md` is superseded. `docs/project spec.md`
is the external/course brief, not project methodology authority.

## Core Behavior

Supported prompt:

```text
source image + global color instruction
        ->
<lut_bos> exactly 64 valid LUT code tokens <lut_eos>
        ->
canonical 17x17x17 residual global LUT
        ->
full .cube LUT + graded image
```

Unsupported prompt:

```text
source image + local/semantic/generative/relighting/reference/etc. instruction
        ->
<unsupported>
```

Mixed prompt:

```text
supported global request + any unsupported component
        ->
<unsupported>
```

The model output contains no prose. Explanations and "what changed" labels are
generated later from metrics and UI logic.

## Canonical LUT Contract

V1 uses one canonical LUT domain:

```text
display-referred IEC 61966-2-1 sRGB
encoded RGB [0,1]
D65
17x17x17 grid
trilinear interpolation
residual = canonical absolute LUT - encoded-sRGB identity grid
```

Tokenizer inputs, decoded runtime LUTs, exported `.cube` files, behavior
vectors, reconstruction metrics, and eval target LUTs all use this domain.
Changing color domain, interpolation, grid order, token flatten order, codebook,
or decoder requires a new manifest and regenerated targets.

## In Scope

- Image-conditioned, instruction-guided global color grading.
- One decoded canonical 17x17x17 residual global LUT.
- Dedicated 64-token LUT vocabulary.
- VQ LUT tokenizer with frozen manifest.
- Qwen2.5-VL-3B-Instruct with QLoRA SFT.
- Generative LUT-token warmup before instruction SFT.
- Prompt attributes: warmth, tint, exposure, contrast, black point, highlights,
  shadows, saturation.
- Measurable style bundles: matte, faded, filmic, cinematic, teal-orange,
  sepia, bleach bypass, natural.
- Unsupported refusal, including mixed partial prompts.
- CLI demo with grammar-constrained decoding and versioned artifacts.
- Later workbench around preview, compare, undo, revise, and naming looks.

## Out Of Scope For V1

- Local edits.
- Object-specific recoloring.
- Subject-only or background-only changes.
- Inpainting, removal, replacement, or new content.
- Relighting.
- Geometry or camera changes.
- Texture/detail edits.
- Reference-image style transfer.
- K-LUT local mixtures.
- Companion behavior, child profiling, rankings, therapy, grading, trait praise,
  or broad creativity claims.

## Data Strategy

Collect broadly:

- PPR10K-derived expert LUTs;
- FiveK-derived expert LUTs;
- Fresh LUTs;
- G'MIC / RawTherapee HaldCLUTs;
- smaller public packs;
- controlled/procedural fillers when coverage is missing.

Train narrowly:

- active SFT set: 10k-15k examples;
- default target: 12k examples;
- held-out eval separate;
- PPR10K/FiveK capped despite high raw count;
- procedural fillers are train-only by default and never headline eval rows.

PPR10K/FiveK handling:

- derive only canonical LUT artifacts;
- reject local/non-LUT XMP tools before rendering where metadata exists;
- pair-fitted LUTs require held-out-pixel error checks, spatial residual checks,
  and per-cell support maps;
- final headline rows require `representability_tier = gold`;
- if accepted yield is low, reduce that source share rather than relax gates.

Diversity culling:

- kNN/FAISS for duplicates, leakage, and density;
- usage-prior buckets plus quota-constrained facility-location/MMR selection;
- coverage-tail budget for rare styles and outliers;
- quotas across source family, image type, LUT behavior, prompt family,
  people/non-people, and usage bucket.

## Evaluation Strategy

Build eval before training.

Free-generation eval measures learned syntax validity. Runtime constrained eval
is the CLI/product path and must reach 100% syntax validity through a token-id
grammar mask.

Primary supported metric:

```text
prompt-to-LUT pass rate on headline-eligible rows
```

Supported pass requires:

- valid grammar;
- exact token count;
- valid decoder output;
- correct direction and minimum magnitude for every gold tag;
- target-fidelity gate pass;
- safety gates pass, including skin-locus;
- style recipe and style-discriminability gates pass when applicable.

Unsupported pass requires:

- exact `<unsupported>`.

Report:

- free-generation valid-token rate;
- constrained syntax-valid rate;
- decode-valid rate;
- target-fidelity pass;
- direction accuracy by attribute;
- safety failures by type;
- unsupported recall, precision, boundary F1, mixed recall;
- over-refusal;
- false-support;
- supported coverage;
- selective risk;
- baseline deltas with paired confidence intervals;
- seed summaries.

## Training Strategy

Order:

1. Build eval harness and frozen eval rows.
2. Collect/scrape sources.
3. Build provenance registry.
4. Derive, canonicalize, and normalize LUTs.
5. Filter low-quality or non-representable global-LUT approximations.
6. Train VQ LUT tokenizer.
7. Freeze tokenizer after mean, tail, per-family, per-target, codebook, and
   roundtrip gates pass.
8. Resize vocabulary and run embedding/head preflight assertions.
9. Run generative LUT-token warmup on 30k-100k synthetic image x LUT pairs.
10. Build active 10k-15k instruction dataset and frozen eval sets.
11. Run SFT smoke tests.
12. Train Qwen2.5-VL-3B-Instruct with QLoRA.
13. Evaluate base, null, constant, deterministic, image-blind, warmup, and SFT
    baselines.
14. Run RS/DPO over scored rollouts if SFT clears gates.
15. Run GRPO only if RS/DPO plateaus and reward correctness is proven.
16. Package CLI demo.
17. Plan workbench.

SFT pass gates are CI-gated and defined in `docs/eval_harness_implementation.md`.
At a high level, SFT must pass free-generation validity, boundary metrics,
target fidelity, safety, and style gates, and beat the null/constant and
deterministic-renderer baselines.

GRPO pass gates are also CI-gated. GRPO must beat the best prior tuned stage, not
just SFT, and must not increase over-refusal or boundary failures beyond the
allowed ceiling.

## CLI First

The CLI is the first product surface.

```text
prompt_to_lut --image input.jpg --prompt "give it a warm faded film look" --out outputs/run_001
```

Artifacts:

```text
input.png
graded.png
preview_side_by_side.png
output.cube
output_tokens.txt
metrics.json
version_manifest.json
```

The CLI must:

- read embedded ICC profiles and convert inputs to canonical sRGB before LUT
  application;
- decode with the grammar-constrained token-id FSM;
- write measured deltas, not invented prompt tags;
- validate the version manifest on startup;
- produce deterministic `output_tokens.txt` and `.cube` hashes for identical
  inputs/model/profile, excluding timestamps.

This proves the model behavior and makes decoded LUT artifacts inspectable.

## Workbench Later

The later child-facing workbench adds:

- original/result comparison;
- version A/B comparison;
- undo;
- revise;
- naming the look;
- visible unsupported boundary;
- descriptive "what changed" labels;
- no rankings, trait praise, or taste grading.

The learning claim remains narrow: the workbench may help children notice,
compare, predict, explain, and revise global color changes. It does not claim
general creativity improvement without a separate study.

## Open Caveats

- Colab GPU availability can limit full Track B training speed.
- VQ tokenizer quality is a hard dependency.
- PPR10K/FiveK derived LUT yield may be lower than raw target counts because
  some edits are not global-LUT representable.
- Style recipes need empirical calibration before final eval freeze.
- Skin preservation is a LUT-domain safety audit, not a semantic editing
  guarantee.
- A deterministic recipe renderer or prompted frontier model might be a strong
  baseline; if it matches the tuned model, claims must narrow accordingly.

## Immediate Next Steps

1. Implement the provenance registry schema.
2. Implement canonical LUT parsing/normalization and quality metrics.
3. Build PPR10K and FiveK derivation scripts with representability gates.
4. Build embedding and usage-aware diversity-culling pipeline.
5. Freeze eval parser, constrained decoder, deterministic checks, and stats.
6. Train tokenizer on accepted canonical LUTs.
7. Run tokenizer roundtrip/tail/per-family diagnostics.
8. Run vocabulary resize and embedding/head preflight.
9. Run generative LUT-token warmup.
10. Build the active 12k SFT dataset.
11. Run QLoRA SFT smoke tests.
