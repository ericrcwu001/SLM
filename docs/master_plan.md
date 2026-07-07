# Master Plan

## Project Thesis

Prompt-to-LUT Color Playground trains a small image-conditioned model to turn a
natural-language global color-grading instruction into one canonical global LUT,
or refuse when the request exceeds what a global LUT can do.

The project should be judged by reliable constrained behavior, not by general
image-editing ability or aesthetic preference.

## Selected Direction

Build the real image-conditioned prompt-to-LUT model with caveats.

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
  reward correctness is proven. Plateau and reward-correctness are defined
  operationally in `docs/training_plan_colab.md` Stage 9.

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
- Dedicated LUT token vocabulary: 256-entry codebook (<lut_000>-<lut_255>) plus <lut_bos>/<lut_eos>/<unsupported>; every supported output is exactly 64 code tokens.
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

1. Build eval harness, schema, parser/decoder interfaces, metrics, and
   non-gating smoke eval rows. Final frozen headline eval rows depend on
   canonicalization, tokenizer freeze, split units, culling, and prompt
   validation and are constructed and frozen at step 10.
2. Collect/scrape sources.
3. Build provenance registry.
4. Derive, canonicalize, and normalize LUTs.
5. Filter low-quality or non-representable global-LUT approximations.
6. Create leakage-safe split units and reserve eval/diagnostic/qualitative
   identities before tokenizer or warmup use.
7. Train VQ LUT tokenizer on train-split accepted LUTs only.
8. Freeze tokenizer after mean, tail, per-family, per-target, codebook, and
   roundtrip gates pass.
9. Resize vocabulary and run embedding/head preflight assertions.
10. Build active 10k-15k instruction dataset and freeze final headline,
    diagnostic, and qualitative eval sets.
11. Materialize `data/warmup/{warmup_set_version}/` from train-only accepted
    LUT/image identities after `active_set_version` and `eval_set_version` are
    frozen.
12. Run generative LUT-token warmup on the frozen train-only warmup set.
13. Run SFT smoke tests.
14. Train Qwen2.5-VL-3B-Instruct with QLoRA.
15. Re-evaluate and report base, null, constant, deterministic, image-blind,
    warmup, and SFT baselines (confirmatory; the image-conditioning ablations
    are gated at Stage 14, not here).
16. Run RS/DPO over scored rollouts if SFT clears gates.
17. Run GRPO only if RS/DPO plateaus and reward correctness is proven (plateau
    rule and reward-correctness test set in `docs/training_plan_colab.md`
    Stage 9).
18. Package CLI demo.
19. Plan workbench.

SFT pass gates are CI-gated and defined in `docs/eval_harness_implementation.md`.
At a high level, SFT must pass free-generation validity, boundary metrics,
target fidelity, safety, and style gates, and beat the null/constant and
deterministic-renderer baselines, plus the image-conditioning ablation baselines (image-blind, blank-image, shuffled-image) on `eval_image_sensitivity`.

GRPO pass gates are also CI-gated. GRPO runs only after the best prior tuned
stage has plateaued and reward correctness is proven against the adversarial
reward-hacking test set (`docs/training_plan_colab.md` Stage 9). To ship, GRPO
must beat the best prior tuned stage, not just SFT, and must not increase
over-refusal or boundary failures beyond the allowed ceiling; the exact ship
boolean is `(improvement OR safety-improvement) AND (all guardrails)`, defined in
`docs/eval_harness_implementation.md`.

## Stage Artifact Contracts

Each stage below is binding. An orchestrator may start a stage only when its
inputs exist and the prior stage's acceptance gate passed. Gate-required baseline and ablation comparisons are computed within the stage whose gate consumes them (for example, the image-blind, blank-image, and shuffled-image ablations that the SFT gate depends on are trained and scored inside Stage 14 before that gate is evaluated); later eval and reporting stages only archive and extend those comparisons and never feed an upstream gate, so no acceptance gate depends on a later stage. Paths are relative to
the artifact root in `docs/training_plan_colab.md` "Artifact Storage". Version
keys are artifact identity fields from the runtime manifest, provenance
registry, tokenizer manifest, active/eval manifests, calibration manifest, gating-slice
registry, warmup manifest, and reward config. Stage numbers follow the Training Strategy order above. Gates are
summarized here; the authoritative doc § holds the exact formulas and
thresholds.

| Stage | Inputs | Output artifact (path + version key) | Acceptance gate | Authoritative doc § |
| --- | --- | --- | --- | --- |
| 1 eval harness + smoke rows | `behavior_spec.md`, `detailed_behavior_spec.md`, Canonical LUT Contract | `eval/` modules + `eval/configs/eval_default.yaml`; smoke rows in `data/eval/` (50 supported/20 unsupported) — `eval_config_version`, `parser_version`, `fsm_version`, `safety_threshold_version` | parser, constrained decoder, and metrics run end to end on the smoke subset; pipeline sanity only, not a pass/fail gate | eval_harness_implementation.md (whole); training_plan_colab.md "Stage 0: Eval Before Training" |
| 2 collect/scrape sources | `configs/source_inventory.yaml` (source inventory priority list) | `luts/raw/`, `data/raw_registry/` raw files + metadata — `file_hash`, `source_pack_id` | excluded sources (DPED, HDR+/ISP, camera-log) rejected; raw file + metadata stored per candidate | data_collection_plan.md "Source Inventory" |
| 3 provenance registry | raw files + source metadata | `data/raw_registry/` provenance row per candidate — `canonical_domain_id`, `active_set_version` (placeholder until Stage 10) | every candidate traceable and removable; all required registry fields present | data_collection_plan.md "Provenance Registry" |
| 4 derive/canonicalize/normalize LUTs | registry rows + raw LUTs / expert XMP / paired images | `luts/canonical_absolute/`, `luts/canonical_residual/` 17x17x17 tensors — `canonical_domain_id = slm_lut_v1_srgb_display_encoded_17_trilinear`, `canonical_absolute_lut_hash`, `canonical_residual_lut_hash` | all artifacts in the canonical domain; raw color-managed before hashing/residual; `normalization_warnings` recorded | data_collection_plan.md "Canonical LUT Domain", "PPR10K Plan", "FiveK Plan", "Public LUT Sources"; model_architecture.md "Canonical LUT Domain" |
| 5 representability + quality filter | canonical residual LUTs + edit metadata + paired pixels | `representability_tier` per registry row + `support_map_path` maps + rejected-row manifest — `quality_filter_version`, `representability_tier` | XMP local-tool hard-reject; `fit_deltaE00_mean <= 3.0`, `p95 <= 7.0`, `input_pixel_supported_rate >= 98%`; headline rows require `representability_tier = gold` | data_collection_plan.md "Derived LUT Representability Gate", "Quality Filters" |
| 6 split manifest + eval reservations | accepted canonical candidates + quality/representability reports + `configs/leakage_thresholds.yaml` | `data/splits/` split manifest — `split_id`, `leakage_report_hash`, `leakage_policy_version` | deterministic split units created; eval/diagnostic/qualitative identities reserved before tokenizer or warmup use; no exact/near-neighbor leakage by image, LUT, source pair, support map, or prompt template, using the pinned thresholds in `configs/leakage_thresholds.yaml` (a `leakage_policy_version` bump forces a fresh `leakage_report.json`) | data_collection_plan.md "Splits And Leakage Rules" |
| 7 train VQ tokenizer | train-split accepted canonical residual tensors + tokenizer-dev holdout; eval-reserved identities excluded | `tokenizer/checkpoints/` candidate — `tokenizer_weights_hash` | heldout mean DeltaE00 <= 2.0, p95 <= 4.0, p99 <= 6.0; mean PSNR >= 35 dB, p5 >= 30 dB; per-family gates; active-code/perplexity alert reviewed; roundtrip tests pass | training_plan_colab.md "Stage 1: LUT Tokenizer Training"; model_architecture.md "LUT Tokenizer" |
| 8 freeze tokenizer | passing tokenizer checkpoint + diagnostics | `tokenizer/final/` frozen decoder + manifest — `tokenizer_version`, `vq_codebook_sha256`, `vq_decoder_sha256`, `flatten_order` | mean/tail/per-family/per-target/codebook/roundtrip gates all pass; frozen manifest fields recorded; per-target SFT admission mean <= 3.0, p95 <= 6.0 | model_architecture.md "LUT Tokenizer" (frozen manifest fields); training_plan_colab.md "Stage 1: LUT Tokenizer Training"; data_collection_plan.md "Post-Tokenizer Filtering" |
| 9 vocab resize + preflight | `Qwen/Qwen2.5-VL-3B-Instruct` base + frozen tokenizer manifest | resized base + preflight report — `vocab_size_after_resize`, `added_special_token_ids`, `tied_embedding_status` | `len(tokenizer) == embed rows == lm_head rows`; token-suffix→codebook index asserted; only the 259 new rows train; old rows unchanged after a smoke step; save/load roundtrip within tolerance | model_architecture.md "Vocabulary Resize And Embedding Preflight"; training_plan_colab.md "Stage 3: Vocabulary Resize And Preflight" |
| 10 active dataset + calibration + frozen eval sets | accepted gold rows + per-target tokenizer reconstruction + embeddings + split units + `dev_human_calibration` blind-rater labels + `configs/model_clients.yaml` | `data/active_sft/` (10k-15k, default 12k) + `data/eval/` frozen sets (usage-weighted headline 800 supported/200 unsupported/100 qual + additive diagnostic slices -> 1300 supported/300 unsupported total; see eval_harness_implementation.md "Eval Splits") + `data/eval/dev_human_calibration/` set + `eval/configs/calibration_manifest.json` (frozen style windows + skin-locus thresholds) + `eval/configs/gating_slice_registry.yaml` — `active_set_version`, `eval_set_version`, `style_window_version`, `skin_locus_threshold_version`, `gating_slice_registry_version` | Active Dataset Acceptance Criteria (scale, no-dominance, no-leakage, provenance + measured behavior, canonical domain, representability + tokenizer recon, tags backed by checks, unsupported coverage); style windows + skin-locus thresholds calibrated on `dev_human_calibration` then frozen (calibration gates in detailed_behavior_spec.md); every ship-gated metric has a `gating_slice_registry.yaml` entry; headline-eligibility assigned | data_collection_plan.md "Active Dataset Acceptance Criteria", "Splits And Leakage Rules"; detailed_behavior_spec.md "Human Calibration"; training_plan_colab.md "Stage 2: Active Dataset Preparation"; eval_harness_implementation.md "Eval Splits", "Initial binding registry" |
| 11 warmup dataset materialization | frozen tokenizer + split manifest + active/eval manifests + train-only accepted LUTs/images | `data/warmup/{warmup_set_version}/manifest.json` + `pairs.parquet` — `warmup_set_version`, `leakage_report_hash` | 30k-100k pairs; every supported target has 64 valid tokens; no eval/diagnostic/qualitative image, LUT, source_pair, support_map, prompt-template, split-unit, or near-neighbor identity; diversity/token reports pass | data_collection_plan.md "Warmup Data Materialization"; training_plan_colab.md "Stage 4A: Materialize Warmup Dataset" |
| 12 generative LUT-token warmup | frozen warmup set + resized base | `models/warmup_adapters/` — `adapter_id`, `adapter_sha256`, `adapter_step`, `warmup_set_version` | 50-example overfit near-perfect free-generation grammar; 200-example reproduces supported sequences + exact `<unsupported>` (where included); `free_generation_valid_token_rate` beats token baseline; no old-vocab drift | training_plan_colab.md "Stage 4B: Generative LUT-Token Warmup" |
| 13 SFT smoke tests | `data/active_sft/` 50/200 subset + warmup adapter + resized base | `models/sft_adapters/` smoke checkpoints (`smoke_50_examples`, `overfit_200_examples`) — `adapter_step` | 50-example overfit near-perfect free-gen token syntax; 200-example reproduces supported tokens + unsupported; single-seed allowed but labeled exploratory | training_plan_colab.md "Stage 5: SFT With QLoRA" (smoke tests), "Stage 0: Eval Before Training" |
| 14 QLoRA SFT | full `data/active_sft/` + warmup adapter + resized base | `models/sft_adapters/` (`sft_final`) — `adapter_id`, `adapter_sha256`, `adapter_step` | SFT Pass Criteria: free-gen valid-token lower CI >= 85%; unsupported recall/precision, boundary_f1, mixed recall lower CI >= 80%; near-boundary pair acc >= 85%; over-refusal upper CI <= 10%; supported target pass >= 60%; beats null +30pp / constant +20pp / deterministic headline +5pp and renderer-hard slice gates; the image-blind (+10pp), blank-image (> 0), and shuffled-image (> 0) ablations on `eval_image_sensitivity` are trained and scored within this stage and must clear before the gate is evaluated; 3 seeds for final claims | eval_harness_implementation.md "Pass Criteria"; training_plan_colab.md "Stage 5: SFT With QLoRA", "Stage 6: SFT Evaluation Gate", "Stage 7: Image-Conditioning Ablations" |
| 15 confirmatory baselines + ablations report | frozen eval sets + base/null/constant/deterministic/image-blind/warmup/SFT checkpoints | `eval_runs/{run_id}/` (`overall_results.csv`, `baseline_delta.csv`, `seed_summary.csv`, ...) — `eval_config_version`, `eval_set_version`, seed policy | runtime constrained `syntax_valid_rate == 100%`; full baseline and ablation deltas archived with paired CIs and min_N registry; reproduces the Stage 14 image-conditioning ablation result as confirmation — this stage is reporting/analysis and is not a prerequisite for the SFT gate | eval_harness_implementation.md "Baselines", "Statistics", "Reports"; training_plan_colab.md "Stage 7: Image-Conditioning Ablations" |
| 16 RS/DPO | passing SFT checkpoint + scored rollouts (4-8 completions/prompt) | `models/rs_dpo_adapters/` — `adapter_id`, `adapter_sha256`, `adapter_step`, `reward_config_version` | ships only if it beats SFT outside paired CI without increasing over-refusal, mixed-boundary failure, or safety failures beyond allowed gates; RS/DPO hyperparameters pinned | training_plan_colab.md "Stage 8: Rejection Sampling / DPO"; model_architecture.md "Rollout Optimization Architecture" |
| 17 GRPO | best prior tuned checkpoint (plateaued) + 1k-3k prompts x4 completions | `models/grpo_adapters/` — `adapter_id`, `adapter_sha256`, `reward_config_version`, `eval_config_version` | GRPO ships only if paired-boot 95% lower bound pass_rate(GRPO - best prior tuned) >= +5pp, or safety upper bound <= -5pp; over-refusal <= +2pp; mixed recall and near-boundary pair acc drop <= 2pp; multi-seed confirms | eval_harness_implementation.md "Pass Criteria" (GRPO ships only if); training_plan_colab.md "Stage 9: Optional GRPO"; model_architecture.md "Rollout Optimization Architecture" |
| 18 package CLI demo | frozen tokenizer decoder + best tuned adapter + eval config + sample images + manifest | `cli_exports/` `prompt_to_lut` bundle + `version_manifest.json` — full manifest key set | `prompt_to_lut --self-check` matches manifest (vocab size, special-token ids, `vq_codebook_sha256`, `vq_decoder_sha256`, flatten order, color pipeline, ICC config, `.cube` serialization); repeat run gives identical `output_tokens.txt` + `.cube` hash only under same locked environment; constrained syntax-valid 100%; unsupported writes `<unsupported>`, no silent identity LUT; `eval_real_world_cli_inputs` reported | training_plan_colab.md "Stage 10: CLI Demo Export"; model_architecture.md "Runtime Inference", "Version Manifest And Startup Assertions" |
| 19 plan workbench | stable CLI inference/decoding/eval | workbench plan (no new model artifact; reuses shipped `version_manifest.json`) | begins only after CLI inference, decoding, and eval are stable | master_plan.md "Workbench Later"; model_architecture.md "Workbench Extension" |

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

`graded.png`, `preview_side_by_side.png`, and `output.cube` are supported-only:
they are written only for valid token sequences. An `<unsupported>` run omits
them and writes the refusal-artifact set only — `input.png`, `output_tokens.txt`
(containing `<unsupported>`), `metrics.json` (`output.kind = "unsupported"`), and
`version_manifest.json` — and applies no LUT (see model_architecture.md "Runtime
Inference" step 6; detailed_behavior_spec.md).

The CLI must:

- read embedded ICC profiles and convert inputs to canonical sRGB before LUT
  application;
- decode with the grammar-constrained token-id FSM;
- write measured deltas, not invented prompt tags;
- validate the version manifest on startup;
- produce deterministic `output_tokens.txt` and `.cube` hashes for identical
  inputs/model/profile under the same locked deterministic environment,
  excluding timestamps.

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

- Colab GPU availability can limit full prompt-to-LUT training speed.
- VQ tokenizer quality is a hard dependency.
- PPR10K/FiveK derived LUT yield may be lower than raw target counts because
  some edits are not global-LUT representable.
- Style recipes need empirical calibration before final eval freeze.
- Skin preservation is a LUT-domain safety audit, not a semantic editing
  guarantee.
- A deterministic recipe renderer or prompted frontier model might be a strong
  baseline; if it matches the tuned model, claims must narrow accordingly.

## Immediate Next Steps

This list is a non-authoritative orientation aid, not the binding execution
order. The **Stage Artifact Contracts** table above ("Each stage below is
binding") is the sole binding stage order and readiness contract, with the
per-stage inputs, outputs, version keys, gates, and authoritative-doc pointers;
where this list's ordering diverges from that table, the table governs.

1. Implement the provenance registry schema.
2. Implement canonical LUT parsing/normalization and quality metrics.
3. Build PPR10K and FiveK derivation scripts with representability gates.
4. Build embedding and usage-aware diversity-culling pipeline.
5. Freeze eval parser, constrained decoder, deterministic checks, and stats.
6. Create split units and reserve eval/diagnostic/qualitative identities.
7. Train tokenizer on train-split accepted canonical LUTs.
8. Run tokenizer roundtrip/tail/per-family diagnostics.
9. Run vocabulary resize and embedding/head preflight.
10. Build the active 12k SFT dataset and freeze eval sets.
11. Materialize train-only warmup data, then run generative LUT-token warmup.
12. Run QLoRA SFT smoke tests.
