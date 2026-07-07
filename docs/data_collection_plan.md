# Data Collection Plan

## Direction

The project uses a collect-broad, train-narrow strategy.

Collect and derive the full candidate corpus for personal research use, but do
not train v1 on every available row. The first active training set should be
usage-aware and diversity-culled to about 10k-15k instruction examples, with 12k
as the default planning target.

Data rights are not a gating constraint for this personal project, but
provenance is mandatory so any source family can be removed later and every
split/eval rule can be audited.

## Dataset Scale

| Dataset Layer | Target |
| --- | --- |
| Candidate registry | all scraped/derived images, LUTs, prompts, targets, metadata |
| Tokenizer LUT pool | all accepted canonical LUTs after quality and representability filters |
| Generative LUT-token warmup | 30k-100k image x LUT pairs derived from accepted LUTs |
| Active SFT set | 10k-15k instruction examples, default 12k |
| Unique image-LUT pairs | about 5k-7.5k if using two prompts per pair |
| Unsupported examples | 5%-10% of active SFT rows, plus boundary/mixed oversampling if needed |
| Held-out eval | separate; minimum 500 supported, 100 unsupported, 50-100 qualitative |

The older 50k/100k targets are scale-up milestones after the first active 12k set
proves the tokenizer, warmup, SFT, CLI, and eval loop.

## Canonical LUT Domain

All accepted LUT artifacts use:

```text
canonical_domain_id = slm_lut_v1_srgb_display_encoded_17_trilinear
display-referred IEC 61966-2-1 sRGB
encoded RGB [0,1]
D65
17x17x17 grid
trilinear interpolation
residual = canonical absolute LUT - encoded-sRGB identity grid
```

Tokenizer training consumes only canonical residual tensors. `target_lut_path`,
token targets, behavior vectors, `normalized_lut_hash`, and eval rows are all
derived from the same canonical absolute tensor.

## Source Inventory

Collect sources in this priority order:

1. PPR10K-derived expert LUTs.
2. FiveK-derived expert LUTs.
3. Fresh LUTs.
4. G'MIC / RawTherapee HaldCLUTs.
5. Smaller public LUT packs.
6. Controlled/procedural fillers only for missing attribute coverage,
   diagnostics, or warmup support.

Explicitly exclude for v1 active/headline training:

- DPED;
- HDR+ / ISP pipeline transforms;
- general camera-pipeline reconstruction datasets;
- local-edit or mask-heavy targets that fail global-LUT representability gates;
- camera-log or unknown-domain LUTs unless conversion into the canonical domain
  is defined.

## Provenance Registry

Every candidate row must be traceable and removable.

Record:

```text
source_family
source_url_or_dataset
download_timestamp
license_or_terms_snapshot
author/uploader/pack_id
image_id
input_image_id
canonical_input_image_hash
input_phash
input_embedding_id
image_split_unit_id
original_image_id
source_photo_id
ppr_group_id
group_id
expert_id
source_pack_id
lut_id
target_id
file_hash
perceptual_hash
normalized_lut_hash
canonical_domain_id
canonical_color_space
canonical_transfer
canonical_white_point
canonical_range
lut_grid_size
lut_representation
interpolation_method
axis_order
cube_table_order
token_flatten_order
raw_color_space
raw_transfer
raw_icc_profile_description
raw_icc_profile_sha256
profile_source
conversion_engine
conversion_intent
black_point_compensation
canonical_absolute_lut_hash
canonical_residual_lut_hash
tokenizer_version
codebook_hash
decoder_hash
normalization_warnings
out_of_gamut_rate_before_canonical_clip
canonical_clip_rate_from_conversion
bit_depth_pipeline
derivation_method
derivation_path
renderer_version
raw_processor_version
color_pipeline_id
raw_edit_metadata_path
xmp_hash
xmp_parser_version
xmp_global_fields_present
xmp_rejected_fields
xmp_local_tool_count
xmp_parse_status
representability_status
representability_tier
reject_reason_codes
fit_deltaE00_mean
fit_deltaE00_median
fit_deltaE00_p95
fit_deltaE00_p99
fit_deltaE00_max
fit_train_deltaE00_*
fit_validation_deltaE00_*
residual_tile_p95
residual_tile_max
residual_xy_r2
residual_moran_i
residual_edge_corr
largest_high_residual_component_pct
support_map_path
support_map_hash
supported_cell_rate
input_pixel_supported_rate
generic_input_supported_rate
quality_scores
quality_filter_version
behavior_vector_version
behavior_probe_id
structured_tags
style_bundle
unsupported_category
mixed_boundary_case
prompt_id
prompt_template_family
prompt_template_hash
teacher_model_version
teacher_prompt_version
prompt_generation_batch_id
prompt_seed
selection_bucket
usage_prior_bucket
usage_weight
selection_reason
procedural_filler
headline_eligible
used_for_tokenizer
used_for_warmup
used_for_sft
used_for_eval
diagnostic_only
source_pair_id
paired_input_image_hash
split_unit_id
split_id
active_set_version
rights_notes
```

Rights are not a blocker, but source metadata still matters for debugging,
future publication decisions, source-family ablations, and leakage prevention.

## Derived LUT Representability Gate

Representability is a row-level acceptance contract before tokenizer, SFT, or
headline eval construction.

Pipeline:

1. Parse edit metadata before rendering or fitting.
2. Hard-reject known local or non-LUT tools.
3. Render or fit only after metadata passes.
4. Fit with held-out pixels and held-out spatial tiles.
5. Analyze spatial residual maps after applying the fitted LUT.
6. Build per-cell support maps for pair-fitted LUTs.
7. Assign `representability_tier`: `gold`, `diagnostic_only`, or `rejected`.

Fit objective (all pair-fit and grid-render-validated LUTs):

```text
loss = mean pixel-wise CIEDE2000(LUT(source_pixel), target_pixel) over stratified training pixels
weighted Lab L2 (L:a:b = 1:1:1, clipped/out-of-gamut pixels downweighted) is the fallback loss if CIEDE2000 is numerically unstable
fit and evaluate in canonical sRGB (display-referred, encoded [0,1], D65)
convert source/target ICC to canonical before fitting
report train and held-out DeltaE00 separately (fit_train_deltaE00_*, fit_validation_deltaE00_*)
```

Fit-time lattice prior (shapes the solve; distinct from post-fit acceptance filters):

```text
smoothness prior: penalize large second differences across the 17x17x17 lattice so sparse cells do not overfit
monotonicity prior: keep the luma response monotonic during the fit
the post-fit smoothness and foldover_or_monotonicity_violations checks (Quality Filters) are unchanged and still applied after fitting
```

Low-support color-cell policy (per lattice cell):

```text
a cell is supported if >= 32 stratified source pixels map into it (provisional); fewer is low-support
low-support cells are filled by regularized extrapolation from supported neighbors, pulled toward the identity residual (zero residual)
filled cells are flagged low_support in the support map and do not count toward supported_cell_rate
a low-support/filled cell cannot by itself satisfy acceptance; supported_cell_rate and input_pixel_supported_rate must be met on genuinely supported cells
```

PPR10K XMP hard-reject fields include brush/paint masks, linear/radial
gradients, AI/object masks, retouch/heal/clone, red-eye, crop/geometry/
perspective, lens corrections, vignette, sharpen/denoise/texture/clarity/dehaze,
and local exposure/color corrections. Unknown XMP schemas are `diagnostic_only`,
not accepted.

Pair-fit provisional thresholds:

```text
xmp_parse_status == parsed and xmp_local_tool_count == 0 for PPR10K acceptance
fit_deltaE00_mean <= 3.0
fit_deltaE00_p95 <= 7.0
fit_deltaE00_p99 <= 10.0
fit_deltaE00_mean <= 2.0 for final eval eligibility
reject if any tile mean residual exceeds max(6.0 DeltaE00, 2.5x global mean)
reject if largest connected high-residual component covers >1% of pixels with mean residual >6.0
reject if residual_xy_r2 > 0.05
reject if abs(corr(residual, x/y/radius)) > 0.25
reject if residual_edge_corr > 0.30
input_pixel_supported_rate >= 98%
input_pixel_supported_rate >= 99% for final eval
generic_input_supported_rate >= 98% for re-paired generic images
```

For a 12k active set, each PPR10K/FiveK family should have at least 3x its target
unique accepted pairs plus eval holdout capacity. If not, reduce that source
share instead of relaxing gates.

Per-source-family derivation attrition report:

```text
funnel per source family: candidates -> XMP-parsed -> allowlist-passed -> fit-accepted -> representability=gold
store counts and rates under the registry: data/raw_registry/derivation_attrition.{csv,json}
```

The 3x-yield rule above reads from this artifact: if a family's gold yield falls
below 3x its target plus eval holdout, reduce that source's share instead of
relaxing gates.

## PPR10K Plan

Treat PPR10K as:

```text
11,161 source images
x 3 expert targets
= 33,483 candidate expert targets
```

Do not treat those 33,483 targets as independent SFT examples. They are
correlated by source image, group, portrait bias, and expert style.

Extraction:

```text
expert XMP target
        ->
parse XMP allowlist/denylist
        ->
apply accepted edit to identity Hald/grid image with recorded renderer/profile
        ->
force tagged sRGB 16-bit or float output
        ->
read transformed grid as canonical LUT
        ->
resample to 17x17x17 if needed
        ->
convert to canonical residual LUT
        ->
apply derived LUT to PPR source image and validate DeltaE to expert-target image within pair-fit thresholds (mandatory)
```

Fallback/validation:

```text
source image + expert target image
        ->
fit global 17x17x17 LUT on stratified pixels
        ->
evaluate held-out pixels and spatial tiles
        ->
run spatial residual and support-map checks
        ->
accept only if representability gates pass
```

Pair-fit fallback must not override an XMP local-tool rejection.

Grid-render source->target validation is mandatory, not fallback-only: after
reading the XMP-derived grid as a LUT, apply it to the PPR source image and
require DeltaE to the expert-target image within the same pair-fit thresholds.
XMP allowlisting alone does not accept a grid-rendered LUT; both the allowlist
and this render check must pass.

Active-set rules:

- Split by PPR10K group id first.
- All three expert variants for the same source image stay in the same split.
- Default active SFT selection uses at most one expert target per source image.
- Allow two or three expert targets from the same source only when their LUTs
  fall into meaningfully different behavior clusters.
- No single PPR10K expert should exceed 40% of active PPR examples.
- PPR10K active contribution target is 15%-20%, with 25% hard cap.
- Track portrait/person-heavy rows explicitly; portraits must not dominate the
  active set.

## FiveK Plan

Treat FiveK as:

```text
5,000 source photos
x 5 expert targets
= 25,000 candidate expert targets
```

Extraction:

```text
source image + expert target image
        ->
read source/target ICC and convert both to canonical sRGB
        ->
optimize global 17x17x17 LUT on stratified pixels
        ->
apply fitted LUT to source
        ->
evaluate held-out pixels and spatial tiles
        ->
run spatial residual and support-map checks
        ->
accept only if representability gates pass
```

Fitting uses the shared fit objective, fit-time lattice prior, and low-support
color-cell policy defined in the Derived LUT Representability Gate. FiveK is
pair-fit only (no XMP grid path); the apply-to-source render check above is its
mandatory source->target validation.

Active-set rules:

- Split by source photo id.
- All five expert targets for one source photo stay in the same split.
- Default active SFT selection uses at most one or two expert targets per source
  image.
- Additional expert targets are allowed only if behavior clusters differ.
- FiveK active contribution target is 15%-20%, with 25% hard cap.

FiveK adds scene breadth relative to PPR10K, but it still has no natural-language
prompt labels and may include edits not representable by one global LUT.

## Public LUT Sources

Fresh LUTs, G'MIC, RawTherapee HaldCLUTs, and smaller packs provide style
coverage and creative diversity.

Pipeline:

```text
download/scrape LUT or HaldCLUT
        ->
store raw file and metadata
        ->
record declared/assumed LUT color space
        ->
reject unknown-domain or camera-log LUTs unless conversion is defined
        ->
parse or convert to canonical LUT tensor
        ->
resample to 17x17x17
        ->
convert to canonical residual LUT
        ->
compute quality and behavior vector
```

Active-set targets:

| Source Family | Active Supported Example Target |
| --- | --- |
| PPR10K-derived | 15%-20%, hard cap 25% |
| FiveK-derived | 15%-20%, hard cap 25% |
| Fresh LUTs | 15%-20% |
| G'MIC / RawTherapee | 20%-25% |
| Smaller public packs | 10%-15% |
| Controlled/procedural fillers | 0%-10%, train-only by default |

Apply quotas to active examples, not raw candidates.

## Input Image Mix

The image paired with a LUT for instruction training should be diverse.

Target mix:

| Input Image Source/Type | Target |
| --- | --- |
| Broad photo images | 60%-70% |
| COCO/OpenImages-style diverse scenes | 20%-30% |
| PPR10K/FiveK source photos as model inputs | <=10%-15% |

Track and cap:

- portraits;
- children/people;
- landscapes;
- interiors;
- night scenes;
- high-key/low-key images;
- strong color casts;
- grayscale or near-monochrome scenes;
- low-quality or highly compressed images.

The goal is not equal representation across every possible scene. The goal is
that no source family, scene type, camera pipeline, or usage bucket dominates the
learned behavior.

## Quality Filters

Every candidate LUT stores:

```text
fit_error_deltaE
target_similarity_deltaE
fit_deltaE00_p95
fit_deltaE00_p99
spatial_residual_metrics
support_map_metrics
smoothness
clip_rate
pre_clamp_out_of_range
foldover_or_monotonicity_violations
neutral_drift
skin_locus_shift
residual_magnitude
source_family
expert_id_or_pack_id
```

Reject or downweight:

- high global fit error;
- spatially structured residuals;
- low support-map coverage;
- severe clipping;
- severe foldover;
- unstable or noisy LUT lattice;
- extreme neutral drift unless explicitly tagged;
- excessive residual magnitude;
- duplicates or near-duplicates;
- transformations dominated by local edits, masks, healing, sharpening,
  denoising, relighting, geometry, or content changes.

`smoothness` and `foldover_or_monotonicity_violations` above are post-fit
acceptance checks on the solved LUT, distinct from the fit-time smoothness +
monotonicity prior in the Derived LUT Representability Gate: the prior shapes the
solve, these filters accept or reject the result. Cells filled by low-support
extrapolation are flagged in the support map, downweighted, and not counted as
genuine support.

Final headline eval rows require `representability_tier = gold`.

## Post-Tokenizer Filtering

Before active SFT/eval inclusion, store per-target tokenizer reconstruction:

```text
encode_decode_mean_deltaE00
encode_decode_p95_deltaE00
encode_decode_max_deltaE00
encode_decode_psnr
tokenizer_tail_error_reason
```

Per-target SFT admission:

```text
encode_decode_mean_deltaE00 <= 3.0
encode_decode_p95_deltaE00 <= 6.0
```

Headline eval uses stricter decoded-target eligibility from the eval harness.

## Measured Behavior Vector

Each accepted LUT gets a measured behavior vector:

```text
temperature_delta_b
tint_delta_a
mean_l_delta
contrast_l_spread_delta
black_point_l_delta
highlight_l_delta
highlight_delta_a
highlight_delta_b
highlight_hue_delta_deg
highlight_chroma_delta
shadow_l_delta
shadow_b_delta
shadow_delta_a
shadow_hue_delta_deg
shadow_chroma_delta
chroma_delta
split_tone_strength
split_tone_high_hue_quadrant
split_tone_shadow_hue_quadrant
style_multi_match_count
style_margin_to_nearest_neighbor
neutral_drift_deltaE
skin_locus_deltaE00_p95
skin_locus_hue_drift_deg_p95
clip_rate
smoothness
foldover_rate
residual_norm
```

This vector is the authority for prompt tags. If a prompt says "warmer" but the
measured behavior is cooler, the row is rejected or regenerated.

## Diversity And Usage-Aware Culling

k-nearest neighbors is useful, but not sufficient as the selection algorithm.

Use kNN/FAISS for:

- exact and near-duplicate detection;
- neighborhood leakage checks;
- density scoring;
- finding overrepresented clusters.

Define a rough v0 usage prior before active culling. Suggested buckets:

- common head: mild warmth/cooling, exposure, contrast, saturation, black point;
- common style: matte, faded, natural, cinematic;
- subtle control: visible but low-magnitude adjustments;
- boundary/refusal: unsupported and mixed near-boundary prompts;
- coverage tail: rare styles, unusual scenes, strong but safe grades.

Use a real selection policy for the active dataset:

1. Apply hard quality and representability gates.
2. Assign source-family, scene, style, and usage-prior quotas.
3. Build embeddings on three axes:
   - image semantics: CLIP/SigLIP/DINOv2 embedding, plus pHash and color stats;
   - LUT behavior: residual-LUT PCA/embedding plus measured behavior vector;
   - prompt/tag semantics: structured tag vector plus text embedding.
4. Discover clusters with k-means, HDBSCAN, or Leiden.
5. Exclude HDBSCAN noise from seeding unless manually approved.
6. Allocate rows by `usage_prior_bucket`.
7. Select examples with facility-location/MMR inside buckets.
8. Reserve a bounded coverage-tail budget for rare styles/outliers.

Simple rule:

```text
kNN finds what is too close.
usage buckets decide what matters.
facility-location/MMR decides what survives inside each bucket.
```

## Instruction Generation

For each accepted image-LUT pair, generate:

- structured tags;
- one concise prompt;
- one more natural/creative prompt.

Example:

```text
tags:
  ["warmer", "muted", "lifted_blacks", "matte"]

concise:
  "Make the image warmer, more muted, and lift the blacks."

natural:
  "Give it a soft warm matte look with gentler colors."
```

Teacher output must not mention:

- local object edits;
- scene content not relevant to global color;
- impossible preservation claims;
- aesthetic rankings such as "best" or "beautiful" unless mapped to a style
  recipe and approved.

## Prompt Difficulty Mix

For the active SFT set:

| Prompt Type | Target |
| --- | --- |
| Simple explicit prompts | 40%-45% |
| Compound explicit prompts | 25%-30% |
| Style-bundle prompts | 15%-20% |
| Unsupported/refusal prompts | 5%-10% |
| Boundary and mixed unsupported prompts | oversample as needed until eval is stable |

Unsupported examples should cover:

- local region edits;
- semantic object recoloring;
- content generation/removal/replacement;
- relighting;
- geometry/detail changes;
- reference-style transfer;
- impossible selective preservation;
- supported global request plus unsupported component.

If refusal eval is weak, oversample unsupported rows during training rather than
increasing their permanent dataset share blindly.

## Splits And Leakage Rules

Create deterministic split units before active culling.

Rules:

- PPR10K: split by group id.
- FiveK: split by source photo id.
- Public LUT packs: same LUT id never crosses train/eval.
- Near-duplicate LUTs stay in the same split by normalized LUT hash and LUT
  embedding kNN.
- Same original image, crop variant, resized copy, pHash duplicate, or close
  embedding neighbor cannot cross train/eval.
- Same generic paired input image cannot cross train/eval.
- Same `derived_lut_id`, `source_pair_id`, `support_map_hash`, or
  `paired_input_image_hash` cannot cross train/eval.
- Eval prompts are generated in separate batches and template families.
- Reject near-identical train/eval prompts by text similarity and template hash.
- Before finalizing eval, remove any train row too close to eval on image
  embedding, LUT embedding, or prompt embedding.

Required eval sets:

- usage-weighted headline;
- coverage macro;
- subtle-control holdout;
- style-discriminability holdout;
- expert-id holdout;
- cross-source expert holdout;
- unseen-family/source-pack holdout;
- unsupported-prompt holdout;
- mixed/boundary holdout;
- procedural diagnostic holdout, if any procedural rows are evaluated;
- qualitative demo holdout.

Procedural filler policy:

- `procedural_filler = true` rows are train-only by default.
- If kept for eval, they go only into `eval_procedural_diagnostic`.
- Procedural diagnostic rows have `headline_eligible = false`.
- They are excluded from overall pass, supported pass, baseline deltas, and ship
  gates.

Expert holdout policy:

- Create PPR10K and FiveK expert-id holdouts where held-out `expert_id`s are
  absent from active SFT and source images/groups are also disjoint.
- Report per-expert and macro-average.
- Add a cross-distribution slice that trains mostly on filter/public/HaldCLUT
  families and evaluates on expert-derived PPR10K/FiveK LUTs.

## Active Dataset Acceptance Criteria

The active dataset is accepted only if:

1. It contains 10k-15k instruction examples, with held-out eval separate.
2. No source family, scene type, prompt family, usage bucket, or LUT-behavior
   cluster dominates.
3. No train/eval leakage exists by group id, image identity, LUT identity,
   near-neighbor embedding, prompt template, source pair, support map, or generic
   paired input image.
4. Every active row has provenance and measured LUT behavior.
5. Every supported active row has canonical-domain metadata.
6. Every supported active row has representability and tokenizer reconstruction
   status.
7. Every explicit prompt tag is backed by deterministic color checks.
8. Unsupported prompts cover all unsupported and mixed categories.
9. PPR10K/FiveK do not overwhelm the active set even though they dominate raw
   candidate counts.

## Candidate To Active Pipeline

```text
scrape/download all sources
        ->
raw provenance registry
        ->
parse/derive LUTs
        ->
canonicalize to display-referred encoded sRGB 17x17x17 absolute LUTs
        ->
convert to canonical residual LUTs
        ->
compute quality, representability, support maps, and behavior vectors
        ->
reject bad global-LUT approximations
        ->
train/freeze tokenizer
        ->
compute per-target tokenizer reconstruction quality
        ->
embed image/LUT/prompt axes
        ->
create leakage-safe split units
        ->
usage-aware quota-constrained diversity culling
        ->
teacher prompt generation
        ->
deterministic tag validation
        ->
judge language quality gate
        ->
active SFT dataset + frozen eval sets
```
