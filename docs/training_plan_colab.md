# Training Plan Using Google Colab

## Objective

Train a small image-conditioned prompt-to-LUT model in Google Colab using the
full prompt-to-LUT architecture with caveats:

- full prompt-to-LUT architecture;
- active instruction SFT set around 10k-15k examples, not 50k for v1;
- generative LUT-token warmup before instruction SFT;
- CLI-first demo;
- workbench later;
- broad source collection but usage-aware, diversity-culled active training;
- QLoRA SFT first;
- RS/DPO before GRPO;
- GRPO only after simpler tuned stages pass behavior gates and plateau.

## Runtime Assumptions

Preferred Colab runtime:

```text
A100 or L4 GPU
```

T4 is acceptable for:

- dependency checks;
- tokenizer smoke tests;
- data validation;
- 50-example overfit run;
- tiny SFT smoke tests.

T4 is not the assumed runtime for full 10k-15k SFT.

## Notebook Structure

Recommended notebooks:

```text
notebooks/
  00_environment_check.ipynb
  01_lut_tokenizer_training.ipynb
  02_dataset_validation.ipynb
  03_vocab_resize_preflight.ipynb
  04_lut_token_warmup.ipynb
  05_qwen_vl_sft_qlora.ipynb
  06_eval_harness.ipynb
  07_rs_dpo.ipynb
  08_optional_grpo.ipynb
  09_cli_demo_export.ipynb
```

Keep heavy scraping and RAW conversion outside fragile notebook sessions when
possible. Colab notebooks should consume checkpointed artifacts from Drive or
Hub.

## Dependencies

Install and record exact versions:

```text
torch
torchvision
transformers
accelerate
peft
bitsandbytes
trl
datasets
qwen-vl-utils
safetensors
huggingface_hub
numpy
scipy
pandas
pyarrow
pillow
opencv-python-headless
scikit-image
colour-science
imageio
tqdm
matplotlib
wandb optional
```

Each run writes:

```text
artifacts/{run_id}/pip_freeze.txt
artifacts/{run_id}/nvidia_smi.txt
artifacts/{run_id}/train_config.yaml
artifacts/{run_id}/git_or_notebook_snapshot.txt
artifacts/{run_id}/version_manifest.json
```

## Artifact Storage

Use a stable artifact root:

```text
/content/drive/MyDrive/prompt_to_lut/
```

or a mounted Hugging Face dataset/model repo.

Required folders:

```text
data/
  raw_registry/
  splits/
  active_sft/
  warmup/
  eval/
configs/
  model_clients.yaml
luts/
  raw/
  canonical_absolute/
  canonical_residual/
tokenizer/
  checkpoints/
  final/
models/
  warmup_adapters/
  sft_adapters/
  rs_dpo_adapters/
  grpo_adapters/
eval_runs/
cli_exports/
```

All long stages must be resumable.

## Stage 0: Eval Before Training

Before SFT:

1. Freeze the detailed behavior spec.
2. Implement strict output parser.
3. Implement grammar-constrained decoder interface.
4. Implement LUT decoder interface.
5. Implement canonical color pipeline.
6. Implement deterministic direction, target-fidelity, style, skin-locus, and
   safety checks.
7. Implement Wilson CI, paired bootstrap, and seed-summary reporting.
8. Build non-gating smoke eval rows now; construct and freeze the final
   headline, diagnostic, and qualitative eval rows only after canonicalization,
   tokenizer freeze (Stage 1), split units, and usage-aware culling (Stage 2).
   Final eval rows must be frozen before warmup data materialization.
9. Run baselines on at least a smoke subset.

Minimum pre-training check:

```text
50 supported eval rows
20 unsupported eval rows
parser, constrained decoder, and metrics working end to end
```

The smoke eval is pipeline sanity only. It is not a pass/fail gate.

Full final eval target:

```text
800 supported rows
200 unsupported rows
100 qualitative/demo rows
```

## Stage 1: LUT Tokenizer Training

Inputs:

```text
canonical 17x17x17x3 residual LUT tensors
train-split quality-filtered and representability-gated candidate LUT pool
tokenizer-dev held-out LUT split
eval-reserved identities excluded
```

Configuration:

| Parameter | Starting Value |
| --- | --- |
| latent grid | 4x4x4 |
| codebook size | 256 |
| token count | 64 |
| VQ type | single-stage VQ v1 |
| batch size | 128 if memory allows |
| optimizer | AdamW |
| learning rate | 3e-4 |
| weight decay | 1e-4 |
| commitment beta | 0.25 |
| grad clip | 1.0 |

Losses:

```text
L_recon
L_deltaE
L_smooth
L_clip
L_neutral
L_commit
```

Use EMA VQ, dead-code revival, family-balanced batches, and LUT augmentation if
codebook usage or tail errors fail.

Gate:

```text
overall heldout mean DeltaE00 <= 2.0
overall heldout p95 DeltaE00 <= 4.0
overall heldout p99 DeltaE00 <= 6.0
overall heldout max DeltaE00 <= 10.0 or reviewed exception
mean PSNR >= 35 dB
p5 PSNR >= 30 dB
per-family mean DeltaE00 <= 2.5 when enough rows exist
per-family p95 DeltaE00 <= 5.0
valid finite decoded LUTs
no severe codebook collapse
active code use alert if <70% or perplexity <64
identity/ramp/flatten/.cube roundtrip tests pass
qualitative reconstructions nearly identical
```

Active code use <70% or perplexity <64 is a tokenizer-health alert, not a
standalone blocker, when reconstruction, tail, per-family, per-target, and
roundtrip gates pass. It blocks freeze only if paired with reconstruction
failure, dead-code collapse that changes token semantics, or an explicit
tokenizer-freeze exception.

Diagnostics before freeze:

- reconstruction histograms overall and by source family/style/residual bucket;
- per-target encode/decode mean, p95, max DeltaE00 and PSNR;
- rejected-row manifest with reason;
- codebook active %, perplexity, top-code share, dead-code count;
- spatial/latent error heatmaps;
- roundtrip contract tests.

If gate fails:

- inspect rejected source families;
- enable/tune EMA and dead-code revival;
- increase codebook usage diversity;
- tune commitment/reconstruction loss balance;
- remove pathological LUTs;
- try a larger code dim/codebook;
- switch to RVQ only with an explicit grammar/doc update;
- consider a different LUT grid only through a new ADR.

Do not start VLM warmup or SFT before tokenizer quality is acceptable.

## Stage 2: Active Dataset Preparation

Active SFT target:

```text
10k-15k instruction examples
default 12k
```

Dataset row contract includes:

```json
{
  "id": "train_000001",
  "image_path": "images/train/000001.jpg",
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
    "skin_locus_deltaE00_p95": 4.2
  },
  "derived_lut_quality": {
    "representability_tier": "gold",
    "fit_deltaE00_mean": 1.4,
    "fit_deltaE00_p95": 4.8,
    "supported_cell_rate": 0.99
  },
  "canonical_domain_id": "slm_lut_v1_srgb_display_encoded_17_trilinear",
  "representability_tier": "gold",
  "tokenizer_version": "...",
  "vq_codebook_sha256": "...",
  "vq_decoder_sha256": "...",
  "split_unit_id": "...",
  "split": "train"
}
```

Unsupported rows use:

```json
{
  "assistant_target": "<unsupported>",
  "target_tokens": [],
  "is_supported": false,
  "support_label": "unsupported",
  "unsupported_category": "semantic_object_recolor",
  "unsupported_components": ["semantic_object_recolor"],
  "mixed_prompt": false
}
```

Validation before training:

- every supported row has exactly 64 target tokens;
- every target token id is 0-255;
- every supported row is canonical-domain v1;
- every supported row has representability tier and tokenizer reconstruction
  status;
- every explicit tag is backed by measured behavior;
- every image path resolves;
- no train/eval leakage;
- `active_set_version`, `eval_set_version`, split manifest,
  `configs/model_clients.yaml`, and leakage report exist before warmup;
- source quotas, usage buckets, and diversity-culling report generated;
- unsupported categories are balanced;
- mixed/boundary examples are present;
- procedural fillers are train-only or diagnostic/headline-ineligible.

## Stage 3: Vocabulary Resize And Preflight

Base:

```text
Qwen/Qwen2.5-VL-3B-Instruct
```

Special tokens:

```text
<lut_bos>
<lut_eos>
<unsupported>
<lut_000> ... <lut_255>
```

Preflight assertions:

- `len(tokenizer) == input_embedding.num_embeddings == lm_head.out_features`.
- LUT code tokens are contiguous or explicitly mapped.
- Special tokens are unique and do not collide.
- Token suffix to codebook index is asserted.
- Codebook size is 256 and flatten order is recorded.
- New token rows are mean/stat-initialized from existing embeddings.
- Tied embedding/head status is detected and logged.
- Only intended rows/modules are trainable.
- Old embedding/head rows remain unchanged after a smoke optimizer step.
- Save/load roundtrip preserves tokenizer ids, resized rows, adapter weights, and
  first-batch logits within tolerance.

Prefer a row-selective embedding/head adapter or gradient mask so only the 259
new rows train.

## Stage 4A: Materialize Warmup Dataset

Create `data/warmup/{warmup_set_version}/` only after `active_set_version`,
`eval_set_version`, and split/leakage manifests are frozen.

Inputs:

```text
frozen tokenizer manifest
active/eval manifests
train-only accepted LUTs and paired input images
split/leakage manifest
```

Outputs:

```text
manifest.json
pairs.parquet
leakage_report.json
diversity_report.json
```

Materialization gate:

```text
30k-100k image x LUT pairs
every supported target has exactly 64 valid tokenizer ids
deterministic materialization seed recorded
no eval/diagnostic/qualitative image, LUT, source_pair, support_map,
  prompt-template, split-unit, or near-neighbor identity appears
source-family, behavior-vector, and token-distribution reports pass
```

## Stage 4B: Generative LUT-Token Warmup

AceTone requires a generative pretraining phase for novel LUT-token outputs. In
this project, use a cheap warmup rather than pretraining from scratch.

Warmup data:

```text
30k-100k synthetic image x LUT pairs
train-only accepted canonical LUTs applied to train-only corpus images
target = tokenizer ids for that LUT
prompt = simple global instruction or LUT-family/style phrase
```

Unsupported/refusal (`<unsupported>`) rows are optional in the warmup set:
refusal behavior is taught at SFT, not warmup, so warmup may include a small
exact-`<unsupported>` slice or omit it entirely. The warmup gate below therefore
checks exact `<unsupported>` reproduction only where such rows are included.

Warmup config must pin:

```text
trainable modules
epochs or max_steps
effective batch size
per-device batch size
gradient accumulation
learning rate per trainable module
scheduler
warmup ratio
max grad norm
max image pixels
seeds
tokenizer_version
warmup_set_version
```

Goals:

- establish the LUT-token prior;
- reduce invalid free-generation syntax before instruction SFT;
- teach the model to emit the new vocabulary distribution;
- validate tokenizer ids, flatten order, and adapter save/load behavior.

Warmup gate:

```text
50-example overfit reaches near-perfect free-generation grammar
200-example overfit reproduces supported token sequences and exact <unsupported> where included
free_generation_valid_token_rate improves materially over token baseline
no old-vocab embedding drift outside allowed tolerance
```

## Stage 5: SFT With QLoRA

Quantization:

```text
4-bit NF4
double quantization enabled
bf16 compute if available, else fp16
```

Module policy:

| Module | Policy |
| --- | --- |
| vision encoder | freeze |
| language attention/MLP projections | LoRA |
| multimodal projector | LoRA or full-trainable if memory allows |
| new token embeddings | train row-selectively |
| LM head rows for new tokens | train row-selectively |

LoRA targets:

```text
q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
```

Starting hyperparameters:

```text
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
effective_batch_size: 32
per_device_batch_size: 1 or 2
gradient_accumulation: set to reach effective batch
epochs: 2
learning_rate_lora: 2e-4
learning_rate_projector: 1e-5 if full-trainable
warmup_ratio: 0.03
scheduler: cosine
max_grad_norm: 1.0
gradient_checkpointing: true
loss_masking: assistant target only
```

Freezing the multimodal projector is a capability downgrade because the vision
encoder is already frozen. A run with frozen vision encoder plus frozen
projector cannot make an image-conditioning claim unless it independently
clears `eval_image_sensitivity`. Record vision encoder dtype/quantization and
projector policy in run config; if the vision path is quantized or frozen beyond
the default, run image-feature parity diagnostics before final SFT claims.

Checkpoint cadence:

```text
smoke_50_examples
overfit_200_examples
sft_step_500_or_1000
sft_mid
sft_final
base_vs_sft_eval
```

Smoke tests:

- 50-example overfit should reach near-perfect token syntax in free generation.
- 200-example overfit should reproduce supported tokens and unsupported outputs.
- If syntax does not improve quickly, inspect tokenizer resizing, label masking,
  collator, special-token handling, and row-selective embedding/head training
  before full training.

## Stage 6: SFT Evaluation Gate

SFT is usable only if the eval harness CI-gated pass criteria clear:

```text
free_generation_valid_token_rate lower CI >= 85%
unsupported_recall lower CI >= 80%
unsupported_precision lower CI >= 80%
boundary_f1 lower CI >= 80%
mixed_unsupported_recall lower CI >= 80%
near_boundary_pair_accuracy lower CI >= 85%
over_refusal_rate upper CI <= 10%
supported_prompt_to_lut_pass_rate lower CI >= 60%
beats best null, best constant, and deterministic renderer gates
beats deterministic renderer on renderer-hard slices as required by eval harness
beats prompt-only/image-blind, blank-image, and shuffled-image baselines on eval_image_sensitivity
```

The prompt-only/image-blind SFT baseline and the blank-image and shuffled-image ablation runs that this gate compares against are trained and scored as part of this stage, before the gate is evaluated. Stage 7 provides the deeper per-image breakdown and confirmatory analysis and is not a prerequisite for this gate.

Every gated metric must have a gating-slice registry entry from
`docs/eval_harness_implementation.md` declaring min_N/min_paired_N, MDE, CI
method, and underpowered behavior before final eval freeze.

Also report:

```text
target_fidelity_by_split
direction_accuracy_by_attribute
safety_failure_by_type
skin_locus_failure
style_discriminability
seen_family_pass
unseen_family_pass
expert_holdout_pass
usage_weighted_headline_pass
qualitative failure examples
```

Final SFT claims require 3 seeds. Smoke/dev runs may be single-seed but must be
labeled exploratory.

If SFT fails:

| Failure | Likely Cause | First Fix |
| --- | --- | --- |
| invalid token count | collator, special tokens, target formatting, weak warmup | fix data/labels; overfit small set |
| wrong direction | noisy prompt tags | strengthen deterministic tag gate |
| target mismatch | weak target-fidelity data or tokenizer tail error | filter/reconstruct targets |
| over-refusal | unsupported oversampled or ambiguous supported prompts | rebalance and add boundary examples |
| false support | unsupported examples too sparse | add/refine unsupported and mixed categories |
| unsafe LUTs | source LUT filters too loose | tighten LUT quality filters |
| weak unseen-family | source overfit | improve usage-aware culling and holdout coverage |
| image-blind parity | targets not genuinely image-conditioned | simplify model or redesign data |

## Stage 7: Image-Conditioning Ablations (Confirmatory)

The prompt-only/image-blind SFT baseline and the blank-image and shuffled-image
ablations are produced and scored in Stage 6, where they gate the SFT pass
decision. This stage is confirmatory and deeper analysis only; it is not a
prerequisite for the Stage 6 gate.

The three ablation runs are:

- prompt-only SFT: same rows and targets, no image input;
- blank-image eval: run trained VLM with a constant image;
- shuffled-image eval: pair prompts with wrong images.

They are scored against the named eval slice `eval_image_sensitivity`, where the same
instruction must produce different safe LUTs across different source images. The
targets must be image-adaptive by construction; simply applying the same global
prompt to arbitrary different images is not enough. Each group stores evidence
that the correct decoded safe LUT differs by image and that a prompt-only/common
LUT cannot pass the group on dev calibration. The
multimodal claim is gated (in Stage 6) on the hard, CI-gated image-conditioning criterion in
the eval harness: paired-bootstrap 95% lower bound for
`supported_prompt_to_lut_pass_rate` on `eval_image_sensitivity` must beat the
prompt-only/image-blind SFT baseline by >= +10pp (provisional; calibratable),
and beat the blank-image and shuffled-image ablations by a positive lower bound.
This stage re-reports that result with per-image breakdowns; if image-blind falls within that margin the Stage 6 gate fails: the VL premise is not
justified and either simplify the model or redesign data.

## Stage 8: Rejection Sampling / DPO

Run before GRPO:

1. Sample 4-8 completions per prompt from SFT.
2. Score with deterministic gates.
3. Fine-tune on winners by rejection-sampling SFT.
4. If useful but insufficient, build winner/loser pairs and run DPO.

RS-SFT and DPO config must pin:

```text
completions_per_prompt
sampling temperature/top_p/max_new_tokens
winner selection rule
DPO beta
reference policy
loss type
learning rate
batch size and gradient accumulation
epochs or max_steps
seeds
reward_config_version
```

RS/DPO ships only if it beats SFT outside paired confidence intervals without
increasing over-refusal, mixed-boundary failure, or safety failures beyond
allowed gates.

Escalate to GRPO (Stage 9) only after the best tuned stage so far (SFT, then
RS/DPO) has plateaued under the plateau rule defined in Stage 9. A stage that
still clears its ship margin has not plateaued; keep tuning that stage instead.

## Stage 9: Optional GRPO

GRPO starts only after SFT and RS/DPO pass syntax, direction, boundary, safety,
target, and baseline gates and then plateau.

Plateau is a paired-CI no-improvement rule, distinct from the ship gate (which
requires improvement):

```text
plateau(current best tuned stage) is true when a further tuning attempt cannot
clear the next stage's ship margin M on the same frozen headline rows:
  paired-bootstrap 95% CI for pass_rate(new attempt - current best tuned stage)
    upper bound < M     # cannot reach M even optimistically; includes CI containing 0
  holds across >= 2 seeds
M:
  RS/DPO after SFT   -> delta CI strictly above 0 (SFT-beating margin)
  GRPO after RS/DPO  -> +5pp vs best prior tuned stage
```

Plateau only authorizes starting the next tuned stage; it never authorizes
shipping. Shipping still requires that stage's ship gate in
`docs/eval_harness_implementation.md`.

Prompt set:

```text
1,000-3,000 prompts
4 sampled completions per prompt
```

Reward order:

1. valid syntax or exact refusal;
2. correct support boundary;
3. correct direction;
4. LUT safety;
5. target fidelity;
6. style discriminability;
7. small style/aesthetic score.

Hard-failure policy:

- Invalid syntax gets no downstream reward.
- False support on unsupported prompts gets hard penalty.
- Mixed-prompt partial support gets hard penalty.
- Wrong direction gets hard penalty.
- Safety failure gets hard penalty.
- Target mismatch gets hard penalty.
- Aesthetic score cannot compensate for any hard failure.

Reward correctness is proven only if the reward passes an adversarial /
reward-hacking test set before any GRPO run:

```text
ranking test (reward must rank genuine gold ABOVE each adversarial negative by
margin > reward_margin_min, calibrated on dev_calibration, on held-out rows):
  held-out gold output  >  constant / train-mean LUT
  held-out gold output  >  direction-only LUT (correct sign, ignores magnitude)
  held-out gold output  >  over-saturated but target-matching LUT
  held-out gold output  >  prompt-ignoring boundary-gaming output
  held-out gold output  >  degenerate repeated-token output

hard-penalty test (each hard failure must out-rank NO valid output and cannot be
recovered by the aesthetic term):
  invalid syntax                <  any valid supported output
  false support on unsupported  <  correct <unsupported>
  wrong direction               <  correct-direction output
  safety failure                <  safety-passing output
```

Record the reward-correctness report (pass/fail per row class, margins, and any
inversions) as a GRPO precondition artifact. If any class inverts, fix the
reward before running GRPO.

GRPO config must pin:

```text
reference policy
KL coefficient/schedule
learning rate
optimizer
batch size and gradient accumulation
max steps
rollout budget
group size
generation backend
decoding mode
sampling temperature/top_p/max_new_tokens
reward normalization/clipping
seeds
reward_config_version
eval config version
```

GRPO checkpoint ships only if it beats the best prior tuned stage, not just SFT,
under the CI-gated eval criteria.

Otherwise ship the best previous tuned model and record GRPO as inconclusive.

## Stage 10: CLI Demo Export

The CLI demo packages:

```text
prompt_to_lut
frozen tokenizer decoder
SFT, RS/DPO, or GRPO adapter
eval config
sample images
version manifest
```

Command:

```text
prompt_to_lut --image input.jpg --prompt "give it a warm faded film look" --out outputs/run_001
```

Artifacts:

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

`graded.png`, `preview_side_by_side.png`, and `output.cube` are supported-only
artifacts, produced only for valid token sequences. An `<unsupported>` run omits
them and writes the refusal-artifact set only — `input.png`, `output_tokens.txt`
(containing `<unsupported>`), `metrics.json` (`output.kind = "unsupported"`), and
`version_manifest.json` — with no LUT applied (see model_architecture.md "Runtime
Inference" step 6).

`metrics.json` includes:

```text
parser result
valid token count
decoded LUT validity
decoding mode and FSM version
canonical domain metadata
measured behavior deltas
direction checks only when expected attributes are supplied
target-fidelity diagnostics when target exists
clip rate
smoothness
foldover
neutral drift
skin-locus metrics
unsupported flag
tokenizer metadata
model checkpoint id
version manifest hash
```

CLI export acceptance:

- `prompt_to_lut --self-check` fails if vocab size, special-token ids, codebook
  size, `vq_codebook_sha256`, `vq_decoder_sha256`, flatten order, color
  pipeline, ICC config, `.cube` serialization, or deterministic-environment
  scope differ from the manifest.
- Same image/prompt/model/profile run twice produces identical `output_tokens.txt`
  and `.cube` hash, excluding timestamps, only under the same locked
  deterministic environment.
- Constrained runtime syntax-valid rate is 100%.
- Unsupported output writes `<unsupported>` and metrics, with no silent identity
  LUT.
- `eval_real_world_cli_inputs` runs before CLI acceptance and is reported as a
  product robustness slice.

## Final Deliverables

Project deliverables:

- active dataset manifest;
- provenance registry;
- tokenizer/decoder artifact;
- warmup adapter;
- SFT adapter;
- optional RS/DPO adapter;
- optional GRPO adapter;
- Colab notebooks;
- eval harness;
- base-vs-tuned results table with CIs;
- CLI demo;
- generated `.cube` examples;
- qualitative before/after examples;
- model card or project report;
- 3-5 minute demo video if needed.

## Schedule Crosswalk

This table is a coarse implementation phase view, not the authoritative
`Stage 0..10` numbering above.

| Phase | Work |
| --- | --- |
| 1 | source scraping, derivation, provenance registry |
| 2 | canonicalization, representability gates, quality filters, usage-aware culling |
| 3 | tokenizer training and gate |
| 4 | eval harness and frozen eval sets |
| 5 | vocab resize and embedding/head preflight |
| 6 | warmup dataset materialization and generative LUT-token warmup |
| 7 | 50/200-example SFT smoke tests |
| 8 | full 10k-15k SFT |
| 9 | base-vs-SFT eval, ablations, error analysis |
| 10 | RS/DPO |
| 11 | optional GRPO |
| 12 | CLI packaging |
| 13 | workbench planning |

The workbench begins only after CLI inference, decoding, and eval are stable.
