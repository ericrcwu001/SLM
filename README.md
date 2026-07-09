# Prompt-to-LUT Eval Harness (Stage 1 spine)

This package implements the **model- and decode-independent spine** of the eval harness
specified in `docs/eval_harness_implementation.md`, per master-plan Stage 1 ("build eval
before training").

The VQ LUT tokenizer is not trained until later stages, so the token→LUT `lut_decoder`
is **disabled** here. Every eval layer that needs a decoded LUT (L2 decode, L3 tokenizer
gate, L4 direction, L5 target fidelity, L6 safety, L7 style, L8 judge) is present as a
typed, guarded interface that reports `not_evaluated: decoder_disabled` — a drop-in
enable after tokenizer freeze. What runs and is tested *now*:

- **L0 boundary + L1 syntax**: strict output parser + grammar FSM (constrained-mode
  `syntax_valid_rate == 100%`).
- **Unsupported / boundary metrics**: recall, precision, false-support, over-refusal,
  coverage, selective risk, boundary accuracy, boundary F1, mixed recall, near-boundary
  pair accuracy.
- **Statistics**: Wilson CI, stratified paired bootstrap, McNemar / exact permutation,
  Holm-Bonferroni over the ship-gate family, seed summaries, `min_N` gating.
- **Baselines** that need no GPU/decoder, plus gated interfaces for the rest.
- **Reports**: the `eval_runs/{run_id}/` file set.
- A minimal **`prompt_to_lut` CLI** demonstrating the refusal path and the strict
  parse/FSM path (decode is cleanly blocked, never a silent identity LUT).

## Quick start

```bash
make install          # editable install (numpy/scipy/pandas/pyarrow/pillow/pyyaml)
make test             # pytest known-answer suite
make smoke            # generate 50/20 smoke rows, run_eval across decoder-free baselines
make cli-demo         # run the CLI on one unsupported + one supported prompt
```

## Layout

```
eval/
  vocab.py                 259 special tokens + provisional id map
  schemas.py               row / output / metric / manifest schemas + LayerResult
  output_parsers.py        strict token/refusal parser (L1)
  constrained_decoding.py  token-id grammar FSM/mask (runtime constrained mode)
  cube_io.py               canonical .cube serialization + identity grid (pure)
  unsupported_metrics.py   L0 boundary + refusal metrics
  stats.py                 Wilson / paired bootstrap / McNemar / Holm-Bonferroni
  baseline_adapters.py     decoder-free baselines + gated interfaces
  report.py                eval_runs/{run_id}/ outputs
  run_eval.py              orchestration
  lut_decoder.py           DISABLED (DecoderDisabled)
  color_pipeline.py        IMPLEMENTED (sRGB<->Lab D65, CIEDE2000) — used by data_pipeline
  deterministic_checks.py  DISABLED (L4/L6/L7)
  target_fidelity.py       DISABLED (L5)
  judge_client.py          gated by configs/model_clients.yaml (L8)
  configs/eval_default.yaml
  fixtures/make_smoke_rows.py
cli/prompt_to_lut.py
tests/
```

## Data-generation pipeline (master-plan Stages 2–9 + 11)

Acquisition-first and runnable end-to-end. Real corpora are downloaded autonomously, then
flow through a real spine: provenance registry → canonicalize → representability/quality/
behavior gates (real ΔE00) → leakage-safe splits → usage-aware selection → active/eval/
warmup manifests. Two blockers remain typed, guarded interfaces that refuse rather than
fabricate: **token materialization** (needs the frozen VQ tokenizer → `RequiresTokenizer`,
rows carry `token_status=pending_tokenizer`) and **instruction text** (needs a pinned
teacher in `configs/model_clients.yaml` → `RequiresTeacher`, `instruction_status=pending_teacher`).

Sources (`configs/source_inventory.yaml`): RawTherapee HaldCLUT (direct zip), G'MIC presets,
FiveK + PPR10K via HuggingFace mirrors, FreshLUTs (authenticated crawl — set
`SLM_FRESHLUTS_EMAIL`/`SLM_FRESHLUTS_PASSWORD`), and a local procedural generator. Downloads
are bounded + resumable by default; `slm_acquire --full` lifts sampled caps.

```bash
make acquire          # Stage 2: download bounded real corpora (network)
make data             # full pipeline over acquired + procedural LUTs (network)
make data-offline     # full pipeline over procedural LUTs only (no network)
```

```
data_pipeline/
  acquire/             Stage 2 connectors + run_acquire orchestrator
  source_inventory.py  configs/source_inventory.yaml loader
  registry.py          provenance registry (Stage 3)
  sources/             procedural generator + derive (HaldCLUT/XMP/pair-fit)  (Stage 4)
  lut_ops.py           apply / resample / HaldCLUT decode
  canonicalize.py      raw LUT -> canonical 17^3 absolute/residual            (Stage 4)
  behavior_vector.py   ~29-field measured behavior vector                     (Stage 5)
  representability.py  gold/diagnostic/rejected gate                          (Stage 5)
  quality_filters.py   safety + skin-locus gates                              (Stage 5)
  leakage.py / splits.py   near-neighbor leakage + split units                (Stage 6)
  embeddings.py / selection.py   usage-aware facility-location/MMR            (Stage 9)
  active_dataset.py    SFT rows + 12-criterion AcceptanceChecker              (Stage 9)
  eval_sets.py         frozen eval slices                                     (Stage 9)
  tokenize_targets.py  GATED (RequiresTokenizer)
  instruction_gen.py   GATED (RequiresTeacher) + tag<->behavior validation
  warmup.py            train-only pair enumeration                           (Stage 11)
  run_pipeline.py      Stage 2->3->4->5->6->9->11 orchestrator
  configs/pipeline_default.yaml
```

Deferred: training the VQ tokenizer (Stages 7–8) to enable token materialization; pinning
`configs/model_clients.yaml` to enable teacher instructions; enabling the eval color/decode
layers (L2–L8) once the tokenizer manifest is frozen.
