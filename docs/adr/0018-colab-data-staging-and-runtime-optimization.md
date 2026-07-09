# Colab Data Staging And Runtime Optimization

Status: Accepted.

Colab runtimes expose two storage layers: an ephemeral local SSD (`/content`, wiped at
session end) and a Google Drive FUSE mount where every file open is a network round-trip.
The v1 corpus is ~6.6 GB of ~8,000 loose full-range JPGs under `luts/raw/`, which is the
worst case for Drive's small-file random access. Two options were previously implicit:
re-run `make acquire` (multi-source scrape plus the authenticated FreshLUTs crawl) every
session, or read training images directly off the Drive mount. Both are slow, fragile,
and burn wall-clock and compute units while the GPU idles on I/O.

Decision one: adopt a specified `slm_stage` module for Colab data movement, following the
existing acquire/datagen conventions (`data_pipeline.staging.run_staging:main` entry
point, `data_pipeline.staging` package, Makefile targets, `configs/staging_default.yaml`).
It has three subcommands. `pack` archives the corpus directories into bounded,
structure-preserving `.tar` shards in a durable root (Drive or Hugging Face Hub) with a
`stage_manifest.json` recording per-shard member count, bytes, and sha256. `stage` copies
and extracts those shards to a local root (default `/content/slm`), verified against the
manifest, disk-aware, and resumable, so the local root becomes a drop-in
`SLM_ARTIFACT_ROOT`. `push` syncs checkpoints and outputs back to the durable root so they
survive session teardown. Implementation reuses `downloaders.sha256_file`, the disk-aware
bulk-extract/resume pattern in `acquire/fivek_kaggle.py`, `paths.artifact_paths`, and the
`AcquireLimits`/`RateLimiter` bounding helpers; it adds a `tarfile`-based packer because
`downloaders.extract_zip` flattens and cannot preserve the nested source layout. Like the
rest of the training path, the module is specified now and built at its stage; this ADR
records the design, not a completed implementation.

Decision two: adopt an explicit runtime/credit optimization policy for the GPU training
stages (warmup Stage 4B, SFT Stage 5), documented as the single source of truth in
`training_plan_colab.md` "Runtime And Credit Optimization". The sanctioned levers are:
cap image resolution / `max_pixels` (the dominant lever, since vision tokens set per-step
cost); raise per-device batch using the memory freed by the resolution cap; conditionally
disable `gradient_checkpointing` only when memory headroom exists; read data from the
`slm_stage` local root rather than the Drive mount (the I/O lever that gates the rest);
and, minor, trim LoRA target modules or `max_seq_len`. Epochs stay at 2 — reducing epochs
is deliberately excluded as a lever.

These levers are distinct from, and can point opposite to, the `model_architecture.md`
"Memory fallback order": capping resolution serves both goals, but gradient checkpointing
is kept on under memory pressure and dropped only for runtime when headroom exists. When a
run is memory-bound, the fallback order takes precedence.

Consequences: an extra durable-storage step (pack once, stage per session) replaces
per-session re-acquisition and small-file Drive reads; on A100 with 2 epochs, a single SFT
run is estimated to drop from ~2.7 h / ~35 CU to ~1 h / ~12 CU with the levers plus the
local packed data path, versus ~1.5-3 h with idle-GPU I/O waste if data still comes off
the Drive mount. Numbers are planning estimates dependent on image-token count and GPU
availability; L4/T4 are credit-comparable per job because cost is FLOPs-bound.

Related: ADR 0008 (staged VLM training scope), ADR 0009 (training sequence).
