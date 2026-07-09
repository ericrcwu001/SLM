# GCS Durable-Root Backend For slm_stage

Status: Accepted.

ADR 0018 specified `slm_stage` (pack/stage/push) with a durable root of "Drive or Hugging Face
Hub". The next GPU training stages will also run on Google Cloud (GCE/Vertex), where a GCS
bucket is the natural durable store and Drive is unavailable. This ADR extends 0018's durable-root
abstraction with a **GCS backend**, without changing the pack/stage/push contract or the shard
format.

Decision: `data_pipeline.staging.backends` defines a small `DurableBackend` interface
(`put`/`get`/`exists`/`list`/`read_text`/`write_text`/`verify`) and a `backend_for(durable_root)`
factory that returns a `GcsBackend` when the durable root is a `gs://bucket/prefix` URI and a
`LocalBackend` otherwise. `LocalBackend` handles both plain directories and the Google Drive FUSE
mount (Drive is just a filesystem). `GcsBackend` shells out to the **`gcloud storage`** CLI
(`cp`/`ls`/`cat`; `gsutil` shares the same verbs as a fallback) rather than adding the
`google-cloud-storage` Python SDK — this keeps authentication entirely in the user's existing
`gcloud auth` context, adds no dependency, and matches the "GCP CLI integration" goal. The CLI is
invoked through an injectable `run_fn` (default `subprocess.run`) so tests exercise the backend
against a local fake without network or credentials.

Credentials are gated, not silently skipped: `GcsBackend.verify()` runs `gcloud storage ls
<bucket>` and returns `(ok, note)`; the first real transfer while unreachable raises
`RequiresManualOptIn` (the repo's existing opt-in/credentials gate in `data_pipeline/errors.py`),
directing the operator to run `gcloud auth login` and confirm the bucket. The durable root is
supplied at run time via `--durable-root` or `configs/staging_default.yaml` (`durable_root`); no
bucket/project is hardcoded.

Consequences: a Colab or GCE session can `slm_stage stage --durable-root gs://<bucket>/prompt_to_lut`
to hydrate `/content/slm` (or a VM SSD) from the same shards used for Drive, and `push`
checkpoints back to the bucket. The HuggingFace-Hub durable backend named in 0018 is also
implemented under the same interface — `HfBackend`, selected by an `hf://datasets/<user>/<repo>`
durable root, driving `huggingface_hub` (upload_file / hf_hub_download / list_repo_files) and gated
by `hf auth login` (a missing token/repo raises `RequiresManualOptIn`, mirroring GCS). Shard
format, `stage_manifest.json`, sha256 verification, disk-aware/resumable staging, and the
`SLM_ARTIFACT_ROOT` drop-in remain exactly as 0018 specified across all three backends.

Related: ADR 0018 (Colab data staging and runtime optimization), ADR 0017 (canonical LUT tokenizer
runtime contract).
