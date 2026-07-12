# Prompt-to-LUT local demo

This FastAPI app turns an uploaded image and a color-grading prompt into a 17×17×17 `.cube` LUT. It previews the same LUT on the upload and six reference photographs, exposes a grounded prompt glossary, and renders `grade`, `clarify`, and `refuse` routes in a no-build static SPA.

The default configuration uses a synthetic generator, so the complete API, LUT export, preview, and UI workflow runs without model weights. Real mode reuses the repository's interpreter, best-of-N generator, and frozen VQ decoder; those modules are imported and are not modified by the webapp.

## Install

Run all commands from the repository root with Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e '.[ml,webapp,dev]'
```

For real inference on macOS/MPS or CPU, also install the non-quantized inference dependencies:

```bash
pip install 'peft>=0.11' 'qwen-vl-utils>=0.0.8'
```

Do not install `bitsandbytes` on Apple Silicon. On a CUDA host, `pip install -e '.[sft,ml,webapp,dev]'` provides the 4-bit path.

## Run in stub mode

`configs/webapp.json` is stub-safe by default. Start exactly one worker:

```bash
WEBAPP_STUB=1 uvicorn webapp.server:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

Open <http://127.0.0.1:8000>. Do not open `index.html` through `file://`.

Inference is guarded by one process-local lock. Multiple workers, `--reload`, or multiple server processes would each load their own multi-GB models and defeat serialization, so they are unsupported.

## Configuration and real models

The registry is [`configs/webapp.json`](../configs/webapp.json). Its main controls are:

- `device`: `mps`, `cuda`, or `cpu`.
- `interpreter`: local full-model/LoRA path and decode limit.
- `generator`: stub flag, base/adapter paths, input mode, best-of-N settings, sampling, and image limits.
- `vq_decoder.final_dir`: frozen decoder directory, or `null` for repository auto-discovery.
- `server`: static, reference, and run directories plus upload, image, and timeout limits.

Environment overrides:

- `WEBAPP_CONFIG=/path/to/config.json` selects another config.
- `WEBAPP_STUB=1` forces stub mode; `WEBAPP_STUB=0` forces real mode.
- `HF_TOKEN` authenticates private Hugging Face downloads only.
- `SLM_ARTIFACT_ROOT` may point frozen-decoder discovery at an externally staged artifact root.

Acquire the real artifacts once:

```bash
export HF_TOKEN='your-read-token'

python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ericrcwu/LUT_SLM_interpreter', allow_patterns=['interp_full/*'], local_dir='models/interpreter')"
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ericrcwu/LUT_SLM_sft_adapters', allow_patterns=['p6_twostage_d0f9c744_smokefull/*'], local_dir='models/sft_adapters')"
python -m sft.vocab_resize --config configs/sft_default.yaml --out models/base_resized
```

Run `sft.vocab_resize` only once when `models/base_resized` is absent, then reuse that directory;
never overwrite or regenerate it between evaluations. As a disk-saving local fallback, download
`Qwen/Qwen2.5-VL-3B-Instruct` into that path instead. The real loader installs the adapter's
authoritative 151,924-token tokenizer and resizes the vanilla table in memory before attaching PEFT.

The download above places the router in `models/interpreter/interp_full`, matching the default config's `interpreter.model_path` (subfolder `interp_full/` on `ericrcwu/LUT_SLM_interpreter`, per `docs/webapp/07_runbook_and_verification.md` §1.6). If your local checkout has it under a different subfolder, point `interpreter.model_path` at that directory instead.

Real mode also requires `tokenizer/final/model.pt` (and its manifest) or an equivalent staged decoder selected by `vq_decoder.final_dir`/`SLM_ARTIFACT_ROOT`. Keep the P6 adapter paired with `input_mode: "attribute_spec_text"` and `spec_bucketize: false`.

Then launch:

```bash
WEBAPP_STUB=0 uvicorn webapp.server:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

On a memory-constrained Mac, start with `best_of_n_N: 1`, `chunk: 1`, `max_pixels: 100352`, and `request_timeout_s: 600` in a separate config. CUDA can use 4-bit loading and a larger best-of-N. The health endpoint reports load failures without hiding the SPA.

## Tests and API smoke checks

Run the focused suite:

```bash
pytest -q \
  tests/test_webapp_lut.py \
  tests/webapp/test_terms.py \
  tests/test_webapp_pipeline.py \
  tests/test_webapp_models_config.py \
  tests/test_webapp_server.py
```

With the stub server running:

```bash
curl -fsS http://127.0.0.1:8000/api/health | python -m json.tool
curl -fsS http://127.0.0.1:8000/api/terms | python -m json.tool
curl -fsS \
  -F 'image=@webapp/assets/references/portrait.jpg' \
  -F 'prompt=make it warmer with strong teal-orange contrast' \
  http://127.0.0.1:8000/api/generate | python -m json.tool
```

Health should report `ok: true`, `stub: true`, and six references. A grade response should contain seven previews and `/runs/<request-id>/output.cube`. Download that URL and confirm `LUT_3D_SIZE 17` plus 4,913 numeric rows.

## Browser verification

Follow [`docs/webapp/07_runbook_and_verification.md`](../docs/webapp/07_runbook_and_verification.md) end to end. At minimum, verify:

1. Grade: `make it warmer with strong teal-orange contrast` produces the user preview, six reference previews, feedback tooltips, and a valid downloadable cube.
2. Clarify: `make it pop` asks for direction and intensity and produces no LUT.
3. Refuse: `remove the person` shows an in-scope explanation and produces no LUT.
4. Robustness: an invalid image returns a friendly `bad_image` response; the browser console remains clean; the layout has no horizontal overflow at desktop, tablet, and phone widths.
5. Real mode: repeat the grade request and confirm `stub: false`, a numeric `quality.behavioral_fidelity`, and `quality.collapsed: false` when the model produces a sound candidate.

Browser evidence is stored under `webapp/_runs/verification/`.

## Outputs and provenance

Every grade request writes an isolated directory under `webapp/_runs/<request-id>/` containing the normalized upload, `output.cube`, and original/graded PNGs for the upload and references. These are generated local artifacts and can be deleted when the server is stopped.

The six bundled reference images and their photographer, source URL, and license are recorded in [`assets/references/references.json`](assets/references/references.json). They come from Unsplash and Pexels under their respective licenses; preserve that manifest when redistributing the images. Selection and normalization details are in [`docs/webapp/06_reference_images.md`](../docs/webapp/06_reference_images.md). Model provenance is defined by the Hugging Face repository IDs and local adapter manifest referenced by the active config.
