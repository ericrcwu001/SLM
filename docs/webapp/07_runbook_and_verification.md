# Doc 07 — Runbook & Verification

**Audience:** the overnight coding agent (ChatGPT Codex + computer‑use), ONE pass. Part 1 is the exact
setup‑and‑run runbook; Part 2 is the driven browser verification with PASS/FAIL criteria and
diagnostics. Coheres with `01_architecture.md` (config §6.1, device §7, model acquisition §8),
`02_pipeline.md`, `03_backend_api.md` (server + stub flag), `05_terms.md`, `06_reference_images.md`,
`08_codex_execution_plan.md`.

**The single most important thing:** the app has a **stub mode** (`generator.stub=true`, the default)
that runs the *entire* UI + API + LUT‑apply + `.cube` export + previews **with zero model weights and
no GPU**. Bring the UI up in stub mode FIRST and run Part 2 fully in stub mode. Only then acquire the
real models and flip `generator.stub=false`. Do **not** block UI verification on model weights.

---

# Part 1 — Setup & run

## 1.0 Environment facts (verified against this repo)

- Repo root `/Users/ericwu/Developer/SLM`, branch `feat/two-stage`. `requires-python = ">=3.10"`; this
  machine has **Python 3.11**. **MPS is available** (Apple Silicon); **no CUDA**.
- The frozen VQ decoder **already ships in the working tree** at `tokenizer/final/` and **decodes on
  CPU** — verified: `eval.behavioral_fidelity.decode_codes(list(range(64)))` → a `[17,17,17,3]` LUT.
  (In a *fresh clone* the `.pt` files are gitignored → stage them, §1.6 step 4.)
- Already importable here: `torch 2.7.1`, `transformers 5.13.0`, `numpy`, `scipy`, `pillow`, and now
  `fastapi`, `uvicorn`, `peft` (installed for the webapp). **Not** present: `bitsandbytes` (CUDA-only,
  no macOS-arm64 wheel); verify `qwen_vl_utils` is installed before real-generator runs.

## 1.1 Dependency extras — CONFIRMED against `pyproject.toml`

`08_codex_execution_plan.md §0` says `pip install -e '.[sft,frontier,color]'` and to *verify the extra
names*. Verified — here is the correction:

| extra      | provides | needed by the webapp? |
|------------|----------|-----------------------|
| `sft`      | peft, qwen-vl-utils, bitsandbytes, accelerate, datasets, trl | **partly** — `peft` + `qwen-vl-utils` are needed for the **real** generator; `bitsandbytes` is **CUDA‑only** and has **no macOS‑arm64 wheel** |
| `ml`       | **torch, transformers**, safetensors, sentence-transformers, open-clip | **yes** (torch/transformers). Note torch/transformers are NOT in `sft`; they arrive transitively via peft/trl, but depend on `ml` explicitly to be safe |
| `color`    | colour-science, opencv-python-headless, scikit-image | **no** — the LUT/ΔE/color ops (`data_pipeline/lut_ops.py`, `eval/color_pipeline.py`, `eval/cube_io.py`) are pure **numpy+scipy+pillow** (base deps). Unused on the webapp path (`02_pipeline.md §2.2` confirms). |
| `frontier` | openai, anthropic | **no** — the demo uses only local models |

`.[sft,frontier,color]` *works* (torch/transformers come in transitively) but pulls unused heavy deps
and tries to install **bitsandbytes**, which fails/skips on Apple Silicon. Prefer the leaner sets below.
**FastAPI is not in `pyproject.toml`** — the webapp adds `fastapi`, `uvicorn[standard]`,
`python-multipart` (the last is required for multipart uploads). Add a `webapp` extra or install
directly:

```toml
# optional, under [project.optional-dependencies]
webapp = ["fastapi>=0.110", "uvicorn[standard]>=0.29", "python-multipart>=0.0.9"]
```

## 1.2 venv + install (pick the tier)

```bash
cd /Users/ericwu/Developer/SLM
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip

# TIER A — LOCAL (Mac/MPS or CPU): stub mode + interpreter + real decode, and the non-quantized
#   real generator (slow). No bitsandbytes (no arm64 wheel; unused off CUDA).
pip install -e '.[ml]' 'peft>=0.11' 'qwen-vl-utils>=0.0.8' \
            'fastapi>=0.110' 'uvicorn[standard]>=0.29' 'python-multipart>=0.0.9'

# TIER B — CUDA box, 4-bit real generator (adds bitsandbytes):
#   pip install -e '.[sft,ml]' 'fastapi>=0.110' 'uvicorn[standard]>=0.29' 'python-multipart>=0.0.9'
```

Pure stub mode (no real generation) needs only `.[ml]` + the three webapp packages —
`peft`/`qwen-vl-utils` are used only when `generator.stub=false`.

Sanity check:

```bash
python -c "import fastapi, uvicorn, torch, transformers; \
print('fastapi', fastapi.__version__, '| mps', torch.backends.mps.is_available())"
python -c "from eval.behavioral_fidelity import decode_codes; \
print('decode ok', decode_codes(list(range(64))).shape)"   # -> (17, 17, 17, 3)
```

## 1.3 HF token (only to DOWNLOAD the private interpreter/adapter — not needed for stub mode)

`.env` has read‑only `HF_TOKEN` (sufficient for downloads; the `SLM_Alpha_Write` token is for uploads
and NOT needed here — `01_architecture.md §8`).

```bash
export HF_TOKEN="$(grep '^HF_TOKEN=' .env | cut -d= -f2- | tr -d '\"'"'"' ')"   # or: hf auth login
```

## 1.4 Reference photos (doc 06)

Place the **6** reference photos + `references.json` in `webapp/assets/references/` per
`06_reference_images.md` (ordered: **City, Landscape, Portrait, Close‑up, Food, Interior**; the manifest
drives `previews[1..6]`; a synthetic fallback plate is provided in doc 06 if sourcing stalls). Verify:

```bash
ls -1 webapp/assets/references/*.jpg | wc -l                       # expect 6
python -c "import json;d=json.load(open('webapp/assets/references/references.json'));print(len(d['references'] if isinstance(d,dict) else d),'entries')"
```

## 1.5 Bundled test image (for Part 2)

```bash
mkdir -p webapp/assets/test
cp "output/imagegen/hairstyle-exploration/variants/07-side-part.png" webapp/assets/test/sample.png  # any RGB photo
python -c "from PIL import Image; print(Image.open('webapp/assets/test/sample.png').convert('RGB').size)"
```

## 1.6 (Real generator only) acquire the four artifacts — per `01_architecture.md §8`

A plain clone ships **no** weights. In stub mode you skip this entirely.

```bash
# 1. Interpreter router (Qwen2.5-0.5B full-FT). Model repo ericrcwu/LUT_SLM_interpreter (also interp_intensity/).
python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='ericrcwu/LUT_SLM_interpreter', allow_patterns=['interp_full/*'], \
local_dir='models/interpreter')"                       # -> models/interpreter/interp_full

# 2. Generator adapter (Qwen2.5-VL-3B QLoRA). DEFAULT = P6 two-stage (input_mode="attribute_spec_text").
#    Real subfolder on hf://ericrcwu/LUT_SLM_sft_adapters is p6_twostage_d0f9c744_smokefull (HANDOFF.md).
python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='ericrcwu/LUT_SLM_sft_adapters', allow_patterns=['p6_twostage_d0f9c744_smokefull/*'], \
local_dir='models/sft_adapters')"

# 3. Resized base (+259 token rows the adapter needs; vanilla base will NOT load the adapter).
python -m sft.vocab_resize --config configs/sft_default.yaml --out models/base_resized

# 4. Frozen VQ decoder — already in this tree at tokenizer/final/. Fresh clone only:
#    slm_stage stage --durable-root hf://datasets/ericrcwu/LUT_SLM --local-root /content/slm
#    export SLM_ARTIFACT_ROOT=/content/slm     # tokenizer.frozen.frozen_final_dir() resolves here
```

> **Adapter ↔ `input_mode` invariant** (`01_architecture.md §6.3`): `generator.input_mode` +
> `spec_bucketize` MUST match how the adapter was trained. The P6 two‑stage adapter uses
> `input_mode:"attribute_spec_text"` (`configs/candidate_two_stage.json`). If you instead point
> `adapter_path` at a one‑stage adapter (e.g. `bl_a0ccbcff_smokefull`, trained on raw text —
> `configs/candidate_one_stage_current.json`), also set `input_mode:"instruction"`. Mismatch → quality
> collapse.

## 1.7 Configure & run

Author `configs/webapp.json` per `01_architecture.md §6.1` + `03_backend_api.md §3`.
**Set `generator.stub:true` for the first bring‑up.** For the real generator:

- **Mac (MPS/CPU):** `device:"mps"` (or `"cpu"`), `load_in_4bit` ignored off CUDA (`load_generator`
  branches on device — `01_architecture.md §7`, `§6.2`), non‑quantized fp16(mps)/fp32(cpu) load
  ≈6–7 GB, best‑of‑N slow → keep `best_of_n_N:4`, `chunk:4`, and raise `server.request_timeout_s`
  (e.g. 600). Recommendation: demo on **stub**; run the **real** generator on a CUDA box.
- **CUDA box:** `device:"cuda"`, `load_in_4bit:true`, `best_of_n_N:16`.

Run (single worker — never more; each reloads multi‑GB models):

```bash
uvicorn webapp.server:app --host 127.0.0.1 --port 8000 --workers 1
# open http://127.0.0.1:8000
# force stub regardless of the file: WEBAPP_STUB=1 uvicorn webapp.server:app --port 8000 --workers 1
```

Fast API‑level smoke (before the browser):

```bash
curl -s http://127.0.0.1:8000/api/health | python -m json.tool          # expect ok:true, stub:true
curl -s http://127.0.0.1:8000/api/terms  | python -c 'import sys,json;d=json.load(sys.stdin);print(len(d),"terms; first:",d[0])'
curl -s -F "image=@webapp/assets/test/sample.png" \
        -F "prompt=make it warmer with strong teal-orange contrast" \
        http://127.0.0.1:8000/api/generate | python -m json.tool         # expect route:"grade", 7 previews, lut.cube_url
curl -s http://127.0.0.1:8000/runs/<id>/output.cube | head -3            # LUT_3D_SIZE 17 / DOMAIN_MIN / DOMAIN_MAX
```

---

# Part 2 — Computer‑use verification checklist (run in stub mode)

Confirm the end‑to‑end UX with `generator.stub=true`, so it's independent of model weights. Repeat the
grade test with `stub=false` only if a CUDA box is available.

**Pre‑checks (terminal):**

- [ ] `GET /api/health` → `{"ok":true,"stub":true, ...}`. **FAIL** → read the health `issues`/
      `load_error` and jump to Diagnostics.
- [ ] `GET /api/terms` → JSON array, length ≥ 40 (doc 05: 47 grounded + 7 style), each item has
      `term, axis, category, definition, example_usage, grounded`.

**Browser run** (launch a browser to `http://127.0.0.1:8000` — the served URL, never `file://`):

1. **Load** — page renders: upload control, prompt field, Generate button, an (empty) results grid, a
   terms/glossary panel.
   - PASS: no console errors; the glossary panel is populated from `/api/terms`.
   - FAIL: blank page / 404 on `app.js`/`styles.css` → static‑path diagnostic.

2. **Grade path** — upload `webapp/assets/test/sample.png`; prompt
   **`make it warmer with strong teal-orange contrast`**; Generate.
   - PASS (all required):
     - a. `route=="grade"`; a **Download .cube** control appears; `GET`ting its URL
       (`/runs/<id>/output.cube`) returns a file whose header is `LUT_3D_SIZE 17` with **4913** data
       rows (17³). Check: `curl -s <cube_url> | grep -cE '^[0-9.-]'` → 4913 (plus the size header line).
     - b. The results grid shows **7 tiles**: `previews[0].name=="user_image"` (your photo) graded, then
       the **6 references** (City, Landscape, Portrait, Close‑up, Food, Interior) each graded. Each tile
       shows before→after; the graded versions are **consistently** shifted (the *same* LUT on all 7).
       (In stub mode the LUT is a fixed synthetic grade — the check is consistency across all 7, not
       aesthetic quality.)
     - c. The **prompt‑feedback panel** shows an `assessment` line and suggested terms; **hovering** a
       suggested term shows its definition + example usage, matching that term's `/api/terms` entry.
       Suggested terms are grounded (`grounded:true`); a style word like *teal‑orange* may appear in the
       glossary as `grounded:false` (doc 05) and must not be offered as a grounded suggestion.
   - FAIL: no `.cube`; grid ≠ 7 tiles; a reference tile missing/broken; graded refs inconsistent;
     feedback panel empty or hover shows nothing.

3. **Clarify path** — new request; prompt **`make it pop`** (vague).
   - PASS: `route=="clarify"`; UI shows the `clarify_message` (a question); **no** LUT/grid; the
     feedback panel suggests grounded directions + an intensity word.
   - FAIL: returns a LUT/grid, or shows no clarify message.

4. **Refuse path** — new request; prompt **`remove the person`** (out of scope).
   - PASS: `route=="refuse"`, `refuse_reason∈{out_of_scope,out_of_gamut}`; UI shows a clear refusal;
     **no** LUT/grid.
   - FAIL: produces a LUT, or crashes.

5. **Robustness** — upload a non‑image (a `.txt` renamed `.png`) → friendly `bad_image` error, not a
   crash; the page stays usable.

**Overall PASS:** steps 1–4 PASS in stub mode (5 is a hardening bonus). Capture screenshots of the grade
results grid + a feedback tooltip as evidence. If a CUDA box is available, set `generator.stub:false`
(+ real models per §1.6) and re‑run step 2, additionally confirming `quality.behavioral_fidelity` is a
float and `quality.collapsed==false` on a good grade.

---

## Common‑failure diagnostics

| Symptom | Likely cause | Fix |
|---|---|---|
| `/api/health` `ok:false` mentioning **references** | `webapp/assets/references/` missing / ≠6 / no `references.json` | add the 6 photos + manifest (doc 06); `ls webapp/assets/references/*.jpg \| wc -l` == 6 |
| health mentions **tokenizer/frozen** / `FrozenTokenizerError` | `tokenizer/final/model.pt` absent (fresh clone) | §1.6 step 4 (stage + `export SLM_ARTIFACT_ROOT`); **stub mode returns a direct LUT and needs no decoder** — verify UI in stub first |
| Real mode: **OOM / process killed** | 3B VLM too big, or 4‑bit attempted off CUDA | on Mac use `stub:true`; for real use CUDA + `load_in_4bit:true`, or MPS fp16 with other apps closed and `best_of_n_N:4`/lower `max_pixels` (`128*28*28`) |
| Real mode on Mac: bitsandbytes/CUDA import error | bnb 4‑bit is CUDA‑only | `load_generator` must branch on `device` **before** importing bnb (`01_architecture.md §7` note); ensure `device:"mps"`; don't install/import bnb off CUDA |
| Blank page / 404 on `app.js`/`styles.css` | static mounted before `/api`+`/runs`, or wrong dir | mount `webapp/static` at `/` **last** (`03_backend_api.md §5`); confirm files under `webapp/static/` |
| Preview / `.cube` URLs 404 | `/runs` not mounted or `runs_dir` mismatch | ensure `runs_dir` created at startup and `app.mount("/runs", StaticFiles(directory=runs_dir))`; URLs are `/runs/<id>/…` |
| `422 Unprocessable Entity` on `/api/generate` | `python-multipart` missing, or field names wrong | install `python-multipart`; form fields must be exactly `image` (file) + `prompt` (str) |
| Overlapping requests both slow / second waits | expected — the inference lock serializes (`03_backend_api.md §6`) | fine for single‑user; raise `server.request_timeout_s`; never add workers |
| `504 generation_timeout` (real mode) | best‑of‑N on MPS/CPU is slow (`01_architecture.md §7`) | raise `server.request_timeout_s` (e.g. 600), lower `best_of_n_N`, or use CUDA |
| "CORS" errors in console | browsing from `file://` | open `http://127.0.0.1:8000` (same‑origin; do NOT add wildcard CORS) |

**Bisect rule:** if the browser misbehaves, re‑run the three §1.7 `curl` calls. Correct JSON/`.cube`
from `curl` → the fault is the frontend (`webapp/static/*`, doc 04); wrong `curl` → backend/pipeline
(docs 03/02). This cleanly separates server (doc 03) from frontend.
