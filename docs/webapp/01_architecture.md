# Webapp 01 — Architecture

**A LOCAL, single-machine demo of the full prompt→LUT pipeline** using the models we have now
(Stage-1 interpreter/router + Stage-2 VQ-code generator + frozen VQ-VAE decoder), built so a future
distilled/RL generator adapter drops in by editing one config file.

- **Repo:** `/Users/ericwu/Developer/SLM`, branch `feat/two-stage`.
- **Audience:** ChatGPT Codex + computer-use, ONE overnight pass. Everything below is prescriptive.
- **Companion doc:** `docs/webapp/02_pipeline.md` (the exact `webapp/pipeline.py` + `webapp/lut.py` spec).
- **Prime directive: REUSE the existing Python.** Do not reimplement generation, decoding, scoring,
  LUT-apply, or `.cube` serialization — every one already exists and is unit-tested. This webapp is a
  thin FastAPI + static-frontend shell around those functions. New code is glue only.

---

## 1. What we are building (and why this shape)

The research result that shapes the product (see `docs/interpreter_results.md`):

1. The **interpreter is production-ready as a ROUTER** — route accuracy 0.884, non-grade recall 1.0,
   refuse-kind accuracy 1.0. It reliably decides `{grade, clarify, refuse{out_of_scope,out_of_gamut}}`.
2. **Grade magnitude from vague text is weak** (direction F1 ≈ 0.47, exact ≈ 0.11), because vague text
   under-determines magnitude. It improves only when the user supplies explicit intensity words.
3. Therefore the generator **collapses under free-running greedy decoding** (exposure bias), and the
   deployable quality fix is **best-of-N sampling reranked by behavioral fidelity**
   (`eval/best_of_n.py`) — it roughly doubled free-running fidelity in the P6 gate.

Product consequences, baked into the UX:

- **Router-first.** `refuse` and `clarify` short-circuit before the generator ever runs.
- **Best-of-N is the default inference path**, not greedy.
- A **prompt-improvement feature** nudges users toward specific, *grounded* terms (from
  `eval/tag_vocabulary.py`) so their prompts carry the intensity/direction the generator can act on.
- The generator model is **config-driven** so future distilled/RL adapters swap in without code edits.

---

## 2. Component overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  BROWSER (static SPA — no build step)                                          │
│  webapp/static/{index.html, styles.css, app.js}                                │
│   • upload image + type prompt                                                 │
│   • POST /api/generate (multipart) ; GET /api/terms ; GET /api/health          │
│   • render route/refusal/clarify, previews (original|graded), prompt feedback, │
│     and a "Download .cube" link                                                │
└───────────────┬────────────────────────────────────────────────────────────── ┘
                │  HTTP (localhost:8000)
┌───────────────▼──────────────────────────────────────────────────────────────┐
│  FastAPI BACKEND — webapp/server.py                                            │
│   • routes: /api/generate, /api/terms, /api/health                             │
│   • parses multipart, saves upload, calls the pipeline, serves static + assets │
│   • one shared PromptToLutPipeline instance (models loaded ONCE at startup)    │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │  in-process Python calls
┌───────────────▼──────────────────────────────────────────────────────────────┐
│  PIPELINE — webapp/pipeline.py :: PromptToLutPipeline   (see 02_pipeline.md)   │
│                                                                                │
│   route_and_spec(prompt) ─────────────► interpreter (Qwen2.5-0.5B-Instruct FT) │
│        │  reuse: interpreter.example.build_prompt_ids                          │
│        │         data_pipeline.attribute_spec.parse / canonicalize             │
│        │                                                                       │
│        ├─ route=refuse  → return refuse_reason (NO generator call)             │
│        ├─ route=clarify → return clarify_message + suggested terms (NO gen)    │
│        └─ route=grade   → attribute_spec_text                                  │
│                                                                                │
│   generate_lut(cond_text, spec_text, image) ─► generator (Qwen2.5-VL-3B +LoRA) │
│        │  reuse: eval.best_of_n.best_of_n_codes  (best-of-N + rerank)          │
│        │         sft.generate.generate_codes     (greedy fallback, N=1)        │
│        ▼                                                                       │
│   codes[64] ─► eval.behavioral_fidelity.decode_codes ─► LUT ndarray[17,17,17,3]│
│        │                    (frozen VQ-VAE decoder, tokenizer/frozen.py)       │
│        ▼                                                                       │
│   webapp/lut.py :: apply_lut(image_rgb, lut) ─► graded image (trilinear)       │
│        │  reuse: data_pipeline.lut_ops.apply_lut_trilinear                     │
│        │  export_cube(lut, path) ── reuse: eval.cube_io.write_cube             │
│        ▼                                                                       │
│   apply to user image + 6 neutral references ; write output.cube               │
└───────────────┬──────────────────────────────────────────────────────────────┘
                │
┌───────────────▼──────────────────────────────────────────────────────────────┐
│  MODEL REGISTRY — webapp/models_config.py + configs/webapp.json                │
│   • resolves hf ids / local paths, device, best_of_n_N, sampling               │
│   • load_interpreter(cfg) / load_generator(cfg) / decoder via tokenizer/frozen │
│   • CONFIG-DRIVEN: swap the generator adapter/base by editing configs/webapp.json│
└────────────────────────────────────────────────────────────────────────────── ┘
```

Every box below the SPA is Python already in the repo; the only *new* modules are
`webapp/{server.py, pipeline.py, models_config.py, lut.py, terms.py}` and the static frontend.

---

## 3. Data-flow walkthrough of ONE request

`POST /api/generate` with `multipart/form-data`: `image` (file) + `prompt` (text).

1. **server.py** validates the parts, writes the upload to a per-request temp dir
   (`webapp/_runs/<uuid>/input.png`), opens it as a PIL RGB image, and calls
   `pipeline.run(prompt, pil_image, run_dir)`.
2. **Route.** `pipeline.route_and_spec(prompt)` tokenizes with
   `interpreter.example.build_prompt_ids(tokenizer, prompt)`, greedy-generates the one-line
   `attribute_spec_text`, and parses it with `data_pipeline.attribute_spec.parse` (guarded — see
   §2 of 02_pipeline.md). Returns `(route, refuse_reason, spec_text, clarify_message)`.
3. **Short-circuit.**
   - `route == "refuse"` → response `{route:"refuse", refuse_reason:"out_of_scope"|"out_of_gamut", …}`,
     no previews, no `.cube`. **The generator is never called.**
   - `route == "clarify"` → response `{route:"clarify", clarify_message:"…", prompt_feedback:{…}}`,
     no previews, no `.cube`.
4. **Grade → generate.** `pipeline.generate_lut(cond_text, spec_text, pil_image)`:
   - `cond_text` is chosen by `generator_input_mode` (config): the interpreter's `spec_text`
     (two-stage P6 default), the raw `prompt` (one-stage), or both (hybrid).
   - `spec_text` (canonical, for **scoring/reranking**) is always the interpreter's serialized spec.
   - Calls `eval.best_of_n.best_of_n_codes(model, processor, image=…, cond_text=…, spec_text=…, n=N, …)`
     → `(best_codes, record)`. If every sample refused (`best_codes is None`), fall back to a single
     greedy `sft.generate.generate_codes`; if that also refuses, return a soft clarify.
5. **Decode.** `codes → eval.behavioral_fidelity.decode_codes(codes)` → absolute LUT `ndarray[17,17,17,3]`
   (frozen VQ-VAE, clipped to [0,1]).
6. **Apply + export.** `webapp/lut.py`:
   - `apply_lut(np_image, lut)` (trilinear) → graded user image.
   - Same LUT applied to the **6 neutral reference images** in `webapp/assets/references/`.
   - `export_cube(lut, run_dir/"output.cube")` → a valid 17³ `.cube`.
7. **Response.** server.py returns JSON with URLs under `/runs/<uuid>/…` (a `StaticFiles` mount) and
   the parsed `attribute_spec_text`, plus `prompt_feedback` from `webapp/terms.py`.

**Latency budget (local Mac, MPS, N=4):** interpreter ~0.3–2 s; generator best-of-N dominates
(see §7); decode + apply + 7 images + `.cube` write ~0.5–2 s.

---

## 4. Directory layout

```
webapp/
  server.py            # FastAPI app: /api/generate, /api/terms, /api/health; static + runs mounts
  pipeline.py          # PromptToLutPipeline: load-once init; route_and_spec; generate_lut; run()
  models_config.py     # WebappConfig dataclass + loaders (interpreter / generator / decoder)
  lut.py               # decode (reuse), apply_lut (trilinear), export_cube (.cube)
  terms.py             # grounded tag vocabulary → /api/terms + prompt_feedback / suggested_terms
  static/
    index.html         # single page, no framework/build step required
    styles.css
    app.js             # fetch() to the API; render route/previews/feedback/download
  assets/
    references/        # 6 neutral reference images (portrait, landscape, skin-tone chart, …)
  _runs/               # gitignored: per-request output dirs (input/graded/cube) served at /runs
  README.md            # how to acquire models + run the server (see §8)
configs/
  webapp.json          # the single source of truth for model ids/paths/device/sampling (§6)
docs/webapp/
  01_architecture.md   # this file
  02_pipeline.md       # pipeline.py + lut.py spec
```

Add `webapp/_runs/` and any downloaded model dirs to `.gitignore`.

---

## 5. Tech stack + rationale

| Layer | Choice | Why (for an agent-run LOCAL demo) |
|---|---|---|
| Backend | **FastAPI + uvicorn** | The model code is Python — an in-process FastAPI app calls `pipeline.run()` directly with no serialization boundary or second language. Multipart upload, `StaticFiles`, and JSON responses are first-class. `uvicorn webapp.server:app` is a one-line start. |
| Frontend | **Static SPA** (`index.html`+`styles.css`+`app.js`, vanilla JS `fetch`) | **No build step** → nothing for the overnight agent to get wrong (no npm/webpack/vite, no node_modules, no transpile). Served directly by FastAPI `StaticFiles`. React is optional and explicitly *not required*. |
| Model load | **Once, at process start** | The 3B VLM + interpreter + VQ-VAE cost seconds–minutes to load; loading per-request is unusable. The pipeline holds them as instance attributes; `load_frozen_vqvae` is already `lru_cache`d. |
| Serving artifacts | **`StaticFiles` mount of `webapp/_runs`** | Graded PNGs and the `.cube` are written to disk and returned as `/runs/<uuid>/…` URLs — no base64 bloat in JSON, and the `.cube` link is a normal download. |

**Non-goals:** auth, multi-user concurrency, GPU queueing, persistence beyond `_runs/`. This is a
single-user local demo. Keep it that way.

---

## 6. Pluggable model registry — `webapp/models_config.py` + `configs/webapp.json`

The registry is the **only** thing that changes to swap models. Nothing in `pipeline.py`,
`server.py`, or the frontend hard-codes a model id or path.

### 6.1 `configs/webapp.json` schema

```jsonc
{
  "device": "mps",                       // "cuda" | "mps" | "cpu"  (see §7)

  "interpreter": {
    // HF repo id OR local dir with a full-FT Qwen2.5-0.5B-Instruct interpreter
    // (interpreter.config default tuning_mode = "full" → load the dir directly).
    "model_path": "models/interpreter/interp_full",
    "base_model_id": "Qwen/Qwen2.5-0.5B-Instruct",  // only used if the artifact is a LoRA adapter
    "tuning_mode": "full",               // "full" | "lora"
    "max_new_tokens": 64
  },

  "generator": {
    // The Stage-2 VQ-code generator: a Qwen2.5-VL-3B QLoRA adapter.
    "adapter_path": "models/sft_adapters/p6_two_stage",   // HF repo/subfolder or local dir
    "base_model_id": "Qwen/Qwen2.5-VL-3B-Instruct",       // the ORIGINAL base id (for reference)
    "resized_base_path": "models/base_resized",           // REQUIRED: vocab-resized base (+259 tokens)
    "input_mode": "attribute_spec_text",  // "attribute_spec_text"(P6 default) | "instruction"(one-stage) | "instruction_and_spec"
    "spec_bucketize": false,              // must match how the adapter was TRAINED
    "load_in_4bit": true,                 // honored only on device=="cuda" (bitsandbytes); ignored on mps/cpu
    "best_of_n_N": 4,                     // 16 on CUDA; small locally (§7)
    "chunk": 4,                           // num_return_sequences per generate call (peak-memory bound)
    "sampling": { "temperature": 1.0, "top_p": 0.9 },
    "max_pixels": 200704,                 // vision-token cap; lower → faster locally
    "min_pixels": 3136
  },

  "vq_decoder": {
    // Frozen VQ-VAE decoder. Defaults resolve via tokenizer.frozen.frozen_final_dir()
    // ($SLM_ARTIFACT_ROOT/tokenizer/final else repo-relative tokenizer/final).
    "final_dir": null                    // null → auto-resolve; or an explicit staged path
  }
}
```

### 6.2 `webapp/models_config.py` responsibilities

- A `WebappConfig` dataclass mirroring the JSON, with `load(path="configs/webapp.json")`.
- Three loaders (exact code in 02_pipeline.md §5):
  - `load_interpreter(cfg) -> (model, tokenizer, device)` — mirrors `interpreter/score.py::_load_model`
    (full-FT: `AutoModelForCausalLM.from_pretrained(model_path)`; LoRA: base + `PeftModel`), extended
    to honor `device ∈ {cuda,mps,cpu}` and its dtype.
  - `load_generator(cfg) -> (model, processor)` — **device-branched** (this is the key portability
    point, see §7):
    - `device=="cuda"`: reuse `sft.loader.load_eval_model(sft_cfg, resized_base_path, adapter_path)`
      (4-bit NF4 + `device_map="auto"`), where `sft_cfg = sft.config.SFTConfig()` supplies the bnb knobs.
    - `device in {"mps","cpu"}`: load `resized_base_path` as `Qwen2_5_VLForConditionalGeneration`
      in fp16 (mps) / fp32 (cpu), attach the LoRA with `PeftModel.from_pretrained`, `.to(device).eval()`.
      **No bitsandbytes** (4-bit is CUDA-only).
  - `decode` uses `eval.behavioral_fidelity.decode_codes(codes, final_dir=cfg.vq_decoder.final_dir)`
    which lazily `load_frozen_vqvae`s and caches — nothing to construct here.

### 6.3 How a FUTURE distilled/RL generator swaps in

The generator contract is fixed by the VQ grammar, not by the checkpoint: **any** model that (a) shares
the resized tokenizer (the 259 special tokens `<lut_bos>/<lut_eos>/<unsupported>/<lut_000..255>`) and
(b) emits the 64-code grammar works unchanged, because `eval/best_of_n.py` + `sft/generate.py` drive it
through `make_prefix_fn` and decode via the same frozen VQ-VAE.

To swap in a distilled or RL-tuned generator, **edit `configs/webapp.json` only**:

- **New LoRA adapter, same resized base:** set `generator.adapter_path` to the new adapter dir/repo.
  If the new adapter was trained on raw instructions, also flip `input_mode` to `"instruction"` (and
  match `spec_bucketize`). Restart the server. No Python changes.
- **A fully-merged / distilled standalone model:** point `resized_base_path` at the merged model and
  set `adapter_path` to `null`; `load_generator` skips the `PeftModel` attach when `adapter_path` is
  falsy. (Add this one `if adapter_path:` guard in the loader — the only forward-looking code.)
- **Different sampling budget for a better model:** raise `best_of_n_N` toward greedy (a strong,
  non-collapsing model needs less best-of-N); N=1 uses the greedy `generate_codes` path.

**Invariant that must hold on every swap:** `input_mode` and `spec_bucketize` MUST match how the
adapter was trained (see `sft/config.py::SFTConfig.input_field` / `spec_bucketize`), or conditioning
drifts from training and quality collapses. Document the chosen adapter's training config next to it.

---

## 7. Device & feasibility (this is a Mac / `darwin`)

The generator is a **Qwen2.5-VL-3B QLoRA** adapter. The repo's canonical inference path
(`sft/loader.py::load_eval_model`) uses **4-bit NF4 via bitsandbytes**, which requires **CUDA**.
On the user's Mac there is no CUDA, so:

| device | Generator load | RAM / notes | best_of_n_N |
|---|---|---|---|
| `cuda` | 4-bit NF4 (`load_eval_model`) + `device_map="auto"` | ~4–6 GB VRAM; fast | 16 (repo default) |
| `mps` (Apple Silicon) | **fp16/bf16** on Metal — NO bitsandbytes; load `models/base_resized` + `PeftModel` | 3B in fp16 ≈ **~6 GB weights** + vision encoder + activations → budget **≥16 GB unified memory**, comfortable at 24–32 GB. bf16 supported on recent torch MPS; fp16 is the safe default. | **4** (small) |
| `cpu` | **fp32** (fp16 matmul is slow/unsupported on CPU) | ~12 GB RAM for weights; **slow** (tens of seconds to minutes per generate) | **2** |

**Honest latency expectation (MPS, N=4, `max_pixels=200704`):** each `.generate` emits ~66 tokens
under the grammar; best-of-N with `chunk=4` is one `.generate` call returning 4 sequences. Expect
**~15–60 s** for the grade generation step on an M-series laptop; CPU is materially slower. Interpreter
(0.5B, text-only) and VQ-VAE decode are sub-second-to-few-seconds and fine on any device.

**Levers to keep the demo responsive locally:** small `best_of_n_N` (4), lower `max_pixels` (e.g.
`128*28*28 = 100352`) to cut vision tokens, and `merge_and_unload()` the adapter after load. `device`
is fully configurable in `configs/webapp.json`; default it to `"mps"` on this machine, but the app must
run on `cpu` too (slower) so it works with no GPU at all.

> Correctness note: **`load_in_4bit` is silently ignored on mps/cpu.** Do not attempt to import or call
> bitsandbytes off CUDA — `load_generator` must branch on `device` *before* touching bnb.

---

## 8. Model-acquisition plan

Four artifacts must be present locally before the server can serve a grade request. `webapp/README.md`
must script these; a plain `git clone` ships **none of the weights** (`.pt`/adapters are gitignored).

Requires an HF token. Per repo memory: `.env` `HF_TOKEN` is read-only (fine for downloads); uploads use
the separate write token. For a read-only demo, `huggingface_hub` picks up `HF_TOKEN`/`hf auth login`.

1. **Interpreter** (Qwen2.5-0.5B-Instruct full-FT router). `interp_full/` subfolder of the model repo
   `ericrcwu/LUT_SLM_interpreter` (also has `interp_intensity/`):
   ```python
   from huggingface_hub import snapshot_download
   snapshot_download(repo_id="ericrcwu/LUT_SLM_interpreter", allow_patterns=["interp_full/*"],
                     local_dir="models/interpreter")   # → models/interpreter/interp_full
   ```
2. **Generator adapter** (Qwen2.5-VL-3B QLoRA). Adapters live at
   `hf://ericrcwu/LUT_SLM_sft_adapters` (interpreter_results.md §Artifacts). Download the P6 two-stage
   adapter subfolder into `models/sft_adapters/p6_two_stage`.
   ```python
   snapshot_download(repo_id="ericrcwu/LUT_SLM_sft_adapters",
                     allow_patterns=["p6_twostage_d0f9c744_smokefull/*"], local_dir="models/sft_adapters")
   ```
3. **Resized base** (`models/base_resized`) — the base with the 259 extra token rows the adapter was
   trained against. **The adapter will NOT load correctly on the vanilla `Qwen/Qwen2.5-VL-3B-Instruct`
   base** (the `<lut_*>` token embeddings would be missing). Either download the staged copy via
   `slm_stage`, or build it locally (downloads the base, resizes, writes artifacts):
   ```bash
   python -m sft.vocab_resize --config configs/sft_default.yaml --out models/base_resized
   ```
4. **Frozen VQ-VAE decoder** (`tokenizer/final/model.pt` + `manifest.json`). These are gitignored and
   ship via HF staging. Stage the corpus, or set `SLM_ARTIFACT_ROOT` to a staged root, so
   `tokenizer.frozen.frozen_final_dir()` resolves `model.pt`:
   ```bash
   slm_stage <args>        # pull the staged tokenizer/final; see data_pipeline/staging/run_staging.py
   export SLM_ARTIFACT_ROOT=/path/to/staged_root   # frozen.py prefers $SLM_ARTIFACT_ROOT/tokenizer/final
   ```
   `load_frozen_vqvae` integrity-checks the weights against `manifest.json` (codebook/encoder/decoder
   SHA-256) and raises `FrozenTokenizerError` if absent or mismatched — a clear failure, not a silent
   identity LUT.

`webapp/server.py` should call a `pipeline.self_check()` at startup that asserts all four artifacts
resolve and fails fast with the exact missing path if not.

---

## 9. Acceptance criteria (architecture)

- [ ] `uvicorn webapp.server:app` starts, loads all models **once**, and logs the resolved
      `device`, generator `adapter_path`, and VQ-VAE `final_dir`.
- [ ] `GET /api/health` returns `{status:"ok", device, models:{interpreter, generator, vq_decoder}, ready:true}`
      only when all four artifacts (§8) resolved; otherwise `ready:false` with the missing path.
- [ ] `GET /api/terms` returns the grounded vocabulary (from `eval/tag_vocabulary.py`) grouped by axis.
- [ ] The static SPA loads at `/` with no build step (no npm/node), uploads an image + prompt, and
      renders each `route` outcome (grade previews, refuse reason, clarify message + suggested terms).
- [ ] Changing `generator.adapter_path` (and `input_mode`) in `configs/webapp.json` and restarting
      swaps the generator with **zero Python edits** — verified by pointing at a second adapter dir.
- [ ] Runs on `device:"cpu"` with no GPU present (slow but functional); `device:"mps"` on Apple Silicon.
- [ ] No model is loaded per-request; no bitsandbytes import path is reached on mps/cpu.
