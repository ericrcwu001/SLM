# Doc 03 — Backend API (`webapp/server.py`)

**Audience:** the overnight coding agent. This is the concrete FastAPI spec for `webapp/server.py`:
app setup, one‑time model load, static + per‑request artifact serving, request **serialization**
(single inference lock), timeouts, multipart image handling, Pydantic response models, structured
errors, and per‑endpoint acceptance criteria.

This doc is **downstream of and coheres with** the sibling docs — it does not redefine them:

- **`01_architecture.md`** — the canonical `configs/webapp.json` schema (§6.1), the `webapp/_runs`
  → `/runs` serving convention, and device/feasibility. **The config schema is owned there; this doc
  adds only the server‑runtime knobs (§3).**
- **`02_pipeline.md`** — `webapp/pipeline.py::PromptToLutPipeline` (`__init__(cfg)`, `run(prompt,
  image, run_dir) -> dict`, `route_and_spec`, `generate_lut`, `self_check`, `.terms`), `webapp/lut.py`
  (`load_image`/`save_image`/`apply_lut`/`export_cube`), `webapp/models_config.py`
  (`WebappConfig.load`, `load_interpreter`, `load_generator`). **The server calls this surface; it
  does not reimplement pipeline logic.**
- **`05_terms.md`** — `webapp/terms.py::all_terms()` and `prompt_feedback(prompt, route_result)`.
- **`06_reference_images.md`** — the 6 reference photos + `references.json` (drives `previews[1..6]`).
- **`07_runbook_and_verification.md`** — install, model acquisition, run, and driven verification.

**Research context (why the API shape is what it is):** the interpreter is a **router**
(`grade`/`clarify`/`refuse`) — routing is production‑ready (route acc 0.884; refuse+clarify recall
1.0) but grade *magnitude* is weak, so intensity comes from the **generator run best‑of‑N** and
reranked by behavioral fidelity, plus a **prompt‑improvement panel** that steers users toward
grounded direction+intensity vocabulary. See `docs/interpreter_results.md` and
`docs/collapse_fix/02_best_of_n_reranking.md`. The API always returns a `route`; on `grade` it returns
a LUT + 7 graded previews + grounded suggested terms; on `clarify`/`refuse` it returns early with a
message and still returns `prompt_feedback`.

---

## 1. Design constraints (pin these)

1. **Local, single user, single worker.** `uvicorn webapp.server:app --workers 1`. Never add workers —
   each reloads multi‑GB models.
2. **Load the pipeline exactly once** at startup into a module global. Every request reuses it.
3. **Serialize inference.** One GPU/CPU, not concurrent‑safe → a single `asyncio.Lock` forces one
   `pipeline.run()` at a time; overlapping HTTP requests queue. (This is the "single inference lock" in
   `08_codex_execution_plan.md §1`. Multi‑user concurrency remains an explicit non‑goal per
   `01_architecture.md §5`.)
4. **Never block the event loop.** `pipeline.run()` is synchronous and CPU/GPU‑bound → run it in a
   threadpool (`run_in_threadpool`) *while holding the lock*.
5. **Artifacts are files under `webapp/_runs/<uuid>/`**, served at `/runs/<uuid>/…` (doc 01 §5). No
   base64 in JSON; the `.cube` is a normal downloadable file.
6. **Structured errors** (§8) so the frontend can distinguish model‑not‑loaded / bad‑image /
   generation‑failure / timeout.

The pipeline (doc 02) **writes its own artifacts into `run_dir` and returns the full API payload dict**
already carrying `/runs/<uuid>/…` URLs. The server's job is therefore thin: validate the upload, mint
`run_dir`, save `input.png`, call `pipeline.run(prompt, pil, run_dir)`, and return the dict (validated
through the Pydantic model). **Do not move artifact writing into the server** — that lives in
`pipeline.run` / `webapp/lut.py`.

---

## 2. Pipeline surface the server depends on (from doc 02 — restated, not redefined)

```python
# webapp/pipeline.py  (doc 02 is the authority; this is the exact surface server.py touches)
class PromptToLutPipeline:
    def __init__(self, cfg: "WebappConfig"): ...        # loads interpreter + generator(or stub) + decoder ONCE
    def self_check(self) -> dict: ...                    # health: assert all artifacts resolve
    def run(self, prompt: str, image, run_dir: str) -> dict: ...   # full flow → API payload; writes run_dir
    terms: "TermsModule"                                 # .all_terms(); .prompt_feedback(prompt, route_result)
```

`run()` returns exactly the canonical contract dict (doc 02 §1.6):

```python
{"route": "grade|clarify|refuse",
 "refuse_reason": "out_of_scope|out_of_gamut|null",
 "clarify_message": "…|null",
 "attribute_spec_text": "route=grade | warmer=+2.3 muted=+2.0 …|null",  # interpreter's serialized spec
 "lut": {"cube_url": "/runs/<id>/output.cube"} | None,
 "previews": [{"name": "user_image", "original_url": "…", "graded_url": "…"}, …],  # [user, then 6 refs]
 "prompt_feedback": {"assessment": "…",
    "suggested_terms": [{"term","axis","definition","example_usage","grounded"}]},
 "quality": {"behavioral_fidelity": float|None, "collapsed": bool, "fell_back_greedy": bool}}  # grade only
}
```

On `clarify`/`refuse`, `lut` is `null` and `previews` is `[]` (generator not called).

---

## 3. `configs/webapp.json` — server‑runtime additions only

The **model/device/sampling schema is owned by `01_architecture.md §6.1`** (`device`, `interpreter{…}`,
`generator{adapter_path, base_model_id, resized_base_path, input_mode, spec_bucketize, load_in_4bit,
best_of_n_N, chunk, sampling, max_pixels, min_pixels}`, `vq_decoder{final_dir}`). Do not fork it.

This doc pins two things the server needs that doc 01 leaves open:

**(a) Stub toggle** (the "STUB_GENERATOR mode" of `08_codex_execution_plan.md`). Add under `generator`:

```jsonc
"generator": {
  "stub": true,        // NEW: true → load_generator returns a stub that yields a fixed synthetic LUT.
                       //      Default true so the app boots + the whole UI is verifiable with NO weights.
  ... // all other generator fields per 01_architecture §6.1
}
```

Env override: `WEBAPP_STUB=1` forces stub on, `WEBAPP_STUB=0` forces it off (wins over the file). The
stub path must require **no** generator weights, **no** bitsandbytes, and — ideally — **no** frozen
decoder (return a directly‑built absolute LUT: identity + a small fixed residual), so UI verification
is fully decoupled from model acquisition (doc 07 Part 2 runs entirely in stub mode).

**(b) Optional `server` block** (all have safe defaults; omit to accept defaults):

```jsonc
"server": {
  "runs_dir": "webapp/_runs",     // per-request artifact dirs (gitignored), mounted at /runs
  "static_dir": "webapp/static",  // SPA, mounted at /
  "max_upload_mb": 20,            // reject larger uploads (413)
  "max_image_edge": 2048,         // downscale longest edge before grading (speed/memory)
  "request_timeout_s": 300        // per-request wall clock; raise for slow MPS/CPU real mode
}
```

`WebappConfig.load()` (doc 02 §6.2) should surface `cfg.generator.stub` and a `cfg.server.*` group
(with the defaults above) alongside the doc‑01 fields.

---

## 4. Files this doc owns

```
webapp/server.py         # this doc
webapp/_runs/            # created at runtime; gitignored; per-request outputs served at /runs
```

`configs/webapp.json`, `webapp/pipeline.py`, `webapp/models_config.py`, `webapp/lut.py`,
`webapp/terms.py`, `webapp/static/*`, `webapp/assets/references/*` are other docs' files. Add
`webapp/_runs/` (and downloaded model dirs) to `.gitignore` (doc 01 §5 already requires this).

---

## 5. App setup, one‑time load, static + `/runs` serving

- `app = FastAPI(title="prompt→LUT demo", version="1")`.
- **Startup (lifespan preferred; `@app.on_event("startup")` acceptable):**
  1. `cfg = WebappConfig.load(os.environ.get("WEBAPP_CONFIG", "configs/webapp.json"))`.
  2. `Path(cfg.server.runs_dir).mkdir(parents=True, exist_ok=True)`.
  3. `try: STATE.pipeline = PromptToLutPipeline(cfg)` — heavy; loads all models once. `except Exception
     as e:` log the traceback, set `STATE.pipeline=None`, `STATE.load_error=str(e)` so `/api/health`
     reports it instead of crashing the process (the SPA still loads and shows the error).
- **Mounts** (order matters — SPA `/` mounts **last** so it never shadows `/api` or `/runs`):
  - `app.mount("/runs", StaticFiles(directory=cfg.server.runs_dir), name="runs")`
  - `app.mount("/", StaticFiles(directory=cfg.server.static_dir, html=True), name="static")`
    (`html=True` serves `index.html` at `/`).
- **CORS:** none. The SPA is served by this same app (same‑origin). Do not add wildcard CORS. Always
  browse the served `http://127.0.0.1:8000`, never `file://`.

---

## 6. Concurrency, timeout, image handling

```python
import asyncio
from fastapi.concurrency import run_in_threadpool

INFERENCE_LOCK = asyncio.Lock()   # serialize all model calls (one device, not concurrent)

async def _run_pipeline(prompt, pil, run_dir, timeout_s):
    async with INFERENCE_LOCK:                                  # queue overlapping requests
        return await asyncio.wait_for(
            run_in_threadpool(STATE.pipeline.run, prompt, pil, run_dir),  # blocking → thread
            timeout=timeout_s)
```

- On `asyncio.TimeoutError` → `generation_timeout` (HTTP 504). The worker thread can't be force‑killed;
  for a single‑user demo it simply holds the lock until it finishes. Set `request_timeout_s` generously
  (best‑of‑N on MPS/CPU is slow — doc 01 §7: ~15–60 s on MPS at N=4).
- **Image handling** (before locking): read the upload fully; reject `> max_upload_mb` (413). Save the
  raw bytes to `run_dir/input.png` after decoding, or decode from memory. Decode with
  `webapp.lut.load_image` semantics (`Image.open(BytesIO(raw)); .convert("RGB")`); on
  `UnidentifiedImageError`/any decode error → `bad_image` (400). **Downscale** the longest edge to
  `max_image_edge` (`im.thumbnail((edge, edge), Image.LANCZOS)`) to bound memory and keep trilinear
  apply fast; re‑encode to strip EXIF. Pass the PIL RGB image to `pipeline.run`.

---

## 7. Endpoints

### 7.1 `GET /api/health` → `pipeline.self_check()`

Liveness + readiness. Never 500s. Returns `pipeline.self_check()` (doc 02) when the pipeline loaded,
else a degraded stub carrying `load_error`.

```python
@app.get("/api/health")
def health():
    if STATE.pipeline is None:
        return {"ok": False, "loaded": False, "load_error": STATE.load_error}
    return STATE.pipeline.self_check()   # doc 02 owns the keys
```

The server **expects `self_check()` to return at least** `{"ok": bool, ...}`; the recommended shape
(doc 02 may extend) is:

```jsonc
{"ok": true, "loaded": true, "device": "mps", "stub": true,
 "interpreter_ok": true, "generator_ok": true, "decoder_ok": true,
 "references": 6, "issues": []}
```

**Acceptance:** returns 200 within ms; `ok` reflects whether a grade request can be served; when startup
failed, `ok:false`/`loaded:false` and a non‑empty `load_error`.

### 7.2 `GET /api/terms` → `pipeline.terms.all_terms()`

The grounded glossary (doc 05). Static per process. If the pipeline failed to load, still serve it by
importing `webapp.terms.all_terms()` directly (it needs no model weights), so the panel works in a
degraded state.

```python
@app.get("/api/terms")
def get_terms():
    if STATE.pipeline is not None:
        return STATE.pipeline.terms.all_terms()
    import webapp.terms as terms
    return terms.all_terms()
```

Item shape (doc 05): `{term, axis, category, definition, example_usage, grounded}` (the frontend gates
suggestions on `grounded`). Pydantic response model (permissive, since doc 05 also carries `sign` on
directional terms):

```python
class Term(BaseModel):
    term: str
    axis: str
    category: str | None = None
    definition: str
    example_usage: str
    grounded: bool = True
    sign: int | None = None
```

**Acceptance:** returns a non‑empty list (doc 05: 47 grounded + 7 style composites); every item has
`term/axis/definition/example_usage`; `term`↔`axis` pairs are consistent with `eval.tag_vocabulary`;
identical across calls.

### 7.3 `POST /api/generate`

`multipart/form-data`: `image` (file, required), `prompt` (str, required). Runs the whole pipeline.

```python
@app.post("/api/generate", response_model=GenerateResponse)
async def generate(image: UploadFile = File(...), prompt: str = Form(...)):
    if STATE.pipeline is None:
        raise APIError(503, "model_not_loaded", STATE.load_error or "pipeline not loaded")
    if not prompt or not prompt.strip():
        raise APIError(400, "bad_request", "prompt is required")
    raw = await image.read()
    if len(raw) > STATE.cfg.server.max_upload_mb * 1024 * 1024:
        raise APIError(413, "image_too_large", f"max {STATE.cfg.server.max_upload_mb} MB")
    run_id = uuid4().hex
    run_dir = Path(STATE.cfg.server.runs_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        pil = load_rgb_downscaled(raw, STATE.cfg.server.max_image_edge)   # §6
        pil.save(run_dir / "input.png")
    except Exception:
        raise APIError(400, "bad_image", "could not decode image (need PNG/JPEG/WebP)")
    try:
        payload = await _run_pipeline(prompt.strip(), pil, str(run_dir),
                                      STATE.cfg.server.request_timeout_s)
    except asyncio.TimeoutError:
        raise APIError(504, "generation_timeout", "inference exceeded request_timeout_s")
    except Exception as exc:
        log.exception("generation failed")
        raise APIError(500, "generation_failure", f"{type(exc).__name__}: {exc}")
    payload["request_id"] = run_id
    return payload   # validated by GenerateResponse
```

**Pydantic response models** (validate doc 02's payload; the response is `pipeline.run`'s dict plus
`request_id`):

```python
class LutRef(BaseModel):
    cube_url: str

class PreviewOut(BaseModel):
    name: str
    original_url: str
    graded_url: str

class SuggestedTermOut(BaseModel):
    term: str
    axis: str
    definition: str
    example_usage: str
    grounded: bool = True

class PromptFeedbackOut(BaseModel):
    assessment: str
    suggested_terms: list[SuggestedTermOut] = []

class QualityOut(BaseModel):
    behavioral_fidelity: float | None = None
    collapsed: bool | None = None
    fell_back_greedy: bool | None = None

class GenerateResponse(BaseModel):
    request_id: str | None = None
    route: str
    refuse_reason: str | None = None
    clarify_message: str | None = None
    attribute_spec_text: str | None = None
    lut: LutRef | None = None
    previews: list[PreviewOut] = []
    prompt_feedback: PromptFeedbackOut
    quality: QualityOut | None = None
```

**Response examples** (canonical contract; `.cube`/previews are `/runs/<id>/…` files the pipeline
wrote):

```jsonc
// grade
{"request_id":"…","route":"grade","refuse_reason":null,"clarify_message":null,
 "attribute_spec_text":"route=grade | warmer=+2.3 more_contrast=+1.8 …",
 "lut":{"cube_url":"/runs/<id>/output.cube"},
 "previews":[
   {"name":"user_image","original_url":"/runs/<id>/user_image_original.png","graded_url":"/runs/<id>/user_image_graded.png"},
   {"name":"City","original_url":"…","graded_url":"…"}  /* …then Landscape, Portrait, Close-up, Food, Interior (doc 06) */
 ],
 "prompt_feedback":{"assessment":"…","suggested_terms":[{"term":"teal-orange","axis":"composite (calibration window)","definition":"…","example_usage":"teal-orange grade","grounded":false}]},
 "quality":{"behavioral_fidelity":0.31,"collapsed":false,"fell_back_greedy":false}}

// clarify
{"request_id":"…","route":"clarify","refuse_reason":null,
 "clarify_message":"Did you want this warmer or cooler, and how strong?",
 "attribute_spec_text":null,"lut":null,"previews":[],
 "prompt_feedback":{"assessment":"…","suggested_terms":[ … ]}}

// refuse
{"request_id":"…","route":"refuse","refuse_reason":"out_of_scope","clarify_message":null,
 "attribute_spec_text":null,"lut":null,"previews":[],
 "prompt_feedback":{"assessment":"This isn't a global color/tone edit …","suggested_terms":[]}}
```

**Acceptance (grade):** valid RGB image + clear grade prompt (*"make it warmer with strong teal‑orange
contrast"*) → 200, `route=="grade"`, `lut.cube_url` `GET`s a parseable 4913‑row `.cube`
(`LUT_3D_SIZE 17`), `len(previews)==7` with `previews[0].name=="user_image"`, each preview has a
working `original_url` + `graded_url`, non‑empty `attribute_spec_text` starting `route=grade`, and
`prompt_feedback.suggested_terms` present. **Acceptance (clarify):** vague prompt (*"make it pop"*) →
200 `route=="clarify"`, non‑null `clarify_message`, `lut==null`, `previews==[]`, non‑empty
`prompt_feedback`. **Acceptance (refuse):** out‑of‑scope prompt (*"remove the person"*) → 200
`route=="refuse"`, `refuse_reason∈{out_of_scope,out_of_gamut}`, `lut==null`, `previews==[]`.

---

## 8. Structured errors

Single shape; the frontend switches on `error.code`.

```python
class APIError(Exception):
    def __init__(self, status, code, message): self.status, self.code, self.message = status, code, message

@app.exception_handler(APIError)
async def _api_error_handler(_req, exc: APIError):
    return JSONResponse(status_code=exc.status,
                        content={"error": {"code": exc.code, "message": exc.message}})
```

| HTTP | `code`               | when                                             |
|------|----------------------|--------------------------------------------------|
| 503  | `model_not_loaded`   | startup load failed / `STATE.pipeline is None`   |
| 400  | `bad_request`        | missing/empty prompt                             |
| 400  | `bad_image`          | upload not a decodable image                     |
| 413  | `image_too_large`    | upload exceeds `server.max_upload_mb`            |
| 504  | `generation_timeout` | `pipeline.run` exceeded `server.request_timeout_s`|
| 500  | `generation_failure` | any other exception in `run` (traceback logged)  |

Frontend contract: on non‑200, read `body.error.code` and show `body.error.message`.

---

## 9. Minimal runnable `webapp/server.py` skeleton

```python
"""FastAPI server for the local prompt->LUT demo (docs/webapp/03_backend_api.md)."""
from __future__ import annotations
import asyncio, logging, os
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from webapp.models_config import WebappConfig            # doc 02 §6
from webapp.pipeline import PromptToLutPipeline           # doc 02 §1
import webapp.terms as terms                              # doc 05

log = logging.getLogger("webapp")

class _State:
    pipeline: PromptToLutPipeline | None = None
    cfg: WebappConfig | None = None
    load_error: str | None = None
STATE = _State()

app = FastAPI(title="prompt->LUT demo", version="1")
INFERENCE_LOCK = asyncio.Lock()

# ---- Pydantic: Term, LutRef, PreviewOut, SuggestedTermOut, PromptFeedbackOut, QualityOut,
#      GenerateResponse  (see §7)  ----

class APIError(Exception):
    def __init__(self, status, code, message): self.status, self.code, self.message = status, code, message

@app.exception_handler(APIError)
async def _api_error_handler(_req, exc: APIError):
    return JSONResponse(exc.status, content={"error": {"code": exc.code, "message": exc.message}})

@app.on_event("startup")
def _startup():
    cfg = WebappConfig.load(os.environ.get("WEBAPP_CONFIG", "configs/webapp.json"))
    if os.environ.get("WEBAPP_STUB") is not None:         # env wins over the file
        cfg.generator.stub = os.environ["WEBAPP_STUB"] == "1"
    STATE.cfg = cfg
    Path(cfg.server.runs_dir).mkdir(parents=True, exist_ok=True)
    try:
        STATE.pipeline = PromptToLutPipeline(cfg)          # loads all models ONCE
    except Exception as exc:
        log.exception("pipeline load failed"); STATE.load_error = f"{type(exc).__name__}: {exc}"
    app.mount("/runs", StaticFiles(directory=cfg.server.runs_dir), name="runs")
    app.mount("/", StaticFiles(directory=cfg.server.static_dir, html=True), name="static")

def load_rgb_downscaled(raw: bytes, max_edge: int) -> Image.Image:
    im = Image.open(BytesIO(raw)); im.load(); im = im.convert("RGB")
    im.thumbnail((max_edge, max_edge), Image.LANCZOS)
    return im

async def _run_pipeline(prompt, pil, run_dir, timeout_s):
    async with INFERENCE_LOCK:
        return await asyncio.wait_for(run_in_threadpool(STATE.pipeline.run, prompt, pil, run_dir), timeout_s)

@app.get("/api/health")
def health():
    if STATE.pipeline is None:
        return {"ok": False, "loaded": False, "load_error": STATE.load_error}
    return STATE.pipeline.self_check()

@app.get("/api/terms")
def get_terms():
    return STATE.pipeline.terms.all_terms() if STATE.pipeline is not None else terms.all_terms()

@app.post("/api/generate", response_model=GenerateResponse)
async def generate(image: UploadFile = File(...), prompt: str = Form(...)):
    ...   # exactly §7.3
```

> **Note (`on_event` deprecation).** Newer FastAPI prefers a `lifespan=` context manager; mount static
> dirs there after the pipeline load. Either works; never add workers.

---

## 10. Server acceptance checklist (see doc 07 for the driven run)

- Boots with `generator.stub=true` (default) and **no model weights**; `GET /api/health` → `ok:true`,
  `stub:true`.
- `GET /` serves `index.html`; `GET /api/terms` returns the grounded glossary.
- `POST /api/generate` (bundled test image + grade prompt) returns the grade contract (§7.3); the
  `.cube` at `/runs/<id>/output.cube` parses (`LUT_3D_SIZE 17`, 4913 rows); all 7 previews load from
  `/runs/<id>/…`.
- Vague prompt → `clarify`; out‑of‑scope prompt → `refuse` (both 200, no LUT/previews).
- Two overlapping `POST /api/generate` calls **serialize** (second starts after the first releases the
  lock); both succeed.
- Corrupt upload → 400 `bad_image`; oversize → 413 `image_too_large`; forced pipeline failure → 500
  `generation_failure`; the process stays up and the next request works.
