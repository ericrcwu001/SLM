"""FastAPI server for the local prompt-to-LUT demo."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel, Field

import webapp.terms as terms
from webapp.gallery import GalleryStore
from webapp.models_config import WebappConfig, repo_path
from webapp.pipeline import PromptToLutPipeline

log = logging.getLogger("webapp")


class _State:
    pipeline: PromptToLutPipeline | None = None
    cfg: WebappConfig | None = None
    load_error: str | None = None
    gallery: GalleryStore | None = None
    load_task: asyncio.Task | None = None   # background model-weight load, started on first warmup/grade


STATE = _State()
INFERENCE_LOCK = asyncio.Lock()
# Serializes the "start the background weight load exactly once" decision (see _start_model_load).
_LOAD_LOCK = asyncio.Lock()
_BACKGROUND_INFERENCE: set[asyncio.Task] = set()


class Term(BaseModel):
    term: str
    axis: str
    category: str | None = None
    definition: str
    example_usage: str
    grounded: bool = True
    sign: int | None = None


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
    suggested_terms: list[SuggestedTermOut] = Field(default_factory=list)


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
    previews: list[PreviewOut] = Field(default_factory=list)
    prompt_feedback: PromptFeedbackOut
    quality: QualityOut | None = None


class GalleryEntryOut(BaseModel):
    id: str
    prompt: str
    spec_text: str | None = None
    quality: QualityOut | None = None
    created_at: float
    before_url: str
    after_url: str
    cube_url: str


class GalleryListOut(BaseModel):
    entries: list[GalleryEntryOut] = Field(default_factory=list)


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message


def _load_config() -> WebappConfig:
    return WebappConfig.load(os.environ.get("WEBAPP_CONFIG", str(repo_path("configs/webapp.json"))))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    cfg = STATE.cfg or _load_config()
    STATE.cfg = cfg
    try:
        # Construction is cheap by design: model *weights* are loaded lazily via /api/warmup
        # (kicked off when the user picks an image), NOT here — so the page serves immediately
        # even on a cold container instead of blocking behind a multi-GB load.
        STATE.pipeline = PromptToLutPipeline(cfg)
        STATE.load_error = None
        log.info("pipeline constructed: device=%s stub=%s adapter=%s", cfg.device, cfg.generator.stub, cfg.generator.adapter_path)
    except Exception as exc:  # keep the SPA and diagnostics available
        log.exception("pipeline construction failed")
        STATE.pipeline = None
        STATE.load_error = f"{type(exc).__name__}: {exc}"
    yield


STATE.cfg = _load_config()
# Resolve serving dirs against the repo root and create them BEFORE mounting, so importing this
# module never crashes on a missing dir and the app can launch from any working directory.
RUNS_DIR = repo_path(STATE.cfg.server.runs_dir)
STATIC_DIR = repo_path(STATE.cfg.server.static_dir)
ASSETS_DIR = repo_path(STATE.cfg.server.references_dir).parent
GALLERY_DIR = repo_path(STATE.cfg.server.gallery_dir)
for _dir in (RUNS_DIR, STATIC_DIR, ASSETS_DIR, GALLERY_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
# Shared, persisted gallery of generated grades.  On Modal the deploy layer wires
# STATE.gallery.commit_hook = <volume>.commit for immediate cross-restart durability.
if STATE.cfg.server.gallery_enabled:
    STATE.gallery = GalleryStore(GALLERY_DIR, STATE.cfg.server.gallery_max_entries)
app = FastAPI(title="prompt-to-LUT demo", version="1", lifespan=lifespan)


@app.exception_handler(APIError)
async def api_error_handler(_request, exc: APIError):
    return JSONResponse(status_code=exc.status, content={"error": {"code": exc.code, "message": exc.message}})


# Guard against decompression bombs: reject before rasterizing, since a small compressed file can
# expand to an enormous in-memory bitmap. Reading .size only parses the header (no full decode).
_MAX_DECODE_PIXELS = 50_000_000  # ~50 MP


def load_rgb_downscaled(raw: bytes, max_edge: int) -> Image.Image:
    with Image.open(BytesIO(raw)) as opened:
        width, height = opened.size
        if width * height > _MAX_DECODE_PIXELS:
            raise ValueError(f"image too large to decode: {width}x{height} exceeds {_MAX_DECODE_PIXELS} pixels")
        opened.load()
        image = opened.convert("RGB")
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return image


def _retain_task(task: asyncio.Task) -> None:
    """Keep fire-and-drain tasks alive until their callback removes them."""
    _BACKGROUND_INFERENCE.add(task)
    task.add_done_callback(_BACKGROUND_INFERENCE.discard)


async def _drain_pipeline_and_release(task: asyncio.Task) -> None:
    """Hold the inference lock until a timed-out worker thread actually exits."""
    try:
        await task
    except BaseException:  # the request already reported the worker failure/timeout
        pass
    finally:
        INFERENCE_LOCK.release()


async def _run_pipeline(prompt: str, image: Image.Image, run_dir: Path, timeout_s: int):
    pipeline = STATE.pipeline
    if pipeline is None:  # explicit guard (never an assert, which -O would strip) BEFORE acquiring
        raise APIError(503, "model_not_loaded", STATE.load_error or "pipeline not loaded")
    await INFERENCE_LOCK.acquire()
    release_here = True
    try:
        worker = asyncio.create_task(run_in_threadpool(pipeline.run, prompt, image, run_dir))
        _retain_task(worker)
        try:
            return await asyncio.wait_for(asyncio.shield(worker), timeout=timeout_s)
        except BaseException:
            if not worker.done():
                # Cancelling the await cannot stop a Python worker thread.  Transfer ownership of the
                # lock to a drain task so the next request cannot overlap the still-running inference.
                release_here = False
                cleanup = asyncio.create_task(_drain_pipeline_and_release(worker))
                _retain_task(cleanup)
            raise
    finally:
        if release_here:
            INFERENCE_LOCK.release()


async def _start_model_load() -> asyncio.Task | None:
    """Kick off the heavy weight load in the background exactly once; return its task.

    Returns None when there is nothing to load (no pipeline, or a stub / already-warm pipeline).
    Does NOT wait for the load — callers that need the models ready should await the task.
    """
    pipeline = STATE.pipeline
    if pipeline is None or pipeline.is_ready():
        return None
    async with _LOAD_LOCK:
        if STATE.load_task is None:
            STATE.load_task = asyncio.create_task(run_in_threadpool(pipeline.load_models))
        return STATE.load_task


async def _ensure_models_ready() -> None:
    """Block until the lazily-loaded models are ready, starting the load if it hasn't begun.

    Cheap when already ready (stub, or a prior /api/warmup finished). Raises 503 if the pipeline
    is absent or the load has failed.
    """
    pipeline = STATE.pipeline
    if pipeline is None:
        raise APIError(503, "model_not_loaded", STATE.load_error or "pipeline not loaded")
    if pipeline.is_ready():
        return
    task = await _start_model_load()
    if task is not None:
        await task
    if not pipeline.is_ready():
        raise APIError(503, "model_not_loaded", pipeline.load_error or "models failed to load")


@app.post("/api/warmup")
async def warmup():
    """Begin loading model weights in the background so a grade is ready by the time the user
    finishes typing.  The front-end calls this the instant an image is selected.

    Fire-and-forget and idempotent: it never blocks on the load and is safe to call repeatedly.
    """
    pipeline = STATE.pipeline
    if pipeline is None:
        return {"status": "error", "load_error": STATE.load_error}
    if pipeline.is_ready():
        return {"status": "ready"}
    await _start_model_load()
    return {"status": "loading"}


@app.get("/api/health")
def health():
    if STATE.pipeline is None:
        return {"ok": False, "loaded": False, "ready": False, "load_error": STATE.load_error}
    report = STATE.pipeline.self_check()
    # Distinguish "still warming up" (weights loading in a background task) from "genuinely broken",
    # so a not-yet-ready report isn't misread as a failure.
    report["loading"] = STATE.load_task is not None and not STATE.load_task.done()
    return report


@app.get("/api/terms", response_model=list[Term])
def get_terms():
    return STATE.pipeline.terms.all_terms() if STATE.pipeline is not None else terms.all_terms()


@app.get("/api/gallery", response_model=GalleryListOut)
def get_gallery(limit: int | None = None):
    if STATE.gallery is None:
        return {"entries": []}
    return {"entries": STATE.gallery.list(limit)}


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(image: UploadFile = File(...), prompt: str = Form(...)):
    if STATE.pipeline is None or STATE.cfg is None:
        raise APIError(503, "model_not_loaded", STATE.load_error or "pipeline not loaded")
    # Weights load lazily — normally already kicked off by /api/warmup when the image was picked.
    # If that hasn't finished (or was never called), start it now and wait rather than 503-ing.
    await _ensure_models_ready()
    prompt = (prompt or "").strip()
    if not prompt:
        raise APIError(400, "bad_request", "prompt is required")
    # Read in bounded chunks so an oversized upload is rejected before the whole body lands in memory.
    max_bytes = STATE.cfg.server.max_upload_mb * 1024 * 1024
    chunks: list[bytes] = []
    received = 0
    while True:
        chunk = await image.read(1024 * 1024)
        if not chunk:
            break
        received += len(chunk)
        if received > max_bytes:
            raise APIError(413, "image_too_large", f"maximum upload is {STATE.cfg.server.max_upload_mb} MB")
        chunks.append(chunk)
    raw = b"".join(chunks)

    run_id = uuid4().hex
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        pil_image = load_rgb_downscaled(raw, STATE.cfg.server.max_image_edge)
        pil_image.save(run_dir / "input.png")
    except Exception as exc:
        log.info("bad image upload: %s", exc)
        raise APIError(400, "bad_image", "could not decode image; use PNG, JPEG, or WebP") from exc
    try:
        payload = await _run_pipeline(prompt, pil_image, run_dir, STATE.cfg.server.request_timeout_s)
    except asyncio.TimeoutError as exc:
        raise APIError(504, "generation_timeout", "inference exceeded the configured request timeout") from exc
    except Exception as exc:
        log.exception("generation failed")
        raise APIError(500, "generation_failure", f"{type(exc).__name__}: {exc}") from exc
    payload["request_id"] = run_id
    # Persist successful grades to the shared gallery.  Best-effort: a gallery write failure
    # must never fail a grade the user already received.  Runs in a thread (PIL + disk + commit).
    if STATE.gallery is not None and payload.get("route") == "grade":
        try:
            await run_in_threadpool(
                STATE.gallery.add_from_run,
                run_dir,
                prompt=prompt,
                spec_text=payload.get("attribute_spec_text"),
                quality=payload.get("quality"),
            )
        except Exception:
            log.exception("gallery save failed for run %s", run_id)
    return payload


# Route registration must precede all mounts; the root SPA mount is intentionally last.
app.mount("/runs", StaticFiles(directory=str(RUNS_DIR)), name="runs")
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
app.mount("/gallery", StaticFiles(directory=str(GALLERY_DIR)), name="gallery")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
