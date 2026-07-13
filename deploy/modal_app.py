"""Deploy the prompt->LUT demo (webapp/server.py) on Modal with real GPU inference.

Two cooperating services, so casual page views never spend GPU credits:
  * `web`  — a cheap CPU container (the PUBLIC front door). Serves the SPA, static assets, the
             glossary (/api/terms) and the shared gallery (/api/gallery + /gallery/*) straight
             from the image + gallery Volume. It reverse-proxies ONLY the model-bound paths
             (/api/generate, /api/warmup, /runs/*) to the GPU service below.
  * `fastapi_app` — the T4 GPU container that actually loads weights and grades. Scales to zero,
             wakes only when the CPU node proxies real inference to it.

So: browsing the landing/dataset/eval/best-of-n pages, reading the glossary, and viewing the
gallery all stay on the CPU node and never wake the T4. The GPU lights up only on warmup/grade.

It also:
  * bakes the repo *code* + frozen VQ decoder into the image (no model weights),
  * caches the model *weights* on a Modal Volume so cold starts don't re-download from HF,
  * uses memory snapshots + lazy weight loading to keep GPU cold starts short.

>>> Share the `web` (CPU) URL, e.g. https://<you>--slm-lut-demo-web.modal.run — NOT fastapi_app.

--------------------------------------------------------------------------------
ONE-TIME SETUP
--------------------------------------------------------------------------------
1. Install + log in (no credit card on the free Starter plan -> you can't be charged):
       pip install modal
       modal token new

2. Store your Hugging Face READ token as a Modal secret (needed to pull the private repos):
       modal secret create huggingface HF_TOKEN=hf_xxxxxxxx

3. Download the weights into the Volume ONCE (interpreter + adapter + resized base).
   Heavy + slow (~10-20 min, downloads the 3B base and resizes it), but you only do it once:
       modal run deploy/modal_app.py::setup_weights

--------------------------------------------------------------------------------
DEPLOY
--------------------------------------------------------------------------------
       modal deploy deploy/modal_app.py
   -> prints TWO public HTTPS URLs (one per web function):
        * https://<you>--slm-lut-demo-web.modal.run           <- SHARE THIS (CPU front door)
        * https://<you>--slm-lut-demo-fastapi-app.modal.run   <- GPU backend (proxy target)
   Share the `web` URL. It serves the whole SPA off a cheap CPU box and only proxies actual
   grading to the T4, which scales to zero and wakes on demand — so idle visitors cost ~nothing.

--------------------------------------------------------------------------------
CUSTOM DOMAIN (optional)
--------------------------------------------------------------------------------
   If you own a domain and want e.g. https://chroma.yourdomain.com instead of *.modal.run:
     * add it in the Modal dashboard (Settings -> Domains) OR pass custom_domains=[...] below,
     * create the CNAME record Modal shows you at your DNS provider.
   (Custom domains may require a paid Modal plan — verify on your account. The *.modal.run
   URL is free and fully public, so start there.)

Notes:
  * `modal deploy` (persistent) vs `modal serve` (ephemeral, hot-reloads for local testing).
  * Modal API surface shifts between versions — if a decorator/kwarg name errors, see the
    inline comments for the older-API fallback.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import modal

# --- paths -------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent   # local repo checkout
_CODE_DIR = "/root/slm"                                # where the repo lands in the container
_WEIGHTS = "/weights"                                  # Volume mount (model weights live here)
_GALLERY = "/data/gallery"                             # Volume mount (shared generated-grade gallery)

# HF repos + subfolders (see docs/interpreter_results.md / docs/webapp/07 §1.6).
INTERPRETER_REPO = "ericrcwu/LUT_SLM_interpreter"
INTERPRETER_SUBDIR = "interp_full_smokefull"   # actual subfolder in the HF repo (verified via list_repo_files)
ADAPTER_REPO = "ericrcwu/LUT_SLM_sft_adapters"
ADAPTER_SUBDIR = "p6_twostage_d0f9c744_smokefull"

# --- image ---------------------------------------------------------------------------------------
# Modal requires every build step to precede add_local_* (adding local files must come LAST, or the
# image rebuilds on every code change). So: build the heavy deps here, fork a GPU variant that
# pre-imports torch, and only THEN add the repo code to both.
_deps_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # Pin torch/transformers to what the repo was built against; leave the rest floating.
    # The Linux torch wheel is CUDA-enabled, and bitsandbytes' CUDA 4-bit path works on Modal.
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",   # Qwen2.5-VL processor (Qwen2VLVideoProcessor) requires torchvision
        "transformers==5.13.0",
        "accelerate",
        "safetensors",
        "peft>=0.11",
        "qwen-vl-utils>=0.0.8",
        "bitsandbytes",
        "sentencepiece",
        "einops",
        "numpy",
        "scipy",
        "pillow",
        "huggingface_hub",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.29",
        "python-multipart>=0.0.9",
    )
    # Separate layer on purpose: adding httpx here doesn't invalidate the big (cached) deps layer
    # above. The CPU edge node uses httpx to reverse-proxy inference to the GPU app.
    .pip_install("httpx>=0.27")
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": f"{_WEIGHTS}/hf_cache"})
)

# GPU variant: fork the deps (marker env layer) and PRE-IMPORT torch in global scope, so the memory
# snapshot (enable_memory_snapshot on the GPU function) captures it and cold starts restore a process
# with torch already loaded instead of paying ~10s to import it. Both `.env()` and `.imports()` are
# registered BEFORE any local files are added, as Modal requires. `.imports()` runs only remotely —
# it's a no-op on your laptop during `modal deploy`. The CPU node keeps `_deps_image` (no torch).
_gpu_deps_image = _deps_image.env({"SLM_ROLE": "gpu"})
with _gpu_deps_image.imports():
    import torch  # noqa: F401  (remote-only global import, captured by the memory snapshot)


def _add_repo_code(img: "modal.Image") -> "modal.Image":
    """Add ONLY the code we need (NOT models/ 10GB, data/, .git, webapp/_runs). Runs last in the
    build so a code change re-adds ~22MB of files instead of rebuilding the deps layers above."""
    pycache = ["**/__pycache__", "**/__pycache__/**", "**/*.pyc"]
    for pkg in ("webapp", "interpreter", "sft", "eval", "data_pipeline", "tokenizer", "configs"):
        ignore = list(pycache)
        if pkg == "webapp":
            ignore += ["_runs", "_runs/**"]   # ephemeral run outputs (regenerated at runtime)
            # NOTE: static/best_of_n (~13MB of precomputed showcase PNGs) IS shipped — the Best-of-N
            # page loads its reference/candidate images from there (best_of_n_showcase.json paths).
        if pkg == "tokenizer":
            ignore += ["checkpoints_mlx", "checkpoints_mlx/**"]  # training checkpoints; final/ kept
        img = img.add_local_dir(str(_REPO_ROOT / pkg), f"{_CODE_DIR}/{pkg}", copy=True, ignore=ignore)
    # include_source=False on the functions disables Modal's 22GB repo automount; re-add just this
    # entrypoint module so the container can locate setup_weights/fastapi_app/web on re-import.
    return img.add_local_python_source("modal_app")


image = _add_repo_code(_deps_image)          # CPU / default image (no torch preload -> stays light)
gpu_image = _add_repo_code(_gpu_deps_image)  # GPU image (torch preloaded for the snapshot)

app = modal.App("slm-lut-demo", image=image)

# Persistent Volume for weights (survives across deploys/containers). Free tier includes 1 TiB.
weights_volume = modal.Volume.from_name("slm-weights", create_if_missing=True)

# Separate persistent Volume for the shared gallery of generated grades. Kept apart from weights
# so user-generated data and model artifacts have independent lifecycles. Modal background-commits
# mounted Volumes every few seconds and on container shutdown, so entries survive scale-to-zero;
# we ALSO wire an explicit commit after each write (below) as insurance against abrupt kills.
gallery_volume = modal.Volume.from_name("slm-gallery", create_if_missing=True)

# HF read token, from: `modal secret create huggingface HF_TOKEN=...`
hf_secret = modal.Secret.from_name("huggingface")


def _write_runtime_config(stub: bool = False) -> str:
    """Write a webapp config and return its path.

    stub=False (GPU service): real mode, device=cuda, model paths pointing at the weights Volume.
    stub=True  (CPU edge node): no models are ever loaded here — it serves static/terms/gallery and
    proxies grading to the GPU service — so device=cpu and generator.stub=True keep construction
    weight-free. Only device + model/decoder paths differ; server dirs stay repo-relative and
    resolve under _CODE_DIR via webapp.models_config.repo_path().
    """
    cfg = {
        "device": "cpu" if stub else "cuda",
        "interpreter": {
            "model_path": f"{_WEIGHTS}/interpreter/{INTERPRETER_SUBDIR}",
            "base_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "tuning_mode": "full",
            "max_new_tokens": 64,
        },
        "generator": {
            "stub": stub,
            "adapter_path": f"{_WEIGHTS}/sft_adapters/{ADAPTER_SUBDIR}",
            "base_model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
            "resized_base_path": f"{_WEIGHTS}/base_resized",
            "input_mode": "attribute_spec_text",
            "spec_bucketize": False,
            "load_in_4bit": True,      # CUDA-only 4-bit; comfortably fits a T4 (16 GB)
            "best_of_n_N": 4,          # keep small on a T4 for latency; raise on a bigger GPU
            "chunk": 4,
            "sampling": {"temperature": 1.0, "top_p": 0.9},
            "max_pixels": 200704,
            "min_pixels": 3136,
        },
        # Point straight at the frozen decoder shipped in the image (skip auto-discovery).
        "vq_decoder": {"final_dir": f"{_CODE_DIR}/tokenizer/final"},
        "server": {
            "runs_dir": "webapp/_runs",
            "static_dir": "webapp/static",
            "references_dir": "webapp/assets/references",
            "max_upload_mb": 20,
            "max_image_edge": 2048,
            "request_timeout_s": 600,
            # Point the shared gallery at the persistent Volume so grades survive scale-to-zero.
            "gallery_dir": _GALLERY,
            "gallery_max_entries": 60,
            "gallery_enabled": True,
        },
    }
    path = f"{_CODE_DIR}/configs/webapp.modal.{'cpu' if stub else 'gpu'}.json"
    Path(path).write_text(json.dumps(cfg, indent=2))
    return path


@app.function(
    volumes={_WEIGHTS: weights_volume},
    secrets=[hf_secret],
    timeout=3600,
    memory=16384,   # resizing the 3B base needs headroom
    cpu=4.0,
    include_source=False,   # don't automount the local repo (weights come from HF into the Volume)
)
def setup_weights():
    """Run ONCE: download the interpreter + adapter and build the resized base into the Volume."""
    from huggingface_hub import snapshot_download

    print("[setup] downloading interpreter router ...")
    snapshot_download(
        INTERPRETER_REPO, allow_patterns=[f"{INTERPRETER_SUBDIR}/*"],
        local_dir=f"{_WEIGHTS}/interpreter",
    )
    print("[setup] downloading generator adapter ...")
    snapshot_download(
        ADAPTER_REPO, allow_patterns=[f"{ADAPTER_SUBDIR}/*"],
        local_dir=f"{_WEIGHTS}/sft_adapters",
    )

    resized = Path(f"{_WEIGHTS}/base_resized")
    if not (resized / "config.json").is_file():
        print("[setup] building resized base (downloads Qwen2.5-VL-3B, adds the adapter's token rows) ...")
        subprocess.run(
            [sys.executable, "-m", "sft.vocab_resize",
             "--config", "configs/sft_default.yaml", "--out", str(resized)],
            cwd=_CODE_DIR, check=True,
        )
    else:
        print("[setup] resized base already present; skipping.")

    weights_volume.commit()   # persist everything written above
    print("[setup] done. Weights cached on the 'slm-weights' Volume.")


# GPU backend: loads weights and grades. Scales to zero (min_containers defaults to 0), capped at
# one container so the app's single inference lock stays meaningful and cost stays bounded. Reached
# by end users only indirectly, via the CPU `web` node's reverse proxy (below).
@app.function(
    image=gpu_image,        # forked image that pre-imports torch in global scope (see gpu_image above)
    gpu="T4",
    volumes={_WEIGHTS: weights_volume, _GALLERY: gallery_volume},
    secrets=[hf_secret],
    memory=8192,            # was 16384; the T4's 16GB VRAM holds the model, host RAM only needs load
    #                         headroom for the 4-bit 3B — 8GB is safe, 4GB risks OOM during load.
    enable_memory_snapshot=True,  # snapshot the (torch-preloaded) process for faster cold starts
    scaledown_window=120,   # stay warm 2 min after the last request, then scale to zero
    timeout=600,            # matches server request_timeout_s
    max_containers=1,       # older Modal: use `concurrency_limit=1`
    include_source=False,   # don't automount the local repo; code is added explicitly to the image
)
@modal.concurrent(max_inputs=20)  # one container serves many HTTP reqs (assets + api) concurrently;
#                                   older Modal: drop this and add allow_concurrent_inputs=20 above.
@modal.asgi_app(
    # custom_domains=["chroma.yourdomain.com"],  # uncomment after adding the CNAME (see header)
)
def fastapi_app():
    # Make the repo importable and configure real mode BEFORE importing the app (it reads
    # WEBAPP_CONFIG at import time). Weights load lazily (on /api/warmup), not here.
    sys.path.insert(0, _CODE_DIR)
    os.chdir(_CODE_DIR)
    os.environ["WEBAPP_CONFIG"] = _write_runtime_config(stub=False)
    os.environ["WEBAPP_STUB"] = "0"

    import webapp.server as web_server  # cheap: construction no longer loads weights

    # Force an immediate durable commit after each gallery write. Modal's background commits already
    # persist the mounted Volume, but an explicit commit guarantees a just-saved grade survives even
    # an abrupt container kill before the next background flush.
    if web_server.STATE.gallery is not None:
        web_server.STATE.gallery.commit_hook = gallery_volume.commit
    return web_server.app


# Paths the CPU edge node cannot serve itself (they need the loaded model or its ephemeral run
# outputs) and must forward to the GPU service. Everything else — the SPA, /assets, /api/terms,
# /api/gallery, /gallery/* — is served locally so it never wakes the T4.
_PROXY_EXACT = {"/api/generate", "/api/warmup"}
_PROXY_PREFIXES = ("/runs/",)
# Hop-by-hop / length headers we must not blindly copy across the proxy hop. We also drop any
# `modal-*` header: Modal injects internal routing headers (e.g. modal-function-call-id) into the
# CPU container's inbound request, and forwarding them to another Modal web endpoint makes its
# ingress reject the request ("prohibited modal header").
_DROP_REQ_HEADERS = {"host", "content-length", "accept-encoding", "connection"}
_DROP_RESP_HEADERS = {"content-length", "transfer-encoding", "content-encoding", "connection"}


def _clean_headers(items, drop: set) -> dict:
    return {k: v for k, v in items if k.lower() not in drop and not k.lower().startswith("modal-")}


# While the GPU container is cold-starting, Modal's routing rejects the inter-container call with a
# 5xx ingress error (e.g. "prohibited modal header: modal-function-call-id") before it ever reaches
# our app. These are safe to retry — the request never executed — so we briefly retry to bridge the
# GPU cold start. Signatures that mark such a transient ingress error (vs. a real app 5xx):
_COLD_START_SIGNATURES = ("modal-http", "prohibited modal header", "failed to respond")
_PROXY_MAX_ATTEMPTS = 12
_PROXY_RETRY_BACKOFF_S = 2.0


def _make_gpu_proxy():
    """Build an ASGI http-middleware dispatch that reverse-proxies inference paths to fastapi_app."""
    import asyncio
    import httpx
    from starlette.responses import JSONResponse, Response

    # Resolved lazily on first use (the GPU app is deployed by the time any inference is requested).
    cache: dict[str, object] = {}

    def _base_url() -> str:
        if "base" not in cache:
            gpu_fn = modal.Function.from_name("slm-lut-demo", "fastapi_app")
            cache["base"] = gpu_fn.get_web_url().rstrip("/")
        return cache["base"]  # type: ignore[return-value]

    def _client() -> "httpx.AsyncClient":
        if "client" not in cache:
            # Read timeout must exceed the GPU's 600s request timeout (cold load + grade).
            cache["client"] = httpx.AsyncClient(timeout=httpx.Timeout(650.0, connect=30.0))
        return cache["client"]  # type: ignore[return-value]

    def _needs_gpu(path: str) -> bool:
        return path in _PROXY_EXACT or path.startswith(_PROXY_PREFIXES)

    async def dispatch(request, call_next):
        if not _needs_gpu(request.url.path):
            return await call_next(request)
        url = _base_url() + request.url.path
        if request.url.query:
            url += "?" + request.url.query
        body = await request.body()
        headers = _clean_headers(request.headers.items(), _DROP_REQ_HEADERS)
        last_exc: Exception | None = None
        upstream = None
        for attempt in range(_PROXY_MAX_ATTEMPTS):
            try:
                upstream = await _client().request(request.method, url, content=body, headers=headers)
            except Exception as exc:  # transport error while the GPU boots — retry a few times
                last_exc = exc
                await asyncio.sleep(_PROXY_RETRY_BACKOFF_S)
                continue
            # A transient Modal ingress error during GPU cold start (not our app) -> retry.
            if upstream.status_code >= 500 and any(sig in upstream.text for sig in _COLD_START_SIGNATURES):
                if attempt < _PROXY_MAX_ATTEMPTS - 1:
                    await asyncio.sleep(_PROXY_RETRY_BACKOFF_S)
                    continue
            break
        if upstream is None:  # exhausted retries on transport errors
            return JSONResponse(
                {"error": {"code": "upstream_unavailable", "message": f"GPU backend error: {last_exc}"}},
                status_code=502,
            )
        out_headers = _clean_headers(upstream.headers.items(), _DROP_RESP_HEADERS)
        return Response(
            content=upstream.content,
            status_code=upstream.status_code,
            headers=out_headers,
            media_type=upstream.headers.get("content-type"),
        )

    return dispatch


# Public CPU front door. No GPU, small RAM, scales to zero. Serves the SPA + static + glossary +
# gallery locally (stub mode: it never loads models) and proxies only real grading to fastapi_app.
@app.function(
    volumes={_GALLERY: gallery_volume},   # read the shared gallery Volume (write happens on the GPU)
    memory=1024,
    cpu=1.0,
    scaledown_window=300,   # keep the cheap CPU box warm across a browsing session
    timeout=700,            # a proxied /api/generate can block on the GPU's 600s request timeout
    max_containers=2,
    include_source=False,
)
@modal.concurrent(max_inputs=100)  # static + proxy fan-out; cheap to serve many at once
@modal.asgi_app()
def web():
    sys.path.insert(0, _CODE_DIR)
    os.chdir(_CODE_DIR)
    os.environ["WEBAPP_CONFIG"] = _write_runtime_config(stub=True)
    os.environ["WEBAPP_STUB"] = "1"

    import webapp.server as web_server  # stub: construction loads no weights

    # Reload the gallery Volume before each /api/gallery read so grades the GPU node just committed
    # show up here too (Modal Volume writes from another container need an explicit reload).
    if web_server.STATE.gallery is not None:
        web_server.STATE.gallery.reload_hook = gallery_volume.reload

    # Register the reverse proxy LAST so it wraps every request (must be added before serving starts).
    web_server.app.middleware("http")(_make_gpu_proxy())
    return web_server.app
