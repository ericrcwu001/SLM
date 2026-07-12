"""Deploy the prompt->LUT demo (webapp/server.py) on Modal with real GPU inference.

This wraps the EXISTING FastAPI app unchanged. It:
  * bakes the repo *code* + frozen VQ decoder into the image (no model weights),
  * caches the model *weights* on a Modal Volume so cold starts don't re-download from HF,
  * serves the whole app (SPA + /api) on a T4, scaling to zero when idle (free-tier friendly).

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
   -> prints a public HTTPS URL like  https://<you>--slm-lut-demo-fastapi-app.modal.run
   That URL is already shareable — anyone can use the live inference. Scales to zero
   when idle, so it only spends credits while a request is actually running.

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

# HF repos + subfolders (see docs/interpreter_results.md / docs/webapp/07 §1.6).
INTERPRETER_REPO = "ericrcwu/LUT_SLM_interpreter"
INTERPRETER_SUBDIR = "interp_full"
ADAPTER_REPO = "ericrcwu/LUT_SLM_sft_adapters"
ADAPTER_SUBDIR = "p6_twostage_d0f9c744_smokefull"

# --- image: heavy deps first (cached), then the repo code last (changes cheaply) -------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    # Pin torch/transformers to what the repo was built against; leave the rest floating.
    # The Linux torch wheel is CUDA-enabled, and bitsandbytes' CUDA 4-bit path works on Modal.
    .pip_install(
        "torch==2.7.1",
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
    .env({"PYTHONUNBUFFERED": "1", "HF_HOME": f"{_WEIGHTS}/hf_cache"})
)

# Upload ONLY the code we need into the image — NOT models/ (10 GB), data/, .git, or webapp/_runs.
# Modal's source AUTOMOUNT (include_source, default True) would otherwise try to ship the whole ~22 GB
# repo from your machine and time out; we disable it on the functions below (include_source=False) and
# add these small dirs (~22 MB total) explicitly. Each lands under _CODE_DIR and is importable once
# _CODE_DIR is on sys.path.
_PYCACHE = ["**/__pycache__", "**/__pycache__/**", "**/*.pyc"]
for _pkg in ("webapp", "interpreter", "sft", "eval", "data_pipeline", "tokenizer", "configs"):
    _ignore = list(_PYCACHE)
    if _pkg == "webapp":
        _ignore += ["_runs", "_runs/**"]                     # ephemeral run outputs (regenerated at runtime)
    if _pkg == "tokenizer":
        _ignore += ["checkpoints_mlx", "checkpoints_mlx/**"]  # training checkpoints; tokenizer/final/ is kept
    image = image.add_local_dir(str(_REPO_ROOT / _pkg), f"{_CODE_DIR}/{_pkg}", copy=True, ignore=_ignore)

app = modal.App("slm-lut-demo", image=image)

# Persistent Volume for weights (survives across deploys/containers). Free tier includes 1 TiB.
weights_volume = modal.Volume.from_name("slm-weights", create_if_missing=True)

# HF read token, from: `modal secret create huggingface HF_TOKEN=...`
hf_secret = modal.Secret.from_name("huggingface")


def _write_runtime_config() -> str:
    """Write a real-mode webapp config pointing the model paths at the Volume; return its path.

    Only the device + model/decoder paths are overridden — server dirs stay repo-relative and
    resolve under _CODE_DIR via webapp.models_config.repo_path().
    """
    cfg = {
        "device": "cuda",
        "interpreter": {
            "model_path": f"{_WEIGHTS}/interpreter/{INTERPRETER_SUBDIR}",
            "base_model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "tuning_mode": "full",
            "max_new_tokens": 64,
        },
        "generator": {
            "stub": False,
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
        },
    }
    path = f"{_CODE_DIR}/configs/webapp.modal.json"
    Path(path).write_text(json.dumps(cfg, indent=2))
    return path


@app.function(
    volumes={_WEIGHTS: weights_volume},
    secrets=[hf_secret],
    timeout=3600,
    memory=16384,   # resizing the 3B base needs headroom
    cpu=4.0,
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


# Serve the FastAPI app. Scales to zero (min_containers defaults to 0), capped at one GPU
# container so the app's single inference lock stays meaningful and cost stays bounded.
@app.function(
    gpu="T4",
    volumes={_WEIGHTS: weights_volume},
    secrets=[hf_secret],
    memory=16384,
    scaledown_window=120,   # stay warm 2 min after the last request, then scale to zero
    timeout=600,            # matches server request_timeout_s
    max_containers=1,       # older Modal: use `concurrency_limit=1`
)
@modal.concurrent(max_inputs=20)  # one container serves many HTTP reqs (assets + api) concurrently;
#                                   older Modal: drop this and add allow_concurrent_inputs=20 above.
@modal.asgi_app(
    # custom_domains=["chroma.yourdomain.com"],  # uncomment after adding the CNAME (see header)
)
def fastapi_app():
    # Make the repo importable and configure real mode BEFORE importing the app (it reads
    # WEBAPP_CONFIG at import time).
    sys.path.insert(0, _CODE_DIR)
    os.chdir(_CODE_DIR)
    os.environ["WEBAPP_CONFIG"] = _write_runtime_config()
    os.environ["WEBAPP_STUB"] = "0"

    from webapp.server import app as web_app  # models load in the FastAPI lifespan on startup
    return web_app
