# Runbook — Colab A100 SFT-Readiness (frozen tokenizer)

> **Purpose.** Drive a hosted Colab **A100** through steps 1–7 to reach *SFT-readiness* for the
> LUT-SLM (prompt→LUT color grading) project, then stop and present a plan.
> This file exists so a **fresh Claude Code session** (after a CLI restart) can execute the task
> without the original chat context. Your `MEMORY.md` still applies.

## Why this runbook exists
The `colab-mcp` server is browser-driven and its `execute_code`/`add_cell` tools only register
**after** `open_colab_browser_connection` returns `true`. We hit the documented *empty-registry*
failure (returns `false`/times out even though the Firefox window shows a green A100), whose only
reliable fix is a **full Claude Code restart** with the Firefox tab already live. See memory note
`colab-mcp-browser.md`.

## Non-negotiable constraints
- **Frozen tokenizer is authoritative and immutable.** Never retrain, re-gate, or re-freeze. USE the
  exact frozen tokenizer that was trained locally and shipped in the HF corpus.
- **Corpus is read-only.** Do **not** modify `data/` or `luts/`.
- **Ask before** any long/expensive run (the ~9.85 GB staging pull; any training) or any push to HF.
- Steps 1–6 execute; **Step 7 stops for user approval in plan mode** — no training, no wiring stubs.

## Frozen tokenizer identity (assert this matches on Colab)
```
tokenizer_version   = vq_v2_srgbres_17to4_cb256_t64__w91cffdd2c82f
vq_codebook_sha256  = bcdf369dd7cd9a99d71f240b0dac67d404f52130dc8c35d14d6a04514349d118
```

## Key facts / gotchas
- Private HF dataset corpus (~9.85 GB, sha256-verified tar shards): `hf://datasets/ericrcwu/LUT_SLM`.
- Repo: `https://github.com/ericrcwu001/SLM`. `slm_stage` = console entry
  (`data_pipeline.staging.run_staging:main`) that pulls+verifies+extracts the corpus.
- **`git clone` brings only `tokenizer/final/manifest.json`** — the real weights
  (`model.pt`/`encoder.pt`/`decoder.pt`/`codebook.npy`) are gitignored and come **only** from staging.
- **Case matters on Linux:** clone lands at **`/content/SLM`** (repo), staged corpus at
  **`/content/slm`** (lowercase). These are *different directories*. The authoritative frozen
  tokenizer is the **staged** one at `/content/slm/tokenizer/final/`.
- Repo state (as of this runbook): `data_pipeline/tokenize_targets.py` and `eval/lut_decoder.py` are
  both `ENABLED = False` (stubs). There is **no SFT training entry point** in `pyproject.toml` yet.

---

## Reconnect first (fresh session)
1. Ensure the **colab-mcp Firefox window** is signed in, on the right notebook, **A100**, **Connected/green**,
   and the notebook's `HF_TOKEN` secret is toggled **on**.
2. Call `open_colab_browser_connection`; require `true` and confirm `execute_code` is now available.
   If it returns `false`/times out → empty-registry again → restart Claude Code and retry.

Run each step as an `execute_code` call; read the output before the next. Optionally mirror durable
cells with `add_cell`.

## Step 1 — Verify runtime (no approval needed)
```python
!nvidia-smi
import sys; print("PY", sys.version)
!df -h /content
```
Confirm: an **A100** (40 or 80 GB), Python 3.1x, tens of GB free on `/content`.

## Step 2 — Clone + install  ⚠ longish install
```python
!git clone https://github.com/ericrcwu001/SLM
!cd SLM && pip install -e '.[ml]'
```
Use `[ml]` (torch/CUDA — authoritative here), **not** `[mlx]` (Apple-only).

## Step 3 — HF auth from the Colab secret
```python
from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))
```

## Step 4 — Stage the corpus  ⏸ ASK BEFORE RUNNING (~9.85 GB pull)
```python
import os
!slm_stage stage --durable-root hf://datasets/ericrcwu/LUT_SLM --local-root /content/slm
os.environ["SLM_ARTIFACT_ROOT"] = "/content/slm"
print("SLM_ARTIFACT_ROOT =", os.environ["SLM_ARTIFACT_ROOT"])
!ls -la /content/slm
```
Confirm: **all 5 shards sha256-verified**, and `/content/slm` contains `luts/`, `data/`,
`tokenizer/final/`.

## Step 5 — VERIFY THE FROZEN TOKENIZER (hard gate)
```python
import json, pathlib
d = pathlib.Path("/content/slm/tokenizer/final")
need = {"model.pt", "encoder.pt", "decoder.pt", "codebook.npy", "manifest.json"}
have = {p.name for p in d.iterdir()}
assert need <= have, f"MISSING weights: {sorted(need - have)}"

m = json.load(open(d / "manifest.json"))
EXPECT_VER = "vq_v2_srgbres_17to4_cb256_t64__w91cffdd2c82f"
EXPECT_SHA = "bcdf369dd7cd9a99d71f240b0dac67d404f52130dc8c35d14d6a04514349d118"
assert m.get("tokenizer_version")  == EXPECT_VER, ("VERSION MISMATCH", m.get("tokenizer_version"))
assert m.get("vq_codebook_sha256") == EXPECT_SHA, ("SHA MISMATCH",     m.get("vq_codebook_sha256"))
print("FROZEN TOKENIZER OK:", m["tokenizer_version"])
```
**If either assert fails → STOP and report to the user. Do not proceed.**

**Clone-vs-staged reconciliation** — guarantee the SFT/tokenize code loads the *staged frozen* weights,
never the weightless cloned stub and never a freshly built tokenizer. Pick one:
- Keep `SLM_ARTIFACT_ROOT=/content/slm` and run code from `/content/SLM` (repo) so the loader resolves
  artifacts via `SLM_ARTIFACT_ROOT` (preferred, no copy), **or**
- Copy the staged tokenizer into the repo:
  ```python
  import shutil, pathlib
  src = pathlib.Path("/content/slm/tokenizer/final")
  dst = pathlib.Path("/content/SLM/tokenizer/final"); dst.mkdir(parents=True, exist_ok=True)
  for p in src.iterdir(): shutil.copy2(p, dst / p.name)
  print("copied:", sorted(x.name for x in dst.iterdir()))
  ```
Then confirm the tokenizer loader actually resolves to the staged weights before any tokenization.

## Step 6 — Read docs, scope tokenizer→SFT prerequisites (read-only)
```python
for f in ["docs/training_plan_colab.md", "docs/master_plan.md", "docs/model_architecture.md"]:
    print("\n" + "="*80 + f"\n{f}\n" + "="*80)
    print(open(f"/content/SLM/{f}").read())
```
Focus (from `training_plan_colab.md` → "Stage 5 SFT" + "Runtime And Credit Optimization"):
- Target tokens must be **materialized by encoding residuals through the FROZEN tokenizer**.
- `data_pipeline/tokenize_targets.py` and `eval/lut_decoder.py` are `ENABLED=False` and must be wired
  against `tokenizer/final/manifest.json` first.
- Determine whether an SFT training entry point + config exist or must be created (repo scan says: **not
  present yet** — expect to create them).

## Step 7 — Readiness report + first-SFT-run plan  ⛔ STOP for approval (plan mode)
Do **not** start training, wire stubs, or modify `data/`/`luts/`. Produce:
- A **readiness report**: runtime specs, staging result, frozen-tokenizer assertion result, exact
  prerequisites between the frozen tokenizer and SFT, and what's missing (stubs disabled, no entry point).
- A **step-by-step plan** to a first SFT run on this A100, respecting runtime levers: **cap `max_pixels`,
  raise batch size, keep `epochs=2`**.
Present via **plan mode** and wait for user approval before implementing anything.
