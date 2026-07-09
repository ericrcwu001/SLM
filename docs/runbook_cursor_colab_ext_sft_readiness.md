<!-- IMPLEMENTATION NOTE (2026-07-09): Step-7 plan approved & underway; see /Users/ericwu/.claude/plans/greedy-stargazing-wall.md.
     A1 scope reduced: only data_pipeline/tokenize_targets.py (encoder) is enabled; eval/lut_decoder.py stays
     DISABLED (enabling cascades into the eval L2-L7 + CLI stack and breaks decoder_disabled tests; not needed
     for SFT-target materialization, which verifies via tokenizer/metrics.py:reconstruct). Decoder enabling is
     deferred to the eval-wiring / Stage-6 phase. -->
# Runbook — Colab A100 SFT-Readiness via the **Cursor / VS Code Colab extension**

> **Purpose.** Reach *SFT-readiness* for the LUT-SLM (prompt→LUT color grading) project on a hosted
> Colab **A100**, driving the runtime through the **official Google Colab extension** in Cursor
> (Open VSX `Google.colab`) instead of the flaky `colab-mcp` browser bridge.
> Execute steps 1–3, **pause before Step 4** (the ~9.85 GB staging pull), run 5–6, and **stop at
> Step 7** for plan-mode approval.
>
> This is the extension-ready sibling of `runbook_colab_sft_readiness.md` (the colab-mcp version).
> Same goal, same gates, same frozen-tokenizer asserts — only the connection method and HF-auth
> step differ.

## Why this variant exists
`colab-mcp` is a known-broken control surface for Claude Code (the notebook tools register only via
`notifications/tools/list_changed`, which our client doesn't re-fetch; connect calls return
`false`/time out and the server dies — see memory `colab-mcp-browser.md`). Google's **official Colab
VS Code extension** (published to Open VSX, so it runs in Cursor) connects a local `.ipynb` to a real
Colab A100 over OAuth — no localhost websocket, no scratchpad, no tool-registration bug.

---

## How the extension actually works (read this — it drives the HF-token design)
- You open a **local `.ipynb`** (kept on your Mac / in the repo). Its **cells execute on the remote
  Colab A100 kernel**. The notebook *file* is local; the *execution* is in the cloud.
- **The filesystem the code sees is the remote Colab VM (`/content`), NOT your local disk.** The
  extension **does not sync your local workspace** to the runtime. So:
  - Your repo-level `.env` on the Mac is **invisible** to the running cells.
  - `.env` is **gitignored**, so `git clone` (Step 2) does **not** bring it to the remote either.
  - ⇒ the HF token must be **placed onto the remote runtime** (Step 3 handles this three ways).
- `!shell` magics and `%%bash` run on the **remote** kernel — all the `!git clone` / `!pip` / `!slm_stage`
  cells below behave exactly as they did under colab-mcp.
- Colab's built-in **Secrets manager is unavailable** in the extension, so
  `from google.colab import userdata` will **fail** — do not use it. Use the loader in Step 3.

## Connect (once, before Step 1)
1. In **Cursor**: Extensions → install **Colab** (`Google.colab`, from Open VSX). If prompted, also
   install its dependency **Jupyter** (`ms-toolsai.jupyter`).
2. Open a new `.ipynb` in the workspace (e.g. `notebooks/colab_sft_readiness.ipynb`).
3. Kernel selector (top-right) → **Colab** → OAuth sign-in with **eric.wu@alphaaiengineering.com**.
4. Choose **Auto Connect** to reuse your live A100, or **New Colab Server** → pick **A100**.
   (A100 requires Colab **Pro/Pro+**.) Status bar should read `Python 3 (Colab)`.
5. **Keep Cursor connected** through the long Step-4 pull (on Pro the kernel dies on idle/disconnect;
   Pro+ gives 24 h background execution). A100 burns ~13 compute units/hr.

## Non-negotiable constraints (unchanged)
- **Frozen tokenizer is authoritative and immutable.** Never retrain, re-gate, or re-freeze.
- **Corpus is read-only.** Do not modify `data/` or `luts/`.
- **Ask before** the ~9.85 GB staging pull, any training, or any push to HF.
- Steps 1–6 execute; **Step 7 stops for user approval in plan mode.**

## Frozen tokenizer identity (assert this matches on Colab)
```
tokenizer_version   = vq_v2_srgbres_17to4_cb256_t64__w91cffdd2c82f
vq_codebook_sha256  = bcdf369dd7cd9a99d71f240b0dac67d404f52130dc8c35d14d6a04514349d118
```

## Key facts / gotchas (unchanged from the mcp runbook)
- Private HF dataset corpus (~9.85 GB, sha256-verified tar shards): `hf://datasets/ericrcwu/LUT_SLM`.
- `slm_stage` = console entry (`data_pipeline.staging.run_staging:main`); pulls+verifies+extracts.
- **`git clone` brings only `tokenizer/final/manifest.json`** — real weights
  (`model.pt`/`encoder.pt`/`decoder.pt`/`codebook.npy`) are gitignored, come **only** from staging.
- **Case matters on Linux:** clone lands at **`/content/SLM`** (repo), staged corpus at
  **`/content/slm`** (lowercase) — different dirs. The authoritative frozen tokenizer is the
  **staged** one at `/content/slm/tokenizer/final/`.
- Repo state: `data_pipeline/tokenize_targets.py` and `eval/lut_decoder.py` are `ENABLED=False`
  (stubs). No SFT training entry point in `pyproject.toml` yet.

---

## Step 1 — Verify runtime (no approval needed)
```python
!nvidia-smi
import sys; print("PY", sys.version)
!df -h /content
```
Confirm: an **A100** (40/80 GB), Python 3.1x, tens of GB free on `/content`.

## Step 2 — Clone + install  ⚠ longish install
```python
!git clone https://github.com/ericrcwu001/SLM
!cd SLM && pip install -e '.[ml]'
```
Use `[ml]` (torch/CUDA), **not** `[mlx]` (Apple-only).

## Step 3 — HF auth (extension-ready; **replaces** `userdata.get`)
The token lives in the **repo-level `.env`** (`HF_TOKEN=...`). Because the remote can't see your local
`.env`, this loader resolves the token from, in order: (1) an already-set env var, (2) a `.env`
**uploaded to the remote** at `/content/SLM/.env` or `/content/.env`, (3) a masked `getpass` paste.
It never prints the token and never writes it into the notebook.

**To use the repo `.env` without pasting:** upload it to the remote once — in Cursor's remote file
explorer drop `.env` into `/content/SLM/`, **or** run
`!` a transfer of your choice. ⚠️ The full repo `.env` also contains Kaggle/freshluts secrets; if you'd
rather not put those on the VM, upload a trimmed file containing only the `HF_TOKEN=` line. Otherwise
just run the cell and paste when prompted (read-only token is enough for staging).

```python
import os, getpass
from pathlib import Path

def _parse_env_file(path):
    vals = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):]
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        vals[k.strip()] = v.strip().strip('"').strip("'")
    return vals

def ensure_hf_token():
    if os.environ.get("HF_TOKEN"):
        return "already in os.environ"
    for p in ("/content/SLM/.env", "/content/.env", ".env"):
        if Path(p).exists():
            v = _parse_env_file(p).get("HF_TOKEN")
            if v:
                os.environ["HF_TOKEN"] = v
                return f"loaded from {p}"
    tok = getpass.getpass("Paste HF_TOKEN (input hidden; copy from repo .env): ").strip()
    if tok:
        os.environ["HF_TOKEN"] = tok
        return "loaded from getpass prompt"
    return None

src = ensure_hf_token()
print("HF_TOKEN source:", src)
print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))
assert os.environ.get("HF_TOKEN"), "HF_TOKEN not set — upload repo .env to the runtime or paste it."
```

## Step 4 — Stage the corpus  ⏸ ASK BEFORE RUNNING (~9.85 GB pull)
```python
import os
!slm_stage stage --durable-root hf://datasets/ericrcwu/LUT_SLM --local-root /content/slm
os.environ["SLM_ARTIFACT_ROOT"] = "/content/slm"
print("SLM_ARTIFACT_ROOT =", os.environ["SLM_ARTIFACT_ROOT"])
!ls -la /content/slm
```
Keep Cursor connected for the duration. Confirm: **all 5 shards sha256-verified**, and `/content/slm`
contains `luts/`, `data/`, `tokenizer/final/`.

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
**If either assert fails → STOP and report. Do not proceed.**

**Clone-vs-staged reconciliation** — make the SFT/tokenize code load the *staged frozen* weights, never
the weightless cloned stub. Either keep `SLM_ARTIFACT_ROOT=/content/slm` and run repo code from
`/content/SLM` (preferred, no copy), or copy the staged tokenizer into the repo:
```python
import shutil, pathlib
src = pathlib.Path("/content/slm/tokenizer/final")
dst = pathlib.Path("/content/SLM/tokenizer/final"); dst.mkdir(parents=True, exist_ok=True)
for p in src.iterdir(): shutil.copy2(p, dst / p.name)
print("copied:", sorted(x.name for x in dst.iterdir()))
```

## Step 6 — Read docs, scope tokenizer→SFT prerequisites (read-only)
```python
for f in ["docs/training_plan_colab.md", "docs/master_plan.md", "docs/model_architecture.md"]:
    print("\n" + "="*80 + f"\n{f}\n" + "="*80)
    print(open(f"/content/SLM/{f}").read())
```
Focus: target tokens materialized by encoding residuals through the **FROZEN** tokenizer;
`data_pipeline/tokenize_targets.py` and `eval/lut_decoder.py` are `ENABLED=False` and must be wired
against `tokenizer/final/manifest.json`; determine whether an SFT entry point + config exist (repo
scan: **not present yet** — expect to create them).

## Step 7 — Readiness report + first-SFT-run plan  ⛔ STOP for approval (plan mode)
Do **not** start training, wire stubs, or modify `data/`/`luts/`. Produce:
- A **readiness report**: runtime specs, staging result, frozen-tokenizer assertion result, exact
  tokenizer→SFT prerequisites, and what's missing (stubs disabled, no entry point).
- A **step-by-step plan** to a first SFT run on this A100, respecting runtime levers: **cap
  `max_pixels`, raise batch size, keep `epochs=2`**; on Pro use Pro+ background exec or keep Cursor
  connected for the run's duration; watch compute-unit burn.
Present via **plan mode** and wait for approval before implementing anything.
