# Codex Execution Plan — one-pass overnight build

This doc tells the implementing agent (ChatGPT **Codex**, with **computer-use** to launch and verify a
browser) how to build the prompt→LUT demo website in a single pass, and how to prove it works. Start
from `00_master_plan.md`, obey its shared contract, and read the numbered docs as you reach each step.

## 0. Prerequisites (confirm before building)
- Repo `/Users/ericwu/Developer/SLM`, branch `feat/two-stage`, working tree clean-ish.
- Python 3.11. **Install per doc 07 (canonical)** — the local/Mac tier is `.[ml]` + `peft` +
  `qwen-vl-utils` + `fastapi uvicorn python-multipart` (NOT `sft,frontier,color`; `bitsandbytes` 4-bit is
  CUDA-only with no macOS-arm64 wheel — on a Mac load unquantized fp16/MPS or use stub mode). No Node needed.
- `HF_TOKEN` set (read) to download models. A GPU helps but is not required — see the stub mode below.
- Everything new goes under `webapp/` (+ `configs/webapp.json`). **Do not modify** the reused pipeline
  modules (`interpreter/`, `sft/`, `eval/`, `data_pipeline/`) — import them.

## 1. Build order (one pass)
Follow `00_master_plan.md` §5. Concretely:
1. Scaffold `webapp/` + `configs/webapp.json` + `webapp/models_config.py` (doc 01).
2. `webapp/lut.py` — `decode_codes` reuse + **trilinear** `apply_lut` + `.cube` export. Unit-test with an
   **identity LUT** (input image ≈ output) before touching models (doc 02).
3. `webapp/pipeline.py` with a **`STUB_GENERATOR` mode** (returns an identity or mild random LUT, no
   model load) AND the real mode (interpreter route + generator best-of-N + decode) (doc 02).
4. `webapp/terms.py` — glossary + `suggest_terms(...)` from the vocabulary tables (doc 05).
5. `webapp/server.py` — FastAPI endpoints, startup load, artifact + static serving, single inference
   lock (doc 03).
6. Reference images into `webapp/assets/references/` + `references.json` (doc 06).
7. `webapp/static/` SPA — design system + components + states (doc 04).
8. Bring up in **stub mode first**; verify the whole UX end-to-end via computer-use (doc 07). THEN set
   real models in `configs/webapp.json`, download them, flip stub off, and re-verify a real grade.

**Parallelism (optional):** if you can run sub-tasks concurrently, the independent tracks are
{`lut.py`+tests} · {`terms.py`+glossary} · {`static/` SPA against a mocked API} · {reference-image
fetch}. `server.py`/`pipeline.py` integrate them and must come after. Keep the API contract fixed so
the tracks converge. If you cannot parallelize, the linear order above is correct.

## 2. Critical correctness guardrails
- **Stub mode is mandatory** and must fully drive the UI, so the demo is verifiable even if local
  heavy-model inference is too slow/OOM on this machine (the generator is a 3B VLM). Make stub vs real a
  single `configs/webapp.json` flag.
- **Serialize inference** (one global lock) — local single-user; the model is not concurrency-safe.
- **Grounded terms only** — every suggested term must come from `eval/tag_vocabulary.py` /
  `data_pipeline/attribute_spec.py`; never invent terms (doc 05).
- **Router-only reality** — `refuse`/`clarify` short-circuit before the generator; only `grade` renders
  a LUT. Render refuse/clarify as first-class states, not errors (doc 04).
- **Downscale** uploaded images for inference/preview speed; keep a reasonable max dimension.

## 3. Computer-use verification loop (must pass before declaring done)
Run `uvicorn webapp.server:app --host 127.0.0.1 --port 8000`, open `http://127.0.0.1:8000`, then:
1. **Grade (specific):** upload a bundled test image, prompt *"make it warmer with strong teal-orange
   contrast"* → expect a returned `.cube` (downloadable) and a gallery of the user's image + **6
   references** all consistently graded; the graded images visibly differ from originals.
2. **Clarify (vague):** prompt *"make it look nicer"* → `clarify` state with grounded magnitude/direction
   term suggestions; hovering a term shows a definition popover.
3. **Refuse (out-of-scope):** prompt *"remove the person in the background"* → `refuse` state, shown
   gracefully.
4. **Prompt panel:** confirm suggested terms are grounded and hover definitions load from `/api/terms`.
5. **Polish:** confirm the design system is applied (dark cinematic theme, real spacing/typography,
   responsive grid) — not a default/bootstrap look.
Capture screenshots of each. If any step fails, use doc 07's failure diagnostics (model OOM → stub mode
+ smaller `N`/device; missing references → fetch/fallback; static/artifact 404 → mount paths).

## 4. Definition of done
All of `00_master_plan.md` §6, verified by the loop above, with screenshots of the grade / clarify /
refuse states and the reference-photo gallery.

---

## 5. Literal kickoff prompt to paste into Codex

> You are implementing a local web app in the repo at `/Users/ericwu/Developer/SLM` (branch
> `feat/two-stage`). **Read `docs/webapp/00_master_plan.md` first, then the numbered docs
> `01`–`07` in `docs/webapp/` as you reach each build step, and follow `08_codex_execution_plan.md`.**
>
> Build the prompt→LUT demo website exactly as specified: a FastAPI backend (reusing the repo's
> `interpreter/`, `sft/generate.py`, `eval/best_of_n.py`, `eval/behavioral_fidelity.py`,
> `data_pipeline/attribute_spec.py`) + a polished single-page static frontend, all under a new
> `webapp/` package plus `configs/webapp.json`. Do NOT modify the reused modules — import them.
>
> Implement in the build order in doc 08 §1. Ship a `STUB_GENERATOR` mode first and verify the entire
> UX end-to-end in the browser with computer-use (doc 08 §3: run `uvicorn webapp.server:app`, open
> `http://127.0.0.1:8000`, exercise the grade / clarify / refuse prompts, the 6-reference gallery, the
> `.cube` download, and the grounded hover glossary). Then wire the real models from
> `configs/webapp.json`, download them (`HF_TOKEN` is set), and re-verify a real grade request.
>
> Keep the UI polished (apply the design system in doc 04 — modern cinematic dark theme; no
> default-bootstrap look). Suggested prompt-terms must be grounded (doc 05) — never invent terms.
> Serialize inference behind one lock. When done, report which verification steps passed with
> screenshots, and note anything that fell back to stub mode.

Adjust paths/ports if the environment differs. Everything the agent needs is in `docs/webapp/`.
