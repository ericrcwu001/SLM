# Prompt→LUT Demo Website — Master Plan

**Read this first.** It is the anchor for a one-pass, overnight build of a **local** web app that
demos the full prompt→LUT pipeline we have *today*, built to plug in future (distilled / RL) models
later. The detailed specs live in the numbered docs in this folder; this doc defines the goal, the
shared contract every other doc obeys, the build order, and the definition of done.

Implementer: a coding agent (e.g. ChatGPT **Codex**) with **computer-use** to launch and verify the
browser. See `08_codex_execution_plan.md` for the exact kickoff prompt and verification loop.

---

## 1. What we're building (product)

A local website where a user:
1. **Uploads an image** and **writes a prompt** describing a color/tone look.
2. Sends it to the pipeline, which returns a **3D LUT** (`.cube`).
3. Sees the result as a gallery: **their image with the LUT applied**, plus **6 neutral, diverse
   reference photos** (city, landscape, portrait, close-up/macro, food, interior) all graded with the
   **same** LUT — a real demo of what the LUT does across content.
4. Gets a **prompt-improvement panel**: the app inspects their prompt and suggests **grounded terms**
   (never full prompts — the user writes their own) to make direction and magnitude clearer. Hovering a
   term shows its definition (plain + technical, e.g. "black point", "matte", "split-tone"). Only terms
   the pipeline provably understands are suggested.

It runs **locally** now (local inference), and is architected so a **future distilled/RL generator**
swaps in via config without touching the app.

## 2. Why it looks the way it does (the research behind the UX)

From `docs/interpreter_results.md`: the Stage-1 interpreter is a **strong router** (grade / clarify /
refuse ~0.88, refuse & clarify recall 1.0) but grade **magnitude** is weak for *vague* prompts
(direction ~0.5, exact magnitude ~0.11) and improves markedly when the prompt carries **explicit
intensity** ("extremely", "slightly"). That drives three product choices:
- The generator is run with **best-of-N reranking** (inference-time quality fix for free-running
  collapse) — deployable now, no retrain.
- The **clarify** route handles genuinely vague requests.
- The **prompt-improvement panel** nudges users toward specific, grounded terms so the model produces a
  clearer LUT. This feature *is* the mitigation for the magnitude finding.

## 3. Shared build contract (every doc coheres to this)

**Repo:** `/Users/ericwu/Developer/SLM`, branch `feat/two-stage`. **Reuse existing Python — do not
reinvent the pipeline:**
- `interpreter/` — Stage-1 router: text → route `{grade, clarify, refuse}` + `attribute_spec_text`
  (`interpreter/example.py:build_prompt_ids`, `interpreter/comparator.py`, `data_pipeline/attribute_spec.py:parse`).
- `sft/generate.py` — free-running VQ-code generation (`generate_codes`, `SpecialIds`, `make_prefix_fn`).
- `eval/best_of_n.py` — best-of-N sampling + rerank by behavioral fidelity. **Use this** for grade.
- `eval/behavioral_fidelity.py:decode_codes(codes)` → frozen VQ-VAE LUT ndarray `[17,17,17,3]`.
- `data_pipeline/attribute_spec.py`, `eval/tag_vocabulary.py` — the grounded term vocabulary.

**Stack:** Python **FastAPI** backend (reuses the repo) + a **single-page static frontend**
(`webapp/static/{index.html,styles.css,app.js}`, **no build step**), hand-crafted polished dark
"cinematic color-tool" design. React/Vite is an optional upgrade, not required.

**Directory layout:**
```
webapp/
  server.py          # FastAPI app, routes, one-time model load at startup, static + artifact serving
  pipeline.py        # PromptToLutPipeline: prompt(+image) -> route -> spec -> codes -> LUT
  models_config.py   # pluggable model registry (interpreter / generator / decoder / device / N)
  lut.py             # decode codes->LUT, apply LUT to image (trilinear), export .cube
  terms.py           # grounded vocabulary + prompt-analysis/suggestion logic
  static/            # index.html, styles.css, app.js  (the SPA)
  assets/references/ # 6 neutral diverse photos + references.json
  _runs/             # per-request generated PNGs + output.cube, served at /runs (doc 03 canonical)
  README.md          # quickstart
configs/webapp.json  # model ids/paths, device, best_of_n N, sampling
```

**API contract:**
- `POST /api/generate` — multipart `{image: file, prompt: str}` → JSON:
  ```json
  {
    "route": "grade|clarify|refuse",
    "refuse_reason": "out_of_scope|out_of_gamut|null",
    "clarify_message": "string|null",
    "attribute_spec_text": "string|null",
    "lut": {"cube_url": "/runs/<uuid>/output.cube"},
    "previews": [
      {"name": "user_image", "original_url": "...", "graded_url": "..."},
      {"name": "City", "original_url": "...", "graded_url": "..."}
    ],
    "prompt_feedback": {
      "assessment": "string",
      "suggested_terms": [
        {"term": "warmer", "axis": "temperature_delta_b", "definition": "...",
         "example_usage": "...", "grounded": true}
      ]
    }
  }
  ```
- `GET /api/terms` → grounded glossary `[{term, axis, category, definition, example_usage}]`.
- `GET /api/health`.

> The JSON/config in this master doc is **illustrative**. The numbered docs are **canonical** where
> they add specifics: `01` (config schema + directory), `02` (pipeline signatures + LUT ops — note
> `apply_lut_trilinear`/`write_cube` already exist and are reused), `03` (exact API models + artifact
> paths under `webapp/_runs/<uuid>/`), `05` (grounded terms), `07` (install/run — canonical deps).

**Pipeline flow:** prompt(+image) → interpreter → route. `refuse` → return refusal (+ friendly copy);
`clarify` → return clarify message + suggested terms; `grade` → `attribute_spec_text` → generator
(best-of-N) → codes → `decode_codes` → LUT`[17³]` → apply (trilinear) to the user image + 6 references
+ export `.cube`. **Model choice is config-driven** so a future distilled/RL generator swaps in.

**Design bar:** polished, modern, cinematic dark theme; strong responsive image grid; hover definition
cards; clear idle/loading/refuse/clarify/error states; download `.cube`. Explicitly **anti-slop** — no
default-bootstrap look.

## 4. The document set

| Doc | Contents |
|---|---|
| `00_master_plan.md` | this file — goal, contract, build order, done criteria |
| `01_architecture.md` | components, data flow, directory, tech stack, pluggable model registry, device/feasibility |
| `02_pipeline.md` | `pipeline.py` + `lut.py` — exact reuse, router logic, best-of-N, decode, trilinear apply, `.cube` export |
| `03_backend_api.md` | `server.py` — FastAPI endpoints, schemas, startup load, artifact serving, single-inference lock, errors |
| `04_frontend.md` | the SPA — design system + components + states + `index.html`/`styles.css`/`app.js` |
| `05_terms.md` | grounded glossary + `/api/terms` payload + the suggestion algorithm |
| `06_reference_images.md` | the 6 neutral diverse photos — categories, sourcing, manifest, fallback |
| `07_runbook_and_verification.md` | setup/run commands + the computer-use verification checklist |
| `08_codex_execution_plan.md` | the one-pass execution plan + literal Codex kickoff prompt |

## 5. Build order (dependencies)

1. **Skeleton + config** — `webapp/` tree, `configs/webapp.json`, `models_config.py` (01).
2. **LUT ops** — `lut.py` (decode reuse, trilinear apply, `.cube` export) — testable with an identity
   LUT before any model loads (02).
3. **Pipeline** — `pipeline.py` wiring interpreter → generator best-of-N → decode (02). Ship a
   **stub-generator mode** (identity/random LUT) so the UI can be built + verified independently of the
   heavy models.
4. **Terms** — `terms.py` glossary + suggestions from the vocabulary tables (05).
5. **Backend** — `server.py` endpoints + artifact serving + startup load (03).
6. **Reference images** — fetch/place the 6 photos + manifest (06).
7. **Frontend** — the polished SPA against the live API (04).
8. **Wire real models** — download interpreter + generator, flip off stub mode, tune `N`/device (01/07).
9. **Verify end-to-end** via computer-use (07).

## 6. Definition of done

- `uvicorn webapp.server:app` serves the SPA at `http://127.0.0.1:8000`.
- Uploading an image + a **grade** prompt returns a valid `.cube` and a gallery of the user's image +
  **6 references** all consistently graded; the `.cube` downloads and opens in a LUT viewer.
- A **vague** prompt hits the `clarify` path with grounded term suggestions; an **out-of-scope** prompt
  ("remove the person") hits `refuse` — both rendered gracefully (not as errors).
- The prompt-improvement panel shows **only grounded** suggested terms; hovering shows definitions.
- The generator model is **config-swappable** (a future distilled/RL adapter needs only a config edit).
- The UI is polished (design system applied), responsive, and works fully in **stub-generator mode**
  even if local heavy-model inference is unavailable.

## 7. Non-goals / guardrails

- No cloud hosting, auth, or multi-user concurrency (local, single-user; serialize inference).
- Do not retrain anything; do not modify the frozen VQ stack or the reused pipeline modules — only add
  the `webapp/` package (plus, if strictly needed, tiny additive helpers). Grade quality is bounded by
  the current generator; best-of-N + the prompt panel are the levers, not model changes.
- Suggested terms MUST be grounded (proven in the pipeline vocabulary) — never invent terms.
