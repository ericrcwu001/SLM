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

**Parallelism (do this):** run the independent tracks concurrently via parallel workers/subagents —
{`lut.py`+tests} · {`terms.py`+glossary} · {`static/` SPA against a mocked API} · {reference-image
fetch}. `server.py`/`pipeline.py` integrate them afterward. Keep the API contract (doc 03) fixed so the
tracks converge cleanly. Only fall back to the linear order if concurrency isn't available.

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

## 5. Kickoff goal to paste into Codex (autonomous + persistent)

Paste this as a standing **goal**, not a one-shot prompt — if your Codex/agent has a persistent
"goal" / "keep-going" mode, use it so it runs until the goal is met or you interrupt.

> **GOAL — build AND fully verify the local prompt→LUT demo website. Work autonomously and keep going
> until every acceptance criterion in `docs/webapp/00_master_plan.md` §6 and
> `docs/webapp/07_runbook_and_verification.md` passes end-to-end. Do NOT stop and do NOT wait for my
> confirmation until it's done and verified, or until I interrupt you.** After each milestone: self-test,
> fix what's broken, and continue. Aim for "perfect and verified," not merely "done."
>
> **Context & specs.** Repo `/Users/ericwu/Developer/SLM`, branch `feat/two-stage`. Read
> `docs/webapp/00_master_plan.md` first, then the numbered docs `01`–`07` as you reach each step (`08`
> is your execution plan). Build a FastAPI backend that REUSES (never modifies) `interpreter/`,
> `sft/generate.py`, `eval/best_of_n.py`, `eval/behavioral_fidelity.py`, `data_pipeline/attribute_spec.py`,
> plus a polished single-page static frontend — all under a new `webapp/` package + `configs/webapp.json`.
>
> **Work in parallel.** Spin up parallel workers/subagents and run the independent tracks concurrently:
> (a) `lut.py` + tests, (b) `terms.py` + glossary, (c) the `static/` SPA against a mocked API, (d)
> choosing/fetching the reference images. Integrate in `server.py`/`pipeline.py` afterward; keep the API
> contract (doc 03) fixed so the tracks converge. Don't build serially if you can parallelize.
>
> **Test relentlessly.** Write and run unit tests first (LUT identity round-trip, `.cube` validity, every
> suggested term is grounded). Bring the app up in `STUB_GENERATOR` mode, then verify the whole UX in a
> real browser with computer-use (doc 07): grade / clarify / refuse prompts, the reference gallery, the
> `.cube` download, and the grounded hover glossary — capture screenshots. Then wire the real models from
> `configs/webapp.json` (`ericrcwu/LUT_SLM_interpreter` → `interp_full/`; `ericrcwu/LUT_SLM_sft_adapters`
> → `p6_twostage_d0f9c744_smokefull/`; `HF_TOKEN` is set) and re-verify a real grade. If anything fails,
> fix it and re-run — iterate until everything is green.
>
> **You have creative freedom — use judgment, never get stuck.** Where the docs don't pin a detail,
> decide well and keep moving. In particular, **CHOOSE the reference photos yourself** — a tasteful,
> genuinely diverse set of neutral/ungraded images (e.g. city, landscape, portrait, close-up, food,
> interior — swap or add if you find better ones) that show the LUT across skin tones, skies, greens,
> neutrals, and highlight/shadow extremes. Make reasonable design and implementation calls within the
> contract + the doc-04 design system. If something is missing, underspecified, or doesn't make sense,
> substitute a sensible fallback (stub mode, procedurally-generated neutral test cards, a smaller
> best-of-N, a different device) and note the deviation — do NOT halt or ask; keep driving toward a
> working, polished, verified demo.
>
> **Hard constraints (do not violate):** don't modify the reused pipeline modules — import them;
> suggested prompt-terms must be grounded (doc 05) — never invent terms; serialize inference behind one
> lock; ship stub-mode first so the UI is verifiable without heavy local inference; keep the UI genuinely
> polished (doc 04 — dark cinematic, no default-bootstrap look). When you believe it's complete, run the
> full verification loop once more and report what passed, with screenshots and any fallbacks taken —
> then keep watching/hardening until I interrupt.

Adjust paths/ports if the environment differs. Everything the agent needs is in `docs/webapp/`.
