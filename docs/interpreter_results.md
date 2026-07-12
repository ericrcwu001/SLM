# Stage-1 Interpreter — results and next steps

## TL;DR

The Stage-1 **interpreter** (a separate, text-only `Qwen/Qwen2.5-0.5B-Instruct`, full fine-tuned) maps
a user's free-text request → `attribute_spec_text` + a route `{grade, clarify, refuse}`, decoupled
from the Stage-2 generator by the frozen `data_pipeline/attribute_spec.py` seam. It was built and
evaluated in parallel to the generator collapse-fix loop.

**Outcome — two clear results:**

1. ✅ **Routing works and is production-ready.** Full run (2761 LUTs): route accuracy **0.884**, all
   non-grade recall **1.0** (refuse + clarify), refuse-kind accuracy **1.0**, over-refusal ~13%,
   parse rate 0.89. This is ADR 0020's core safety goal — never silently grade a request that should
   be refused or clarified.
2. ❌ **Grade magnitude is not learnable from vague text.** The exact-magnitude score plateaued at
   `attribute_f1[real_lut] ≈ 0.11` and did not improve with 5× data (full run) **or** with an
   intensity-aware caption fix (`bucket_f1` stayed ~0.15). Cause is task underdetermination, not a
   bug or a data-volume problem.

**Decision: ship the interpreter as a ROUTER** (grade/clarify/refuse gate) and use the **one-stage
generator** (collapse-fix loop) for grade LUT intensity. Do **not** run a full intensity regen.

---

## What was built (all on `feat/two-stage`)

| Component | Path | Role |
|---|---|---|
| Caption corpus generator | `scripts/generate_captions.py` | teacher captions of real LUTs → grade rows (`--workers`, resume-safe, `--no-image`) |
| Route supplement | `scripts/generate_route_supplement.py` | clarify + out_of_gamut rows (additive; validated by absence-of-direction; per-item framing seeds) |
| Corpus unifier | `scripts/build_interpreter_corpus.py` | joins grade + refuse + clarify; stamps each caption with its source LUT's `split_unit_id` (the leakage fix) |
| Interpreter package | `interpreter/{config,corpus,example,train,comparator,score}.py` | standalone config, leakage-safe holdout, text example builder, full-FT trainer, spec-vs-spec comparator, scorer |
| Configs | `configs/candidate_interpreter.json`, `configs/candidate_interpreter_intensity.json` | |
| Notebooks | `notebooks/interpreter_slice_run.ipynb`, `notebooks/interpreter_full_run.ipynb` | slice de-risk; full run + intensity-fix test + HF persist |

Base model = Qwen2.5-0.5B-Instruct, full fine-tuning (bf16 on A100 / fp32 on T4), `upsample_nongrade=1`,
5 epochs. Scored on a leakage-safe `split_unit_id` holdout.

### Eval metrics (the "grade-axis ladder")
For grade rows, three F1s over the asserted per-axis changes, from loosest to strictest:
- **`attribute_direction_f1`** — sign only (did it get the *direction* right, e.g. warmer vs cooler).
- **`attribute_bucket_f1`** — sign **+** coarse magnitude bucket (slight/moderate/strong/extreme via
  the seam's `_bucket_mag`). This is the granularity the generator actually consumes.
- **`attribute_f1`** — sign **+** magnitude within tolerance (exact).

Plus routing metrics: `route_accuracy` (3-way), per-route recall, `refuse_kind_accuracy`,
`interpreter_over_refusal_rate` (grade routed to non-grade), `parse_ok_rate`. The headline `METRIC` is
a unit-macro joint (route × attribute_f1) — **read the components, not METRIC**, which is dominated by
holdout unit-composition.

---

## Results

Routing + grammar (reliable at full scale, n=684 holdout):

| metric | slice (500) | **full run (2761)** | note |
|---|---|---|---|
| route_accuracy | 0.873 | **0.884** (CI 0.858–0.906) | vs always-grade 0.877; catches all non-grade |
| clarify recall | 1.0 (n=11) | **1.0 (n=40)** | |
| refuse recall / kind | 1.0 / 1.0 | **1.0 / 1.0** | |
| grade recall | 0.824 | 0.868 | |
| over-refusal rate | 0.176 | **0.132** | |
| parse_ok_rate | 0.881 | 0.886 | |

Grade axis quality — the ladder (full run, real LUTs):

| | direction | bucket | exact |
|---|---|---|---|
| **full run (non-intensity)** | 0.468 | 0.159 | 0.112 |
| **intensity-fix slice** | 0.380 | 0.148 | 0.138 |

- Exact magnitude was **flat across 5× data** (slice 0.10 → full 0.11) — not a data-volume problem.
- Coarse **bucket** magnitude is barely above exact (0.16) and far below direction (0.47) — so the low
  exact score is **not** a metric artifact; even coarse intensity is mostly wrong.
- The **intensity-fix test** (re-caption with the measured bucket surfaced + "reflect strength in
  wording") did **not** move the aggregate (`bucket_f1` 0.148). The only gain was `literal`-style
  `attribute_f1` → **0.217** (best of any style).

---

## Diagnosis: why grade magnitude fails

**Task underdetermination.** A vague caption ("make it warmer") does not encode *how much*, but the
target is the specific LUT's exact measured magnitude. Across LUTs the same phrasing maps to different
magnitudes, so `(text → magnitude)` supervision is contradictory and unlearnable. The model correctly
learns **direction** (the words carry it) and cannot learn **magnitude** (the words don't). The
fingerprint is direction (0.47) ≫ magnitude (0.11), and the confirmation is that only the `literal`
style — the one that can carry explicit intensity — improved under the intensity fix. The magnitude
ceiling is bounded by **input-language specificity**, which for real vague requests is low. Neither
more data nor intensity-aware prompts change that.

---

## Decision & rationale: router-only

- **Interpreter = router / gatekeeper** for `{grade, clarify, refuse{out_of_scope, out_of_gamut}}`.
  This is the strong, deployable result and the two-stage's real value (ADR 0020: don't silently
  produce a LUT for a request that should be refused/clarified).
- **Grade LUT intensity → the one-stage generator** (raw text → LUT, the collapse-fix loop). It learns
  magnitude end-to-end precisely because it does not bottleneck through an intensity-free spec.
- **Do not run a full 2761-LUT intensity regen** — the slice shows it won't move the aggregate; it
  would only spend teacher budget.

---

## What to do next

1. **Wire the router into the deploy path (small, no training).** On a user request, run the
   interpreter first:
   - `route == refuse` → return the refusal (surface `refuse_kind`); do not call the generator.
   - `route == clarify` → ask the clarifying question; do not call the generator.
   - `route == grade` → forward the **raw user text** to the one-stage generator (not the interpreter's
     spec — magnitude would be lost). This is the architecture the evidence points to; it's glue, not a
     model change.
2. **Finalize the HF artifact paths** in the "Artifacts" section below (replace `<CONFIRM_REPO>`).
3. **Optionally use the interpreter's direction as a soft hint** to the generator (direction is ~0.5
   reliable) — only if a cheap experiment shows it helps; not required for shipping.
4. **Reopen the grade path only if** the input distribution changes to carry intensity — e.g. a
   structured/guided UI where users specify strength, or a preprocessing step that elicits intensity.
   In that regime `literal`-style results (0.22) suggest some magnitude is recoverable; vague free-text
   is not. Do not reinvest in vague-caption magnitude.

---

## Artifacts on Hugging Face

Uploaded by the maintainer — **update these paths** (`<CONFIRM_REPO>`):
- **Trained interpreters:** `hf://<CONFIRM_REPO>` (model), subfolders `interp_full/` (full run) and
  `interp_intensity/` (intensity-fix test); base `Qwen/Qwen2.5-0.5B-Instruct`.
- **Caches + corpus:** `hf://<CONFIRM_DATASET>/interpreter/` — `caption_cache*.jsonl`,
  `route_supplement_cache.jsonl`, `interpreter_rows*.jsonl`, `interpreter_corpus_manifest.json`.
- Related: corpus dataset `hf://datasets/ericrcwu/LUT_SLM`; generator adapters
  `hf://ericrcwu/LUT_SLM_sft_adapters`.

Load the router:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
# tok/model from hf://<CONFIRM_REPO>/interp_full ; then interpreter.example.build_prompt_ids +
# interpreter.comparator.parse on the generated text -> route + spec.
```

---

## Reproduce / run

- **Slice de-risk:** `notebooks/interpreter_slice_run.ipynb` (Qwen2.5-0.5B-Instruct, ~1 h on a T4).
- **Full run + intensity test:** `notebooks/interpreter_full_run.ipynb` — CELL 1–4 full run, CELL 5–8
  intensity-fix test with HF persist after training. A100 + High-RAM; ~2.5–4 h dominated by captioning.
- **Score any adapter:** `python -m interpreter.score --config <cfg> --adapter <dir>` → prints
  `score_summary` + `METRIC=`.

### Appendix: runtime bugs fixed during the runs (transformers 5.x / GPU)
- `apply_chat_template(tokenize=True)` returns a `BatchEncoding` → render-then-tokenize to `list[int]`.
- Training must use bf16 (A100) or fp32 (T4), never raw fp16 (NaN).
- The captioner needs the `[frontier]` extra for the `openai` SDK (`.[sft,frontier]`).
- Captioner resume-poisoning fix (only `status=="generated"` counts as done) and clarify fixes
  (absence-of-direction validation + per-item diversity seeds).
