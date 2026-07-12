# Stage-1 Interpreter — build, results, and artifacts

The **interpreter** is the Stage-1 model of the two-stage prompt→LUT system (ADR 0020): a *separate*,
text-only `Qwen/Qwen2.5-0.5B-Instruct` (full fine-tuned) that maps a user's free-text request →
`attribute_spec_text` + a route `{grade, clarify, refuse}`. It is decoupled from the Stage-2
generator by the frozen `data_pipeline/attribute_spec.py` seam and was built/run entirely in parallel
to the generator collapse-fix loop.

## Pipeline (all committed on `feat/two-stage`)
- **Caption corpus:** `scripts/generate_captions.py` (teacher, `--workers`, resume-safe) → grade rows.
- **Route supplement:** `scripts/generate_route_supplement.py` → clarify + out_of_gamut rows
  (additive; validated by absence-of-direction; per-item framing seeds for diversity).
- **Corpus unifier:** `scripts/build_interpreter_corpus.py` → `data/interpreter/interpreter_rows.jsonl`,
  stamping each caption with its source LUT's `split_unit_id` (the leakage fix).
- **Package `interpreter/`:** `config.py`, `corpus.py` (leakage-safe holdout), `example.py`,
  `train.py` (full-FT, bf16/fp32), `comparator.py` (route + direction + attribute F1),
  `score.py` (unit-macro METRIC + component columns). Notebooks: `interpreter_slice_run.ipynb`,
  `interpreter_full_run.ipynb`.

## Results

**Slice de-risk (500 LUTs)** and **full run (all 2761 LUTs)**, scored on the leakage-safe
`split_unit_id` holdout, Qwen2.5-0.5B-Instruct full-FT, `upsample_nongrade=1`, 5 epochs:

| metric | slice (3-way) | full run | notes |
|---|---|---|---|
| route_accuracy | 0.873 | **0.884** (CI 0.858–0.906) | vs always-grade 0.877; catches all non-grade |
| clarify recall | 1.0 (n=11) | **1.0 (n=40)** | |
| refuse recall / kind | 1.0 / 1.0 | **1.0 / 1.0** | |
| grade recall | 0.824 | 0.868 | |
| interpreter_over_refusal_rate | 0.176 | **0.132** | |
| parse_ok_rate | 0.881 | 0.886 | |
| attribute_direction_f1[real_lut] | 0.42 | 0.468 (~0.54 cond.) | sign-only |
| **attribute_f1[real_lut]** | 0.101 | **0.112** | sign+magnitude; **flat despite 5× data** |
| METRIC (unit-macro joint) | 0.78 | 0.53 | drop is holdout unit-composition, not regression |

**Verdict:**
- **Routing is production-ready** and improved at scale — this is ADR 0020's core safety goal
  (never silently grade a refuse/clarify request). Bankable win.
- **Exact magnitude did NOT scale** (`attribute_f1` 0.10→0.11 with 5× data). Diagnosis: **task
  underdetermination** — vague captions don't encode intensity, but the metric demands each LUT's
  exact measured magnitude. Evidence: `attribute_f1_by_style` literal 0.16 > concept 0.09
  (specificity helps), and direction (0.47) ≫ magnitude (0.11).
- **Resolved (bucket metric):** `attribute_bucket_f1[real_lut] = 0.159` — barely above exact
  (0.112) and far below direction (0.468). So coarse magnitude is ALSO poor; the low exact score was
  NOT a metric artifact. **Grade magnitude is genuinely not learned.** Root cause = intensity-free
  captions: the captioner writes "make it warmer" regardless of whether the LUT is slightly or
  strongly warmer, so `(text → magnitude)` supervision is contradictory across LUTs and unlearnable.
  Direction is learned (words carry it); magnitude is not (words don't).
- **Intensity-fix test (RESOLVED the fork → router-only).** Re-captioned a 500-LUT slice with
  intensity-aware prompts (teacher told each axis's bucket + to reflect strength in wording), retrained
  `interp_intensity`, scored: `attribute_bucket_f1[real_lut] = 0.148` — unchanged from 0.16, NO jump.
  The fix did not rescue aggregate magnitude. Only tell: `literal`-style `attribute_f1` rose to 0.217
  (best of any style) — intensity IS learnable when the language explicitly carries it, but the vague
  styles (concept/mood/slang, ~like real user requests) can't carry it and dominate the macro. So the
  magnitude ceiling is bounded by input-language specificity, not data or the fix.
- **DECISION: ship the interpreter as a ROUTER** (grade/clarify/refuse gatekeeper — route acc 0.88,
  all non-grade recall 1.0, refuse-kind 1.0; the ADR-0020 safety win) and use the **one-stage
  generator** (collapse-fix loop) for grade LUT intensity. Do NOT do a full 2761-LUT intensity regen —
  the slice shows it won't move the aggregate. The two-stage's value is routing + a clean refuse/
  clarify gate, not text→magnitude.

Three GPU/transformers-5.x bugs were fixed live: `apply_chat_template(tokenize=True)` returns a
BatchEncoding (→ render-then-tokenize to `list[int]`); training must use bf16/fp32 not raw fp16
(NaN); the captioner needs the `[frontier]` extra for the openai SDK.

## Artifacts on Hugging Face
Uploaded by the maintainer (repo/revision to confirm — update the paths below):
- **Trained interpreter** (`interp_full` full-FT model): `hf://<CONFIRM_REPO>` (base
  `Qwen/Qwen2.5-0.5B-Instruct`).
- **Caption cache** (`data/active_sft/caption_cache.jsonl`) + **route-supplement cache**
  (`data/active_sft/route_supplement_cache.jsonl`): `hf://<CONFIRM_REPO/DATASET>`.
- **Results:** this document + the `score_summary` JSON from `interpreter/score.py` (full-run values
  in the table above).

See also the corpus dataset (`hf://datasets/ericrcwu/LUT_SLM`) and generator adapters
(`hf://ericrcwu/LUT_SLM_sft_adapters`).
