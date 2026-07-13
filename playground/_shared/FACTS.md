# FACTS.md — single source of truth for website updates

Every number below is cited to a repo file (path:line or section). Rule: if a
claim is not backed by a source here, do NOT state it. Items marked
**NOT FOUND — do not state** were not verifiable in the repo.

Provenance of the authoritative numbers:
- Pipeline funnel/tiers/dedup: `data/run_summary.json`, `data/raw_registry/derivation_attrition.json`, `data/raw_registry/acquisition_report.json`, `data/raw_registry/provenance.jsonl` (7,760 rows, counted).
- Active set: `data/active_sft/active_rows.jsonl` (3,033 rows, counted live).
- Split: `data/splits/frozen/split_43e6cd41ca0ee35e/frozen_split.json`, `data/run_summary.json`.
- Tokenizer recon: `tokenizer/final/manifest.json` (`gate_report`).
- Frontier: `docs/frontier_baseline_results.md` (documented headline) + `playground/_shared/frontier_scores_fresh.csv` (fresh re-score).
- Model metrics: `docs/collapse_fix/README.md`, `docs/interpreter_results.md`, `HANDOFF.md`.

---

## 1. Canonical identifiers & URLs

- GitHub repo: `https://github.com/ericrcwu001/SLM` — source: `git remote -v` (origin = `https://github.com/ericrcwu001/SLM.git`)
- HF dataset repo (corpus, durable staging root): `hf://datasets/ericrcwu/LUT_SLM` — source: `configs/staging_default.yaml` (`durable_root`); `README.md:37`; `docs/interpreter_results.md:137`
- HF interpreter repo + subfolder (DEPLOYED): `ericrcwu/LUT_SLM_interpreter`, subfolder `interp_full_smokefull/` — source: `deploy/modal_app.py:61-62` (`INTERPRETER_REPO`, `INTERPRETER_SUBDIR = "interp_full_smokefull"`, "verified via list_repo_files"). NOTE: `docs/interpreter_results.md:131-133` names subfolders `interp_full/` and `interp_intensity/`; the deploy path uses `interp_full_smokefull/`. Use the deploy value.
- HF adapter (generator) repo + subfolder: `ericrcwu/LUT_SLM_sft_adapters`, subfolder `p6_twostage_d0f9c744_smokefull/` — source: `deploy/modal_app.py:63-64`; `docs/interpreter_results.md:135`; `HANDOFF.md:321`
- HF interpreter cache dataset: `hf://datasets/ericrcwu/LUT_SLM_interpreter_cache` — source: `docs/interpreter_results.md:133-134`
- Base model — generator: `Qwen/Qwen2.5-VL-3B-Instruct` — source: `docs/model_architecture.md:83`; `deploy/modal_app.py:141`
- Base model — interpreter (as trained/deployed): `Qwen/Qwen2.5-0.5B-Instruct` (full fine-tune) — source: `docs/interpreter_results.md:5,38`; `deploy/modal_app.py:134`. NOTE the architecture-doc *default* is `LiquidAI/LFM2.5-350M-Base` (`docs/model_architecture.md:17,53,70`) — a spec default, NOT the built interpreter. State Qwen2.5-0.5B-Instruct as the real one.
- Vercel eval-harness URL: `https://evaluation-harness-hafxqrbf8-alpha-eric.vercel.app/` — source: provided by task owner
- Modal app name: `slm-lut-demo` — source: `deploy/modal_app.py:114` (`modal.App("slm-lut-demo")`). Public URL pattern (username-dependent, not fixed): `https://<user>--slm-lut-demo-fastapi-app.modal.run` — source: `deploy/modal_app.py:26`

---

## 2. Dataset pipeline numbers (dataset explainer)

Sources & raw candidate counts by family (registry, counted from `data/raw_registry/provenance.jsonl`, total = 7,760):
- PPR10K (`ppr10k_derived`): 4,055 — source: provenance.jsonl (counted)
- Scraped web (`scraped_web`): 1,695 — source: provenance.jsonl (counted)
- FiveK (`fivek_derived`): 800 — source: provenance.jsonl (counted)
- FreshLUTs (`fresh_luts`): 727 — source: provenance.jsonl (counted)
- G'MIC/RawTherapee HaldCLUT (`gmic_rawtherapee`; site labels "HaldCLUT film"): 344 — source: provenance.jsonl (counted)
- Smaller public packs (`smaller_public_packs`; site labels "ON1 pack"): 89 — source: provenance.jsonl (counted)
- Procedural (`controlled_procedural`): 50 — source: provenance.jsonl (counted)
- Total raw candidates sourced: 7,760 — source: `data/run_summary.json` (`stages.2_acquire.raw_rows`); `derivation_attrition.json` (`candidates`)

Dedup (scraped-web only): 3,321 attempted → 2,866 distinct hashes (= 455 exact duplicates dropped) — source: `data/raw_registry/acquisition_report.json` (`attempted:3321`, note "2866 total distinct hashes"). NOTE the note also records "790 unparsable (.3dl/.look)"; only 1,695 scraped_web rows were ultimately registered (provenance count above).

Funnel:
- Sourced (candidates): 7,760 — source: `derivation_attrition.json` (`candidates`)
- Derived to a LUT: 7,741 — source: `derivation_attrition.json` (`derived`); `data/run_summary.json` (`4_5_derive_filter.derived`)
- Canonicalized: 7,741 — source: `derivation_attrition.json` (`canonicalized`)
- Passed quality gates (accepted = gold + diagnostic): 4,620 — source: `frozen_split.json` (`accepted_candidates.accepted:4620`); = 204 + 4,416
- Active set for training: 3,033 — source: `data/active_sft/active_rows.jsonl` (counted: 3,033 lines)

Canonical grid: 17³ = 4,913 nodes per LUT — source: `docs/model_architecture.md:161-168` (17x17x17); `docs/eval_harness_implementation.md:245-248`

Tier counts (representability/quality gate):
- Gold: 204 — source: `derivation_attrition.json` (`gold`); `data/run_summary.json`
- Diagnostic-only: 4,416 — source: `derivation_attrition.json` (`diagnostic_only`)
- Rejected: 3,140 — source: `derivation_attrition.json` (`rejected`)

Gate thresholds (as displayed on site; all confirmed):
- Gold mean ΔE00 ≤ 2.5; accept mean ΔE00 ≤ 3.0 — source: `docs/model_architecture.md:301-302` (per-target ≤3.0 / ≤6.0 p95), `docs/eval_harness_implementation.md:328-331` (target fidelity ≤3.0/≤8.0); tokenizer gold gates `docs/model_architecture.md:296-300`
- Pixel support ≥ 0.98 — source: `data/active_sft/active_manifest_prereaudit.json` acceptance detail is "support" gate; site value; also `docs/model_architecture.md:60` (`supported_cell_rate` 0.99 example). (Threshold value 0.98 shown on site; treat as the pipeline gate.)
- Clip rate ≤ 0.5% — source: `docs/eval_harness_implementation.md:398` (Clip rate ≤ 0.5% sampled channels)
- Neutral drift ΔE00 ≤ 3.0 — source: `docs/eval_harness_implementation.md:402`
- Skin hue drift p95 ≤ 8° — source: `docs/eval_harness_implementation.md:422` (`skin_locus_hue_drift_deg_p95 <= 8`)

Split (leakage-safe):
- Split id: `split_43e6cd41ca0ee35e` (site abbreviates `split_43e6cd41`) — source: `frozen_split.json`; `data/run_summary.json`
- Split units: 2,804 — source: `frozen_split.json` (`unit_count`); `data/run_summary.json` (`6_splits_leakage.unit_count`)
- Leakage status: pass — source: `data/splits/frozen/split_43e6cd41ca0ee35e/leakage_report.json` (`status:"pass"`); `run_summary.json`
- Split ratios: train 0.80 / eval 0.10 / diagnostic 0.07 / qualitative 0.03 — source: `frozen_split.json` (`ratios`)
- Split unit distribution: train 3,810 / eval 394 / diagnostic 267 / qualitative 130 — source: `frozen_split.json` (`split_distribution`)
- Policy version / seed: `frozen_v1`, seed 1234 — source: `frozen_split.json` (`leakage_policy_version`, `seed`)

Active set totals:
- Supported: 2,761 (91.03%) — source: `active_rows.jsonl` (counted `is_supported=true` = 2,761); 2761/3033 = 91.03%
- Unsupported (refuse): 272 (8.97%) — source: `active_rows.jsonl` (counted `is_supported=false`/`route=refuse` = 272)
- Total: 3,033 — source: `active_rows.jsonl` (3,033 lines)
- Supported by family (counted from `active_rows.jsonl`, supported only): PPR10K 956 · scraped_web 799 · FiveK 538 · FreshLUTs 197 · gmic_rawtherapee 191 · smaller_public_packs 41 · controlled_procedural 39 — source: `active_rows.jsonl` (counted). These match the site's STAGE-06 family bars and the selection-quota panel EXACTLY.
- token_status: 2,761 `materialized`, 272 `not_applicable` — source: `active_rows.jsonl` (counted)
- Unsupported corpus composition (272 train rows): out_of_scope (11 categories) + out_of_gamut (3) + mixed prompts — source: `HANDOFF.md:79-81` (OUT_OF_SCOPE 11, OUT_OF_GAMUT 3, mixed 6 buckets); `data/active_sft/unsupported_gen_manifest.json` (`train_rows:272`, `by_category`)
- 5 caption "voices" (literal/metaphor/mood/concept/slang) — source: `docs/interpreter_results.md:79` ; `HANDOFF.md:198`

Eval reserve (NOT frozen; `eval_set_pending_freeze`):
- Headline-eligible supported (usage-weighted headline): 382 — source: `data/eval/eval_manifest.json` (`headline_eligible_count:382`, `usage_weighted_headline_supported:382`); `run_summary.json`
- Targets (aspirational): 800 supported / 200 unsupported / 100 qualitative — source: `eval_manifest.json` (`sizes_target`); `docs/eval_harness_implementation.md:488-492`

Shards / delivery:
- Corpus staged to `hf://datasets/ericrcwu/LUT_SLM` as sha256-verified tar shards; ~9.85 GB total; luts/raw is ~9 GB of JPGs — source: `docs/collapse_fix/README.md:80-81` ("~9.85 GB corpus"); `configs/staging_default.yaml` (pack include dirs; "luts/raw is the bulk (~9 GB of JPGs)"). NOTE: the site's exact 5 shard NAMES (`corpus-raw.tar`, `corpus-canonical.tar`, `corpus-active-sft.tar`, `corpus-eval-reserve.tar`, `corpus-tokenizer.tar`) and the "5 shards" count are NOT verifiable from local files (shards live on HF). `README.md:41` only says "packed as `corpus-*.tar` shards". State "~9.85 GB, sha256-verified tar shards on a private HF dataset"; do NOT assert the specific 5 filenames as verified.

### MISMATCHES to fix (dataset_pipeline_explainer.html)

1. **Acceptance "10 OF 12 PASS" (lines ~1194-1209) is NOT verifiable from the repo, and the ONLY local manifest contradicts it.** The sole local manifest, `data/active_sft/active_manifest_prereaudit.json`, reports `acceptance.overall = "fail"` with 6 pass / 3 FAIL / 3 pending:
   - FAIL: `tags_backed_by_checks` ("200 tag/behavior mismatches"), `unmentioned_behavior_handled`, `expert_source_capped` ("ppr10k+fivek fraction=0.54 vs cap 0.50").
   - PENDING: `representability_and_recon`, `unsupported_coverage`, `generic_input_support`.
   The site instead shows those three FAILs as PASS and lists the two pending as "Reconstruction check" + "Paired-image support". Auto-memory says a post-audit manifest with `acceptance overall=PASS` exists on HF (not in the repo). ACTION: verify the "10 of 12" claim against the CURRENT HF `active_manifest.json` before publishing; the local file it can be checked against says otherwise.
2. **Expert-source cap "49%, under the 50% cap" (line ~1190) depends on the denominator.** (PPR10K 956 + FiveK 538)/3,033 = 49.3% (site's figure, uses full active set). But `active_manifest_prereaudit.json` computes it over *supported* rows and reports 0.54 → FAIL. State the cap result only if the current HF manifest confirms pass; note the denominator.
3. **"Largest single family is 32%" (line ~1190):** 956/3,033 = 31.5% ≈ 32% (of the full active set). `active_manifest_prereaudit.json` `no_dominance` reports `max_family_fraction=0.34` (over supported rows, 956/2,777). Both are < 50%; just a denominator difference — keep "32% of the set" only if measured against 3,033.
- Everything else on the page (7,760 sourced; per-source raw counts; 3,321→2,866 dedup; 7,741 derived; 204/4,416/3,140 tiers; 4,620 survivors; 2,804 split units; split id; 2,761/272/3,033 active set; all supported family bars; 17³=4,913; tokenizer 1.31 ΔE00 / ~37 dB) is ACCURATE against the sources.

---

## 3. Eval harness facts (evaluation-harness page)

Nine deterministic layers L0–L8 — source: `docs/eval_harness_implementation.md:140-151`:
- L0 Boundary — gold unsupported passes only with exact `<unsupported>`; gold supported fails if refused (route grade/clarify/refuse under ADR 0024, `docs/eval_harness_implementation.md:591-598`)
- L1 Syntax — supported output = BOS + exactly 64 valid LUT tokens + EOS
- L2 Decode/export — 64 tokens decode to finite canonical 17³ residual LUT and export a valid `.cube`
- L3 Tokenizer gate — frozen tokenizer passes mean/tail/per-family/per-target reconstruction gates
- L4 Direction — every gold tag moves in the correct measured direction + minimum magnitude
- L5 Target fidelity — acceptance_mode gate: exact single target, any of K references, or behavior window
- L6 LUT safety — clip, out-of-range, smoothness, foldover, neutral drift, skin-locus
- L7 Style recipe — style-recipe windows + discriminability
- L8 Judge — LLM/VLM score recorded; CANNOT override L0–L7
- Supported pass = boundary ∧ syntax ∧ decode ∧ direction ∧ fidelity ∧ safety ∧ style — source: `docs/eval_harness_implementation.md:154-172`. Unsupported pass = exact `<unsupported>` — source: `:181-183`
- v1 spine reality: L2 + L4–L7 (decode-dependent) are DISABLED / decoder-free proxies in the current harness; running now = L0, L1, unsupported/boundary metrics, stats — source: `README.md:8-25`; site "decoder-free v1 spine"

Primary headline metric: `supported_prompt_to_lut_pass_rate` on headline-eligible rows — source: `docs/eval_harness_implementation.md:10-15`; `CONTEXT.md:123-124`

Ship-gate thresholds (all confirmed against `docs/eval_harness_implementation.md:881-904`):
- Supported pass rate: Wilson 95% lower ≥ 60% (`:892`)
- Free-generation valid-token rate: Wilson lower ≥ 85% (`:884`)
- Unsupported recall ≥ 80%, precision ≥ 80% (`:885-886`)
- Boundary F1 ≥ 80%, mixed_unsupported_recall ≥ 80% (`:887-888`)
- Near-boundary pair accuracy ≥ 85% (`:889`)
- Over-refusal: Wilson upper ≤ 10% (`:890`)
- Safety failure: Wilson upper ≤ 5% (`:891`, v1 absolute gate)
- vs deterministic renderer on headline: paired lower ≥ +10pp (v1 rebind from +5pp) (`:895`, `:906-912`)
- GRPO ship gate (if run): paired lower ≥ +5pp pass_rate OR ≤ −5pp safety, plus guardrails — source: `:948-963`

Eval set sizes:
- Realized headline-eligible supported reserve: ~382 — source: `eval_manifest.json` (`headline_eligible_count:382`)
- v1 headline floor: min_N = 350 (because reserve ~382), +10pp claim — source: `docs/eval_harness_implementation.md:497-503`, `:906-922`
- Aspirational: 800 supported / 200 unsupported / 100 qualitative — source: `:488-492`
- Reconciled frozen budget (aspirational): 1,300 supported / 300 unsupported / 100 qualitative = 1,700 rows — source: `:526-530`

CI / statistics method:
- Single-rate CI: Wilson 95% — source: `:712-714`
- Paired delta: stratified paired bootstrap, B ≥ 10,000 — source: `:719-726`
- Paired binary tests: McNemar / exact paired permutation — source: `:731-733`
- Multiplicity: Holm–Bonferroni, family_alpha 0.05 — source: `:775-784`
- Seeds: 3 for final reporting — source: `:736-741`

### MISMATCHES to fix (evaluation-harness.html)

1. **Version-manifest table (line ~1980) lists the Interpreter as "LFM2.5-350M + revision".** The actually built/deployed interpreter is `Qwen/Qwen2.5-0.5B-Instruct` (`docs/interpreter_results.md:5`; `deploy/modal_app.py:134`). `LFM2.5-350M-Base` is only the architecture-doc default. FIX to Qwen2.5-0.5B-Instruct (or explicitly mark it as the spec default vs the built model).
2. **"67 tokens" output budget / "67 max new tokens" (hero + row-meta):** the exact supported grammar is BOS + 64 codes + EOS = 66 tokens (`docs/model_architecture.md:126`); generation budgets in code are `max_new_tokens=68` (`sft/generate.py`, per `docs/collapse_fix/README.md:109`) and 64 for the interpreter (`deploy/modal_app.py:135`). "67" is a rounded illustrative value — acceptable if the page keeps its "illustrative" labeling, but it is not an exact spec number.
- The page correctly labels its CI numbers (36.2 / 46.8 / 64.7%, +12.1pp) and pass verdicts as "illustrative" — these are intentionally synthetic; do NOT replace them with (nonexistent) real results. Everything factual otherwise (9 layers, all thresholds, min_N 350, reserve ~382, 3 seeds, 10k bootstrap, Wilson, Holm, decoder-free v1 spine) is ACCURATE.

---

## 4. Frontier SOTA results (authoritative — dashboard)

Model ids + list prices ($/1M input/output) — source: `docs/frontier_baseline_results.md:12-18`:
- Opus 4.8 — `claude-group/claude-opus-4-8` — $5 / $25 — effort medium — ~1,277 input tok
- Sonnet 4.6 — `claude-group/claude-sonnet-4-6` — $3 / $15 — effort medium — ~971 tok
- GPT-5.5 — `openai-group/gpt-5.5` — $5 / $30 — effort medium AND xhigh — ~849 tok
- Gemini 3.1 Pro — `gemini-group/gemini-3.1-pro` — $2 / $12 — no effort control — ~1,908 tok

Documented headline valid rate — 5 single-attribute rows (r1 warmer, r2 cooler, r3 more_magenta, r4 more_green, r5 brighter) — source: `docs/frontier_baseline_results.md:23-37`:
- Opus 4.8: **4/5** valid (passed r1, r3, r4, r5; failed r2). Main-grid cost $9.76, ~51m
- Sonnet 4.6: **3/5** (passed r2, r3, r4; failed r1, r5). $7.71, ~70m
- GPT-5.5 xhigh: **2/5** (passed r2, r3; failed r1, r4, r5). $9.05, ~16m
- GPT-5.5 medium: **0/5** (refused r2–r5, failed r1). $0.48, ~3m
- Gemini 3.1 Pro: **0/5** (all truncated/invalid). $3.22, ~29m
- Main-grid grand cost ≈ $30.22 — source: `:37`

Per-valid-LUT economics: a usable frontier LUT costs ~$1.4–2.4 and ~8–17 min, best model lands ≤ 80% of the time — source: `docs/frontier_baseline_results.md:106`. Example cells: Opus r1 $2.41 / 766s; Opus r3 $2.25 / 714s — source: `:30`.

Direction/safety on COMPLETED LUTs = 100% (pass/pass) — every one of the 9 completed LUTs across models passed direction + safety — source: `docs/frontier_baseline_results.md:39-53,93`. "Color intent is never the failure — completion is."

Failure modes & findings — source: `docs/frontier_baseline_results.md:91-100`:
- Token-ceiling / truncation is the dominant supported-behavior failure (not wrong color).
- Gemini 3.1 Pro is structurally blocked: its route hard-caps output at ~65K tokens (~2,000 rows); a full ~100K-token LUT is impossible regardless of prompt (`:96`).
- Effort gates GPT-5.5: medium refuses/bails (0/5); xhigh engages (2/5) at ~15–20 min/row.
- Named styles are the worst case: **0/3** (sepia truncated for Opus, Sonnet, GPT-5.5 xhigh) — source: `:69-76,98`.
- Composites work: **3/3** ("warmer + softer contrast" all pass) — source: `:59-66,97`.
- Boundary/refusal is the models' strength: **3/3** correctly refused the mixed trap ("warmer AND remove background") for ~$0.01 — source: `:79-86,99`.

Fresh re-score (`playground/_shared/frontier_scores_fresh.csv`, 70-row smoke; valid rates LOW because each model only attempted 5–8 of 70 rows — the documented per-attempt results above are the authoritative headline):
- opus_4_8: raw_cube_valid_rate 0.0714 (CI 0.031–0.157); direction_pass_rate 1.0 (N=5); safety_pass_rate 1.0 (N=5); lut_quality 1.0 (N=5); boundary_accuracy 0.7286 — source: frontier_scores_fresh.csv row `opus_4_8`
- sonnet_4_6: valid 0.0571; direction 1.0 (N=4); safety 1.0 (N=4); lut_quality 1.0 (N=4); boundary_accuracy 0.7286 — source: csv row `sonnet_4_6`
- gpt_5_5_xhigh: valid 0.0429; direction 1.0 (N=3); safety 1.0 (N=3); lut_quality 1.0 (N=3); boundary_accuracy 0.7286 — source: csv row `gpt_5_5_xhigh`
- gpt_5_5 (medium): valid 0.0; direction N=0; over_refusal 0.08; boundary_accuracy 0.6571 — source: csv row `gpt_5_5`
- gemini_3_1_pro: valid 0.0; direction N=0; over_refusal 0.0; boundary_accuracy 0.7143 — source: csv row `gemini_3_1_pro`
- Headline for fresh re-score: on the completed LUTs, direction = 1.0 and safety = 1.0 for opus (N=5), sonnet (N=4), gpt_5_5_xhigh (N=3) — target fidelity `not_evaluated:no_target_luts_in_frozen_eval_set`.

---

## 5. MY MODEL's measured metrics (dashboard — CONSERVATIVE)

Structural output contract (the reliability-by-construction claim):
- The generator emits exactly `<lut_bos>` + 64 code tokens + `<lut_eos>` (or `<unsupported>`); the 64 codes go through the FROZEN VQ decoder → a canonical 17³ residual LUT → always a valid canonical 17³ `.cube` (4,913 rows). Grammar-constrained decoding makes runtime syntax validity 100% by construction. — source: `docs/model_architecture.md:14-36,122-151`; `docs/eval_harness_implementation.md:185-224` (constrained decode; `syntax_valid_rate == 100%` in constrained mode). This is the "100%-valid-cube-by-construction" advantage.
- Output budget: 64 VQ tokens (256-entry codebook) vs ~100K raw-float tokens / 4,913-row `.cube` — source: `docs/model_architecture.md:234-242`; `docs/frontier_baseline_results.md:96,106` ("64 VQ tokens vs ~100K raw floats").

Tokenizer reconstruction (frozen VQ, on the 382-LUT dev holdout) — source: `tokenizer/final/manifest.json` `gate_report.overall`:
- mean ΔE00 = 1.31 (1.3076); p95 ΔE00 = 3.34; p99 ΔE00 = 5.00; max ΔE00 = 9.85
- mean PSNR = 37.03 dB; p5 PSNR = 28.37 dB (below the ≥30 dB gate — a signed-off reviewed exception; on the gold subset p5 PSNR = 31.24 ≥ 30) — source: `manifest.json` `reviewed_exception`
- codebook: 256/256 active codes (100%), 0 dead, perplexity 164.2 — source: `manifest.json` `gate_report.codebook`
- tokenizer_version `vq_v2_srgbres_17to4_cb256_t64__w91cffdd2c82f`; vq_codebook_sha256 `bcdf369d…`; vq_decoder_sha256 `4372d4a8…` — source: `manifest.json`

Interpreter / router (separate Qwen2.5-0.5B-Instruct, full run over 2,761 LUTs, leakage-safe holdout n=684) — source: `docs/interpreter_results.md:56-77`:
- route_accuracy = 0.884 (CI 0.858–0.906); refuse recall/kind = 1.0/1.0; clarify recall = 1.0 (n=40); grade recall = 0.868; over-refusal = 0.132; parse_ok = 0.886
- Grade-axis "ladder" (full run, real LUTs): direction F1 = 0.468; bucket F1 = 0.159; exact `attribute_f1` = 0.112. Finding: direction is learnable, magnitude is NOT (task underdetermination) → interpreter ships as a ROUTER only. — source: `docs/interpreter_results.md:69-92`

Generator token accuracy (teacher-forced, unit-aware holdout — a decoder-free proxy, NOT a harness pass rate):
- One-stage baseline (instruction input): 0.362, CI [0.337, 0.387] — source: `HANDOFF.md:234,248,320`
- Two-stage P6 (attribute_spec_text input): 0.329, CI [0.302, 0.355] — source: `HANDOFF.md:319-320`. Gate not cleanly cleared (−3.3pp; CIs overlap; comparison confounded by different training corpora + refuse-row cost) — source: `HANDOFF.md:318-342`
- Behavior_v2 seam is information-lossless (0 lossy collisions; a perfect spec-mapper ceiling = 1.0) — source: `HANDOFF.md:253-267`

Behavioral fidelity / free-running collapse (P6 adapter `p6_twostage_d0f9c744_smokefull`, 64-row slice of the 120-row holdout, conditioned on attribute_spec_text) — source: `docs/collapse_fix/README.md:15-20`:
- teacher-forced argmax: behavioral fidelity 0.708, collapse 0% (OPTIMISTIC ceiling — target codes are the answer; do NOT present as "the model understands the spec")
- free-running greedy: 0.159, collapse 94% (exposure bias — this is the real deployed-greedy number)
- free-running sample t=0.7: 0.091, collapse 14%
- ruler's own ceiling (real corpus codes): ~0.89
- Diagnosis: exposure bias, NOT a broken seam or architecture problem — source: `docs/collapse_fix/README.md:24-26`

Oracle@N / best-of-N (inference-time mitigation, measured):
- oracle@32 ≈ 0.42 (sampling covers good trajectories) — source: `docs/grpo/00_grounding.md:18`; `docs/grpo/05_eval_and_gates.md:32`; `docs/grpo/01_reward.md:17`. CAVEAT: an earlier/smaller-slice figure of 0.30 appears in `docs/collapse_fix/corpus_audit.md:12` and a stale in-code note (`eval/best_of_n.py:6`). Prefer ≈0.42 (the canonical, re-measured figure); if stating a single number, note it is on the ~64-row holdout slice.
- best-of-N sample+rerank (deployable today): ≈ 0.42 behavioral fidelity — source: `docs/grpo/05_eval_and_gates.md:31`; `docs/grpo/00_grounding.md:18`
- GRPO is a PLAN to lift greedy toward 0.42; no shipped GRPO result — source: `docs/grpo/*` (plans); `HANDOFF.md` (GRPO not run)

Cost / latency / determinism / local:
- Deterministic by design: `do_sample=false`, `num_beams=1`, fixed seed; bit-identical tokens/`.cube` under the same manifest+hardware — source: `docs/model_architecture.md:508-530`. TRUE, state it.
- Local / offline: runs as a local CLI and on a single T4 (Modal, scale-to-zero) — source: `deploy/modal_app.py:205-231`; `docs/model_architecture.md:479-503`. TRUE.
- "sub-second, ~$0.00x per LUT": this is the PROJECT'S design/comparison CLAIM in `docs/frontier_baseline_results.md:106`, NOT a benchmarked latency/cost measurement. Present as a design target/comparison, not a measured number.

**CRITICAL — do NOT state:**
- End-to-end `supported_prompt_to_lut_pass_rate` for my model: **NOT FOUND — do not state.** The decode-dependent harness layers (L2, L4–L7) are disabled in the current spine; only token-accuracy + behavioral-fidelity proxies exist. There is NO end-to-end harness pass rate yet.
- Any headline quality gate PASS for my model vs baselines: **NOT FOUND — do not state.** Gates are declared, not yet met/run at scale.
- A shipped GRPO result: **NOT FOUND — do not state.**

---

## 6. One-paragraph honest comparison narrative

Frontier models *can* produce good global-color LUTs when they finish one — on this
probe every completed LUT was direction-correct and safe (documented 9/9 pass/pass;
fresh re-score direction=1.0, safety=1.0 for the LUTs Opus/Sonnet/GPT-5.5-xhigh
actually completed). Their real limit is not color taste but *completion*: emitting a
full 4,913-row `.cube` runs into the models' output token ceiling, so the best model
(Opus 4.8) lands only 4/5 single-attribute rows, named styles fail 0/3, Gemini 3.1 Pro
is hard-capped at ~65K output tokens, and a usable LUT costs roughly $1.4–2.4 and
8–17 minutes. This project targets the same task from the other direction: a small
local model emits 64 VQ tokens that a frozen decoder turns into a canonical 17³ `.cube`
*by construction*, so the artifact is always a valid, safe-domain LUT and generation is
deterministic, local, and cheap — trading the frontier's occasional high-quality-but-
truncated output for guaranteed-valid, tractable output. What is NOT yet proven is the
model's end-to-end color quality: the frozen tokenizer reconstructs held-out LUTs at
mean ΔE00 1.31 / ~37 dB, and the router is production-ready (route acc 0.884, refuse/
clarify recall 1.0), but the generator currently collapses free-running greedy
(behavioral fidelity 0.159 vs a 0.42 best-of-N/oracle ceiling and a 0.708 teacher-forced
proxy), and there is no end-to-end harness pass rate yet. Honest framing:
reliability-and-tractability-by-construction plus cost/latency/local/deterministic
advantages that are true today — not a claim of beating frontier color quality.
