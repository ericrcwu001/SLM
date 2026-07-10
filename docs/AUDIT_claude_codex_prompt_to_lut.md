# Consolidated Read-Only Audit — Prompt-to-LUT Training System

**Two independent audits (Claude multi-agent + Codex), reconciled.** Read-only: no files were modified, no dataset/tokenizer changed, no training started, the LUT decoder was never enabled.

- **Claude audit:** seven parallel investigation agents (collection, dataset construction, instruction generation, VQ tokenizer, fine-tuning, architecture, evaluation) + one adversarial red-team pass; load-bearing claims re-verified against code/data, including read-only encode→decode probes on the frozen tokenizer.
- **Codex audit:** single-pass end-to-end trace with read-only corpus statistics and an adversarial critique.

**Attribution & confidence conventions used below:**
- **[Both]** — independently found by both audits (highest confidence).
- **[Claude]** / **[Codex]** — contributed by one audit; corroborated where noted.
- **[VERIFIED]** — confirmed in code/data. **[HYPOTHESIS]** — plausible, not yet proven; needs an experiment.

---

## 0. The single most important framing (adopted from Codex; Claude concurs)

The repository **confirms a real diversity/semantics problem**, but it **does not yet prove that corpus diversity is the *dominant* cause** of "turn it red / make it Mars" failures. The diagnosis is **confounded** by problems that are established facts:

1. The evaluation is **leaked, source-biased, teacher-forced, and in-distribution** — it cannot see OOD at all.
2. SFT supervision **suppresses semantic language** and **ignores its richer prompt field**.
3. The shipped adapter received **zero effective refusal training** and several declared module policies are **not implemented**.

So the most defensible current statement is:

> Strong LUT behaviors exist in the corpus, but **language is not bound to reusable color attributes**, and direct prompt→code training encourages association with dense style bundles rather than compositional reasoning. Whether this — versus semantic drift, image-blindness, target ambiguity, or the representation ceiling — is the *dominant* cause is a **hypothesis that the current metric cannot test.**

A crucial corollary (Codex, elevated): "Mars" was almost certainly known to pretrained Qwen. Fine-tuning may have **failed to connect** that knowledge to LUT attributes — **or may have partially damaged it** via full embedding/head updates. These are different diseases with different cures, and today's eval can't distinguish them.

**Therefore the first work item is not "collect red LUTs." It is to make the ruler honest and OOD-aware, and to train + measure the refusal path — so every later change is steerable.**

---

## 1. Executive summary

The "make it red / make it Mars" failure decomposes into **three stacked problems**:

1. **Language ceiling (partly fixable).** The instruction ontology collapses the whole hue circle onto two Lab axes — `temperature_delta_b` (warm/cool) and `tint_delta_a` (magenta/green). A teacher LLM is hard-restricted to a **22-tag vocabulary** (21 realized in data), and a deterministic gate rejects any tag not backed by measured behavior. `red`/`orange`/`purple`/`cyan` and all metaphors are absent from supervision. *(Nuance: `green` and `magenta` — the tint poles — appear ~800/~700 times in the trained text, so the model does learn two color names; it's the rest of the hue circle + metaphors that are missing.)* **[Both, VERIFIED]**

2. **Representation boundary (hard blocker for the literal extreme tail).** The frozen VQ tokenizer round-trips strong warm/cool casts, heavy desaturation, and teal-orange at corpus quality, but **cannot** represent pure primary-hue casts, 120° hue rotations, or infrared channel-swaps. This is a **distribution-coverage boundary, not an absolute capacity ceiling.** **[Both]** — Claude via read-only probes (ΔE 13–16); Codex via the manifest's waived p5-PSNR and worst-tail composition.

3. **Measurement blindness (the highest-leverage problem).** The optimized metric is an **in-distribution, teacher-forced, exact-match token accuracy on a leaked holdout** (48.5% of held-out rows share a near-duplicate LUT unit with training), scored on a **source-biased first-48 slice**, and the eval harness **never scores LUT quality at all.** **[Both, VERIFIED]**

**The reframe both audits converge on:** by `CONTEXT.md`'s own definitions, "turn it red" (out-of-gamut, no measurable backing axis) and "make it Mars" (semantic metaphor) are arguably **out of scope** — that is what the refusal path is for. But the system **does not cleanly refuse them either** (refusal is untrained and unmeasured), so it **silently emits a conventional muted LUT.** The honest in-scope defect is: *the model produces a wrong, conventional LUT for requests it should decompose or refuse, and the metric cannot see it.*

---

## 2. Consensus findings — cross-confirmed by both audits (highest confidence)

| # | Finding | Evidence | Both? |
|---|---|---|---|
| A | **Zero refusal supervision reached the adapter.** All 272 unsupported rows use absolute `/Users/...` image paths; `resolve_image` leaves absolute paths unchanged; training skips unloadable rows. Overnight report: 5,184 trained (2,592×2 epochs) + **544 skips (272×2)** = zero `<unsupported>` targets seen. | `example.py:26`, `train.py:147-153`, `OVERNIGHT_SFT_REPORT.md:32`, AGENTS.md:47 | **[Both, VERIFIED]** |
| B | **SFT holdout is contaminated.** `sft/holdout.py` hashes row id and ignores the pipeline's `split_unit_id`. 169 holdout rows → **82/169 (48.5%) share a split unit with train**, 47 units cross, **28/169 exact instruction match**, **3/169 exact 64-code target**. | `holdout.py:21`, `splits.py:60` | **[Both, VERIFIED]** (replicated by both) |
| C | **The metric is narrow & deployment-misaligned.** Teacher-forced argmax over code positions only; default scores **first 48 rows (34 PPR10K + 14 G'MIC; 5 families absent)**. A context-only teacher-forced bigram already reaches ~0.21–0.24 = **~58% of the 0.4137**. No free-generation, grammar, length, refusal, OOD, or perceptual measurement. | `score_tokens.py:75-99` | **[Both, VERIFIED]** |
| D | **SFT ignores the richer language it already has.** Training consumes only `row["instruction"]` (concise; **182 word types**), discarding `instruction_natural` (**548 word types**), `gold_tags`, and `measured_behavior` — contradicting the team's own design doc, which says SFT should see the natural prompt. | `example.py:76`, `lut_methodology_improvement_plan.md:414` | **[Both, VERIFIED]** |
| E | **Teacher vocabulary is a closed ~22-tag set** built for literal behavior-grounded edits, not metaphor/slang/concepts. `red`/`orange`/`mars` ≈ 0 in both concise and natural. | `instruction_gen.py:54-126,470-497` | **[Both, VERIFIED]** |
| F | **Data gates cull the bold tail.** 7,760 raw → 204 gold / 4,416 diagnostic / **3,140 rejected (40.5%)**. `clip_rate_max=0.005` + foldover are the dominant killers; clip-rejected rows have ~5–7× the chroma of gold. Skin-locus only *demotes* to diagnostic. (Neutral-drift is NOT the culprit — the `_tinted` bypass makes it fire on only 4 rows.) | `quality_filters.py:20-28`, `run_summary.json:4`, `run_pipeline.py:107-119` | **[Both, VERIFIED]** |
| G | **Image conditioning is weak and confounded.** 100% of images are PPR10K/FiveK; **1,267/2,761 (45.9%) are generic pairings whose LUT is explicitly independent of the image**; no same-prompt/different-image counterfactuals. | `pair_generic_images.py:1-13`, `active_manifest.json:74` | **[Both, VERIFIED]** |
| H | **Selection doesn't implement its documented design.** The active embedding is 9 **unnormalized** behavior scalars + tag multi-hot (residual-PCA/image/prompt embeddings unused). **7,710/7,760 rows lack a usage bucket** → default `common_head`; pre-audit selection = 2,755 common-head / 12 common-style / 10 subtle-control / **0 coverage-tail**. PPR10K = **34.6%** despite a 25% cap (cap denominator is the 3,826-row pool, not the final set). | `embeddings.py:15`, `run_pipeline.py:297-305`, `selection.py:120`, `active_manifest_prereaudit.json:14` | **[Both, VERIFIED]** |
| I | **Leakage "pass" covers LUT space only.** The frozen split's active axes are normalized-LUT equality + PCA-64 LUT-neighbor; image and prompt semantic axes were **skipped (no embeddings)**, and the split is built **before** instruction-gen and generic image pairing. | `frozen_split.json:24`, `leakage_report.json:1` | **[Both, VERIFIED]** |
| J | **Dead/undeclared SFT policies.** `freeze_vision_encoder`, `projector_policy`, `learning_rate_projector` are declared but **never applied**; `modules_to_save=["embed_tokens","lm_head"]` trains **full matrices** (old-vocab drift risk, warned in the arch doc). | `config.py:39`, `train.py:103-119`, `model_architecture.md:350` | **[Both, VERIFIED]** |
| K | **Frozen tokenizer has a real extreme-tail boundary (not codebook collapse).** 256/256 codes active, perplexity 164, top-code 3.46%. Raw **p5 PSNR 28.37 < 30 (waived)**; worst tail = faux-infrared, green-mono, solarized — exactly the unusual transforms an expansion would target. | `manifest.json:84,161` | **[Both, VERIFIED]** |

---

## 3. Complete data & training pipeline

```
Source connectors (8 families)            data_pipeline/acquire/run_acquire.py
  ↓  RawArtifact{LUT, image-pair, tags, optional authored instruction}  — NO caption field
Registry / provenance                     registry.py   (to_registry_row DROPS `extra` → titles lost)
  ↓
LUT parse OR global fit from pairs         run_pipeline.py:_derive_lut
  ↓
Canonicalize → sRGB 17³ (HARD-CLAMP [0,1]) canonicalize.py:83  (destroys wide-gamut headroom)
  ↓  ~29-field behavior vector (hue only as a*=tint, b*=temperature, +chroma)  behavior_vector.py
Gates: clip≤.005 · foldover≤.005 · neutral_drift≤3 (bypassed if tinted) · skin(DEMOTE) · smoothness
  ↓  204 gold / 4,416 diagnostic / 3,140 rejected
Split units (union shared id + LUT near-neighbors)   splits.py:60   → train/eval/diag/qual
  ↓
MMR selection (9 unnormalized behavior scalars + tag bits; caps by family)  selection.py
  ↓
Teacher (claude-sonnet-4-6): {gold_tags, concise, natural}   instruction_gen.py
  ↓  deterministic validate_tags_against_behavior (tags↔behavior; NEVER reads prose)
Generic image pairing for LUT-only rows (LUT ⟂ image)   pair_generic_images.py
  ↓
Frozen VQ tokenizer → 64 tokens / 256-codebook   materialize_target_tokens.py (admission mean≤3/p95≤6, else writes nothing)
  ↓
Qwen2.5-VL-3B QLoRA SFT (image + concise only; +259 tokens; epochs=2)   sft/train.py
  ↓
METRIC = teacher-forced code-token accuracy on an id-hash holdout   sft/score_tokens.py
EVAL: L0 syntax + refusal-boundary ONLY; L2–L8 = not_evaluated:decoder_disabled   eval/run_eval.py
```

**Corpus snapshot [Both, VERIFIED]:**

| Family | Raw | Accepted | Active supported |
|---|---:|---:|---:|
| PPR10K | 4,055 | 2,390 | 956 |
| Scraped web | 1,695 | 1,024 | 799 |
| FiveK | 800 | 609 | 538 |
| FreshLUTs | 727 | 270 | 197 |
| G'MIC/RawTherapee | 344 | 233 | 191 |
| Smaller packs | 89 | 52 | 41 |
| Procedural | 50 | 42 | 39 |
| **Total** | **7,760** | **4,620** | **2,761** (+272 unsupported) |

Expert photo (PPR10K+FiveK) = **1,494 = 54%** of supported (49% of the 3,033 total — the cap "passes" only by diluting with unsupported rows).

---

## 4. Dataset diversity assessment

- **Supply skews conventional & subtle.** 54% expert photo (~99% subtle by pair-fit construction); the most *creative* family (G'MIC/RawTherapee film sims) starved at 6.9% vs a 20–25% target. **[Both]**
- **Tag distribution is heavily "flatten and mute":** `muted` 1674, `less_contrast` 1641, `brighter` 1475, … tail `more_saturated` 240, `cinematic` 128, `filmic` 10, `sepia`/`natural` 1. **No tag/style/cast balancing exists anywhere.** **[Both]**
- **Compositional mix is inverted vs the design.** 97% of supported rows carry **≥3 tags** — the opposite of the documented **50% simple / 30% compound** curriculum. So the model rarely sees clean single-attribute supervision it could compose from. **[Codex, VERIFIED]** (`lut_methodology_improvement_plan.md:387`)
- **Language is narrow where it counts.** Concise: 2,471 unique / **182 word types**; natural: 2,742 unique / **548 word types**; both ≈0 for red/orange/mars/teal/neon/dream. High uniqueness ≠ linguistic diversity (single-teacher idiolect — see §8). **[Both]**
- **Extreme looks exist but never reach headline supervision** and are relabeled into conventional tags (a B&W LUT with `chroma_delta≈64` → `muted`). 526 B&W/mono LUTs, faux-infrared, solarized — all diagnostic. **[Claude]**
- **Coverage-tail bucket got 0 rows** (H): the mechanism intended to guarantee rare-look coverage was a no-op. **[Codex]**

---

## 5. Failure modes & root causes

### Established facts [VERIFIED]

- **F1 — No semantic color/metaphor grounding in supervision.** Hue collapsed to 2 Lab axes; teacher locked to 22 tags; deterministic gate rejects unbacked tags; `red`=0 across 2,761 instructions. **[Both]**
- **F2 — Refusal never trained; the model fails *silently*, it does not refuse.** (Correction to an initial Claude hypothesis that "red → refuse": the 272 refusal rows never train because their absolute paths are skipped, so the model learns *nothing* about red and most likely emits a muted in-distribution LUT.) **[Both]**
- **F3 — Representation boundary blocks the literal extreme tail.** Read-only probes: pure-red 14.65, hue-rot-120 13.06, infrared 13.81, mono 2.70/p95 6.26 ΔE00; teal-orange 1.07 and warm 1.38 pass. Materialization writes nothing above mean 3.0/p95 6.0. **[Claude probes; Codex corroborates via waived p5-PSNR]**
- **F4 — The optimized metric is leaked (48.5%), in-distribution, teacher-forced, exact-match, and source-biased (first-48).** **[Both]**
- **F5 — The eval harness scores no LUT quality** (primary metric `not_evaluated:decoder_disabled`); no OOD/metaphor/paraphrase/regression slice exists. **[Both]**
- **F6 — Image conditioning is structurally weak** (dead projector knobs; 45.9% image-independent pairings). **[Both]**
- **F7 — Semantic metadata is discarded at ingestion.** FreshLUTs/ON1 titles live in `RawArtifact.extra`; `to_registry_row()` doesn't persist `extra`; scraped-web/pack tags are blanked before selection/teacher. So named-style language ("faux infrared", cinematic titles) that *is* exactly the vocabulary we want is thrown away while the LUT behavior survives. **[Codex, VERIFIED]** (`base.py:61`, `freshluts.py:237`, `on1_local.py:115`, `run_pipeline.py:50`)
- **F8 — Partial-truncation blind spot.** The truncation guard detects only *complete* loss of the assistant span, not partial loss of the 64 codes; the scorer accepts any positive count of surviving code positions. With `max_pixels`/`max_seq_len` near the limit, some of the 64 targets can be silently dropped. **[Codex, VERIFIED]** (`example.py:95`, `score_tokens.py:99`; cf. AGENTS.md max_pixels≤401408 truncation warning)

### Hypotheses requiring experiments [HYPOTHESIS] (mostly Codex; Claude concurs)

1. **Semantic binding is a bigger current bottleneck than behavior coverage** — many strong casts exist; the *language→attribute* link is missing.
2. **Single-teacher idiolect** — one Claude teacher + one Claude judge + Claude in the baseline creates correlated blind spots and a phrasing monoculture (also a home-field-advantage leak; see §8/§9).
3. **Unnormalized MMR favors large-magnitude behavior outliers.**
4. **Direct prompt/image → 64 categorical codes encourages bundle memorization** over composition.
5. **Full embedding/head training may have eroded pretrained metaphor/rare-word semantics** ("catastrophic semantic drift") — a distinct disease from "never learned the mapping," with a distinct cure (row-selective training).
6. **The model may have learned to ignore the image** because ~46% of targets are image-independent.
7. **The 4³ latent favors smooth, low-frequency global transforms** — consistent with the extreme tail failing, but perceptual causality can't be established without the (prohibited) decoder.

---

## 6. Coverage gaps (style / language / composition / evaluation)

| Dimension | Present | Absent |
|---|---|---|
| Color | warm/cool, magenta/green, chroma up/down | **red, orange, purple, pink, cyan, any hue angle** |
| Style | matte, faded, filmic, cinematic, sepia, bleach-bypass | monochrome/B&W *as a label*, cross-process, duotone, day-for-night, infrared, cyberpunk/vaporwave |
| Mood / metaphor / place | — | moody, dreamy, nostalgic, "Mars/sunset/underwater", seasons, decades |
| Language form | terse imperative templates (182 word types trained) | paraphrase, slang, questions, indirect; richer `natural` field discarded |
| Composition | 97% ≥3-tag bundles | clean single-attribute supervision; held-out & *tested* novel combinations |
| Negatives | 272 task-type refusals (never train) | extreme-but-global cases; hard negatives; ambiguous prompts |
| Eval | syntax + refusal-boundary; token accuracy | LUT quality, OOD, metaphor, paraphrase-consistency, regression, unseen-family |

---

## 7. Findings unique to each audit (so nothing is lost)

**Codex-only (now folded in):** curriculum inversion (97% ≥3 tags vs 50/30 doc); **LUT title/metadata discarded at `to_registry_row`** (F7); **no caption/scene-semantics ingestion** (blocks scene-balanced sampling & image-leakage checks); **partial-truncation blind spot** (F8); **0 coverage-tail rows**; **first-48 source bias** (34 PPR10K/14 G'MIC); the **catastrophic-drift hypothesis** and its row-selective-training remedy; **nonce-concept** and **counterfactual-ranking** evals; strong **causal-humility** stance.

**Claude-only (kept):** direct **read-only encode→decode tokenizer probes** with hard ΔE numbers (F3); the **adversarial correction** that "red→refuse" is wrong for the deployed model (F2); the finding that `validate_tags_against_behavior` **never reads the instruction prose** (so backed-synonym paraphrase is *safe* but new color-name supervision is *unbackable*); the **model-monoculture leak** (teacher+judge+baseline all Claude); the observation that the **improve-loop ledger shows zero successful knob gains** (all "gain" is smoke→full scaling); the **prompt-decomposition front-end** mapped to the spec's own "Ambiguous Child-Language Policy," with the caveat that it's a product feature, not a model fix.

---

## 8. Where the two audits diverge (genuine tensions to resolve)

1. **Causal confidence.** Claude initially asserted a fairly confident 3-part root cause; Codex insists diversity-as-dominant-cause is *unproven* and confounded. **Resolution: adopt Codex's stance** — the confounds (§2) are facts; "diversity is dominant" is a hypothesis the fixed metric must test.
2. **Decode-for-eval.** Claude's adversarial pass argued the tokenizer's *own* frozen decoder (`tokenizer.frozen`, already used read-only by `materialize_target_tokens.py`) is distinct from the disabled `eval/lut_decoder.py`, so a read-only *perceptual* OOD eval is technically reachable. Codex keeps the entire suite **decoder-free** and labels codebook distance a proxy only. **Resolution:** default to Codex's rich decoder-free proxies (NLL, counterfactual ranking, per-position/rare-code accuracy); treat "read-only decode via the tokenizer's own frozen decoder" as an **optional, owner-gated** enhancement for perceptual ground truth — *not* something to do without an explicit governance decision.
3. **Tokenizer ceiling evidence.** Claude has probe ΔE numbers; Codex has manifest corroboration but cautions perceptual causality is unproven without the decoder. **Resolution:** report the probe numbers as the strong signal, with Codex's caveat that final perceptual impact per slice needs a *tokenizer-oracle* measurement reported separately.

---

## 9. Adversarial critique (merged)

The strongest objection (both red teams agree): **the repo does not establish "insufficient diversity → unseen-concept failure."** Current evaluation cannot separate missing data from semantic drift, representation error, image blindness, target ambiguity, and memorization.

Specific challenges to the strongest proposals:
- **More paraphrases may *falsely* succeed** by repeating the same LUT and teacher fingerprint → split by LUT unit, generator, template, lexicon; match optimizer budgets. **[Both]**
- **"Train on `natural`" cannot fix OOD** — `natural` also has red/orange/mars = 0. Keep it for paraphrase robustness, not as the fix. **[Claude]**
- **Adding "red/Mars" as *supported* instructions is unbackable** — `validate_tags_against_behavior` can't verify a hue/metaphor tag, so it would inject exactly the spurious semantic→LUT mappings the gate exists to prevent. **[Claude]**
- **"Relax the clip gate" is useless-to-harmful** — clipping LUTs are the out-of-gamut regime the tokenizer reconstructs worst, and the aggregate materialization gate can tip the whole batch to write-nothing. **[Claude]**
- **Procedural "Mars" risks a `Mars→LUT X` lookup** → train components; hold out the *name* and the *full combination*; test overrides ("an icy-blue Mars"). **[Codex]**
- **Retrieval can conceal memorization** → exclude split units, source packs, templates, images, LUT neighbors. **[Codex]**
- **Contrastive negatives may be equally plausible** → use directionally-wrong / component-missing counterfactuals, not arbitrary LUTs. **[Codex]**
- **Semantic IR may be lossy** (many LUTs share a behavior summary) → first measure an oracle `ground-truth IR → codes` upper bound before committing to an IR stage. **[Codex]**
- **Codebook distance is not proven perceptual** → keep it a diagnostic proxy. **[Both]**
- **Extreme-data expansion may exceed the frozen tokenizer** → report tokenizer-oracle error separately per extreme slice. **[Both]**
- **"Unseen in SFT" ≠ "unseen to Qwen"** → compare base model vs current adapter vs row-selective adapter on decomposition to detect catastrophic drift. **[Both]**
- **Model monoculture leak** — teacher (`claude-sonnet-4-6`) + judge (`claude-opus-4-8`) + prompted-frontier baseline all include the same Claude models: Claude-authored *and* Claude-judged supervision, benchmarked against the authors. **[Claude]**
- **The improve loop has made no real gains** — the ledger's 0.044→0.4137 is smoke→full data scaling with identical hyperparameters; the one knob change died on a connection loss. **[Claude]**

**Evidence that *would* establish diversity as the dominant cause (pre-register these):**
1. On a frozen, human-authored, unit-disjoint suite, the **base model can decompose** concepts but the **SFT adapter cannot**.
2. **Row-selective** embedding/head training does **not** close the gap (rules out drift).
3. **Matched-budget** prompt-diversity ablations produce repeatable OOD gains.
4. **Tokenizer-oracle** accuracy is adequate on the relevant extreme slices.
5. Gains **survive multi-reference / behavior-window** scoring.

**Steelman (both concur):** "red/Mars" support may be **out of scope by design** (`CONTEXT.md` scopes supported attributes to *measurable* global grades; refusal is a first-class output). Where the steelman breaks: the system doesn't *cleanly refuse* either — refusal is untrained and unmeasured. So the honest synthesis is: **supporting red/Mars is legitimately out of scope, but silently emitting a muted LUT for them — instead of refusing or decomposing — is a real, in-scope, currently-invisible defect.**

---

## 10. Prioritized roadmap (merged & de-duplicated)

**Guiding principle (both audits): repair evaluation and training-contract correctness first. Until then, data/architecture changes can raise `METRIC=` while the product gets no better. All data work creates new, versioned artifacts — never mutate the corpus or frozen tokenizer; every `data/` write is ADR/owner-gated.**

### Quick wins

| # | Recommendation | Benefit | Effort | Main risk | Changes | Experiment → success/fail |
|---|---|---|---|---|---|---|
| Q1 | **Unit-aware, stratified holdout** (bucket on `split_unit_id`; family/behavior-stratified) | Trustworthy signal; unblocks tuning | S–M | Headline score drops (expected) | `sft/holdout.py`, scorer, eval manifest | 0 unit/prompt/LUT-neighbor crossing; report micro vs macro. **Success:** overlap=0 & measurable drop quantifying prior inflation. **Fail:** no drop. **[Both]** |
| Q2 | **Score every held-out row + per-slice/per-family CIs** (drop first-48) | Removes source bias | S | More GPU | `score_tokens.py`, bridge `--score-limit` | Macro family/style/tail accuracy w/ group-bootstrap CIs. **[Both]** |
| Q3 | **Restore refusal via portable staged paths** | Actual boundary learning (fixes F2) | S (+ADR for data) | Over-refusal | New versioned unsupported staging; loader asserts | 0 unsupported skips; refusal P/R, mixed-recall, over-refusal. **Success:** refuses OOD/local instead of muted LUT. **[Both]** |
| Q4 | **Alternate concise + existing `natural` at equal LUT exposure** | Lexical robustness (not OOD) | S | Teacher fingerprint; LUT overweighting | example sampler only | concise-only vs mixed at matched steps & unique-LUT count. **Success:** paraphrase-set gain, no in-dist regression. **[Both]** |
| Q5 | **Assert all 64 target positions survive** (fixes F8) | Prevents silent partial targets | S | More rows may fail | example builder + scorer asserts | every trained/scored supported row has exactly 64 code positions. **[Codex]** |
| Q6 | **Row-selective new-token training + trainable-module manifest** (implement/record J) | Preserve pretrained semantics; test drift | M | PEFT/tied-weight complexity | row-selective masks; manifest | old rows unchanged after step; unseen-word retention A/B (base vs adapter vs row-selective). **[Both]** |
| Q7 | **Build OOD + refusal eval slices** (unseen wording/concepts/nonce/paraphrase) scored decoder-free | Makes the failure visible & every fix falsifiable | S–M | proxy coarseness | `make_smoke_rows.py`, `gating_slice_registry.yaml`, `unsupported_metrics` | tracked attempt/refusal/validity per slice. **[Both]** (see §11) |

### Medium-term experiments

| # | Recommendation | Benefit | Effort | Risk | Changes | Experiment |
|---|---|---|---|---|---|---|
| M1 | **Add an absolute-hue axis to `behavior_vector`** + reference target LUTs | Makes hue intent *backable & scorable* — the real in-scope unlock | M | coarse hue → mislabels | `behavior_vector.py`, `frontier_scoring.TAG_DIRECTIONS`, hue-sector validator | hold out "warm-red/amber/teal"; measure output-cast hue error. **[Claude]** |
| M2 | **Recover LUT titles & source-style metadata** (fix F7) | Free named-style supervision | M | trademark/filename noise | persist `extra` in `to_registry_row`; validation layer | title-visible vs behavior-only; hold out title/source families. **[Codex]** |
| M3 | **Atomic→compound→style/metaphor curriculum** (fix the 97%-compound inversion) | Reusable factors; unseen combinations | M | synthetic shortcuts | new versioned LUT/prompt curriculum | train primitives; hold out whole pairs/triples & concept names. **[Codex]** |
| M4 | **Multi-teacher + human prompts, 4–8 families/LUT (backed paraphrase only)** | Linguistic breadth; breaks idiolect | M | label drift; fingerprints | versioned prompt table w/ teacher/template IDs | hold out generator, template, lexicon, human-authored. **[Both]** |
| M5 | **Representable procedural stylization** (teal-orange, moderate casts, near-mono) + tags | Fills the *reachable* stylized gap | M | must stay in-manifold (no pure hue) | `sources/procedural.py`, tag maps | eval held-out representable stylized prompts. **[Both]** |
| M6 | **Diverse image pool + relate images to LUTs; then implement projector training** | Reduce image blindness (fix F6) | M–L | storage; global LUT ≠ all scenes | image registry + captions/scene tags; pairing groups; then `projector_policy` | same LUT × image families; blank-image ablation. **[Both]** |
| M7 | **Normalize selection axes + add residual-PCA/image/prompt embeddings + tail quotas** | Meaningful coverage (fix H) | M | embedding bias | `embeddings.py`, `selection.py` | coverage radius, cluster entropy, rare-code recall, macro score. **[Both]** |
| M8 | **Auxiliary semantic-IR + behavior heads (multi-task); contrastive paraphrase/LUT alignment** | Makes decomposition observable; paraphrase invariance | M | loss competition; lossy IR | new heads + losses; contrastive sampler w/ hard negatives | attribute F1, behavior MAE, held-out-combination accuracy; prompt↔LUT R@k; **oracle IR→codes upper bound first**. **[Both]** |
| M9 | **(Owner-gated) Read-only LUT-quality eval via the tokenizer's own frozen decoder** | Perceptual ground truth on OOD | M | governance vs decoder-freeze intent | eval-only decode path (NOT `eval/lut_decoder.py`) | direction/safety/hue on OOD slices. **[Claude — gated]** |

### High-risk / high-reward (new ADRs; out of current scope)

| # | Recommendation | Benefit | Effort | Risk |
|---|---|---|---|---|
| H1 | **Boundary → recipe decomposition → code prediction** (explicit compositional stage) | Genuine composition | L | parser errors; exposure gap |
| H2 | **Spatial 4³ / codebook-aware prediction head** | Uses tokenizer geometry; avoids causal-order errors; soft/near-code credit | L | codebook distance ≠ perceptual |
| H3 | **Retrieval + generative residual** | Safe known-style fallback | M–L | disguised memorization (strict leakage exclusion) |
| H4 | **Prompt-decomposition front-end** ("Mars"→primitives via the wired frontier client) | Usable OOD behavior *today*, zero frozen-stack contact | S | **product feature, not a model fix**; bounded by F3; frontier dependency erodes the SLM's cheap/offline raison d'être |
| H5 | **Parametric recipe→LUT renderer** (bypass VQ tokens) to serve out-of-gamut extremes | Ships extreme looks despite the frozen stack | L | different model contract; new ADR |
| H6 | **v2 tokenizer** (RVQ / larger codebook / bigger latent) | Lifts the representation boundary | XL | forbidden here; re-materialization + retrain; new ADR |
| H7 | **MoE by style/regime** | Specialized capacity | XL | **unjustified at 2.7k rows**; routing collapse — defer until scale + eval improve |

**Drop / downgrade:** "train on `natural` as an OOD fix" (→ robustness only); "add red/Mars as supported"; "relax the clip gate"; treating H4 as *the* fix; MoE now.

---

## 11. Proposed OOD evaluation suite (freeze before new training)

Tiers: **NOW** = decoder-free (SLM forward pass only); **DECODE** = tokenizer's own frozen decoder (owner-gated, M9); **JUDGE** = VLM/human on a graded image. All rate gates reuse `stats.py` (Wilson, paired bootstrap, McNemar, Holm-Bonferroni, `min_N`).

| Slice | Split rule | Primary question | Key metric |
|---|---|---|---|
| **Unseen wording** | same behavior/LUT; generator + wording family held out | meaning invariant to wording? | NOW attempt/validity; DECODE direction ≥0.85 & ≥ in-dist −5pp **[Both]** |
| **Named concepts** | concept term absent from SFT; component attributes present (Mars, sodium-vapor night, oxidized copper) | can pretrained concepts map to known attributes? | DECODE hue/direction+safety; JUDGE tie-break **[Both]** |
| **Nonce concepts** | invented term defined in-context ("Varellian = muted violet shadows + amber highlights") | genuine decomposition vs pretraining recall? | NOW/DECODE component match **[Codex — high value]** |
| **Unseen combinations** | all atoms seen; exact pair/triple absent | attributes compositionally controllable? | DECODE all-component direction; flag cancellations **[Both]** |
| **Unseen styles** | whole source pack + behavior cluster + title family + LUT neighborhood held out | generalize outside known bundles? | DECODE style-window membership + NN margin **[Both]** |
| **Extreme-safe LUTs** | behavior-tail quantiles within admitted support | degrade near safe extremes? | DECODE safety ≥0.95; magnitude scales (no identity collapse) **[Both]** |
| **Tokenizer stress** | diagnostic extreme LUTs, reported *separately* | is failure semantics or representation? | tokenizer-oracle ΔE per slice **[Both]** |
| **Boundary / refusal** | minimal global/local pairs ("make everything red" vs "the shirt red") | respects capability boundary? | NOW refusal P/R, boundary F1, over-refusal **[Both]** |
| **Image sensitivity** | same prompt across image panels (portrait/night/interior/grayscale/saturated) | does vision affect prediction appropriately? | NOW/DECODE per-image variance; blank-image gap **[Both]** |
| **Counterfactual ranking** | correct target + structured wrong targets (opposite sign / missing component / excess magnitude) | prefers semantically correct behavior? | NOW target-NLL ranking margin **[Codex — decoder-free, high value]** |
| **In-distribution regression** | promote the Q1-fixed holdout as a first-class gated slice, baseline frozen | any quality loss on the canonical distribution? | NOW token acc **with CI**; DECODE paired-delta vs baseline **[Both]** |

**Leakage controls — group/hold out by:** `split_unit_id` & LUT-neighbor; source pack, photographer/image identity, image near-neighbor; prompt template, teacher model, generation batch, concept family; procedural generator family; style/title family. **[Both]**

**Decoder-free metric set:** macro teacher-forced code accuracy; exact 64-code rate; target NLL + correct-vs-counterfactual ranking margin; per-position/corner/edge/interior accuracy; rare-code recall; free & constrained grammar validity + exact token count; refusal P/R, boundary F1, mixed recall, over-refusal; paraphrase-group consistency; attribute-set F1 & behavior-vector MAE (if IR added); OOD gap vs matched in-distribution controls; nearest-training-neighbor similarity; group-bootstrap CIs; multi-reference / behavior-window scoring for ambiguous concepts. **Codebook-embedding distance may be included but must be labeled a representation-space proxy — not ΔE / perceptual quality.** **[Codex, endorsed]**

Build order: **In-distribution regression → Unseen wording → Named + Nonce concepts → Counterfactual ranking → Paraphrase** first (cheapest, most on-target); the rest as M1/M9 land.

---

## 12. Constraints & governance flags

- **Hard-forbidden (need a new ADR to even scope):** retraining/re-gating/re-freezing the tokenizer (H6); enabling `eval/lut_decoder.py`; modifying `luts/` or the frozen tokenizer weights.
- **ADR / owner sign-off required:** any write under `data/` (all D-/M-series data work — `active_rows.jsonl` is git-tracked and `paths` are locked); the parametric renderer (H5, changes the 64-VQ-token output contract); projector training (M6, unfreezes a module outside the tunable-knob list); read-only decode-for-eval (M9, touches the decoder-freeze *intent* even via the tokenizer's own decoder).
- **Within existing tunable knobs only:** `learning_rate_lora`, `lora_r/alpha/dropout`, `warmup_ratio`, `max_grad_norm`, `weight_decay`, `max_pixels`(≤401408). Locked: `epochs`(2), batch triple, `num_new_tokens`(259), `base_model_id`, quant, `max_seq_len`, `seed`, paths.
- **Every extreme-data proposal must report tokenizer-oracle reconstruction error separately** — data that can't be represented is training the model to emit tokens that can't express the look.

---

## 13. Bottom line

Both audits independently reach the same practical conclusion: **the system is a well-engineered pipeline for conventional, measurable global grading, and the "red/Mars" gap is real but mis-attributed if called simply "not enough diverse data."** The dominant, *fixable-now* problems are that **the metric can't see the failure (leaked + in-distribution + no LUT-quality signal)** and that **the model neither supports nor cleanly refuses out-of-scope requests, so it fails silently.** Beneath that sit a **language ontology** that can't name most of the hue circle and a **frozen tokenizer** that can't represent the literal extreme tail.

**Do first:** unit-aware/stratified holdout + full per-slice scoring + a frozen OOD/refusal eval suite (Q1–Q2, Q7), restore refusal training (Q3), and row-selective new-token training with a base-vs-adapter drift test (Q6). Only once the ruler is honest should data (backed paraphrase, recovered titles, representable stylization, curriculum, diverse images), the **absolute-hue axis** (M1, the real in-scope unlock), and the semantic-IR/decomposition tracks proceed — each behind its governance gate and each validated against a tokenizer-oracle upper bound.

*No code, data, tokenizer, or decoder was modified in producing this audit.*
