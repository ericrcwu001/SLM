## Purpose
Prompt-to-LUT Color Playground is a small, bounded tool for visual experimentation. A child writes a color instruction, the system returns one global LUT, and the child can compare the result against the original image. If the request needs local selection, object knowledge, relighting, replacement, or new image content, the system should say so instead of pretending a LUT can do it.
The bet is modest: kids can get better at noticing global color changes when the tool lets them try an edit, compare versions, and revise. The project should not claim that it teaches taste, improves creativity in general, or turns a child into a better artist. The useful claim is narrower: better noticing, better vocabulary, and better revision inside one visual medium.
### In scope
- Image-conditioned, instruction-guided global color grading.
- One decoded 17x17x17 residual global LUT added to identity and exported as a normal LUT.
- Prompt attributes: warmth/coolness, tint, exposure, contrast, black point, highlights, shadows, saturation, and a small number of style bundles with measurable color behavior.
- Evaluation of token validity, decoded LUT validity, direction of color change, target fidelity, clipping, smoothness, foldover, neutral drift, skin-locus/neutral preservation, unsupported recall/precision, over-refusal, and boundary F1.
- A child-facing workflow built around preview, compare, undo, revise, and name the look.

### Out of scope
- Local region edits.
- Object-specific recoloring.
- Background-only or subject-only changes.
- Inpainting, removal, replacement, relighting, geometry edits, texture/detail edits, and other new-pixel work.
- Companion behavior, therapy, open-ended tutoring, grading, ranking, trait praise, learning-outcome evaluation, or social comparison.

## DOK 4 - spiky points of view (SPOVs)
- SPOV 1: The frontier can already grade color - it just can't emit the artifact - so the moat is the output contract, not the model's taste. Prompted to dump a raw 17x17x17 .cube (~4,913 rows, ~100K floats), every LUT that finished was direction- and safety-correct; every failure was truncation or refusal. Read it right: not "SOTA is bad at color" -> "raw dense floats are the wrong output contract for an LLM." Two mechanisms behind the win: (1) a tractable, in-distribution budget -> 64 learned VQ tokens vs ~100K blind floats; (2) decomposition paid once at build time -> the teacher/judge pipeline is a reasoning-compiler, so inference is closer to lookup (why the frontier nailed composites 3/3 and refusals 3/3 but lost named styles 0/3 - it had to decompose *and* serialize in one budget). Plus deployment: offline/local, ~$0.00x, sub-second, deterministic byte-exact artifact. Honesty rails -> the budget leg is a *format* claim, not "only fine-tuning can grade": a fair structured-output frontier baseline (recipe or coarse grid + our renderer) might complete too, deferred to a post-training image-to-image comparison; if it matches, narrow to locality/cost/latency/auditability. "Color never fails" is survivorship bias (n=9, fidelity unscored). And the VQ vocab doesn't delete failure -> it relocates it into a silent quantization error on the hardest scraped_web LUTs.
- SPOV 2: A global LUT should be the desired outcome because its limit is visible - same input color, same output color, everywhere. Most editors hide the mechanism: they can swap content, identity, meaning, and still look plausible. A LUT can't. Warmth, saturation, contrast, shadow lift, highlight rolloff is measurable; picking out one object or region is not. That hard, inspectable limit is the teaching surface, not a weakness. v1 stays one 17x17x17 global LUT and makes before/after, the decoded LUT, and its metrics easy to see. If the prompt asks for local or semantic work -> the honest answer is <unsupported>, because the boundary falls straight out of the medium --> doesn't teach kids the vocabulary or precision with their creative intent.
- SPOV 3: Natural language is the right input because kids have the intent before they have the vocabulary. "Make it warmer / less heavy / make it pop" comes years before "black point, saturation, curves." The model's job -> turn everyday intent into supported global color, or name the boundary when the real ask is texture/detail/local ("make it sharper" -> more contrast, or refuse). Sliders and presets are fine baselines, but they start from the tool's words; natural language starts from the child's -> then walks them to the color concept they just moved. Only pays off if the child can inspect, undo, compare, reject, and learn the vocabulary after - otherwise it's just prompt magic.
- SPOV 4: The deliverable is also the workbench loop, not just the model - and what it teaches is process, not taste. A valid LUT only proves feasibility; the learning lives in the interface. Upload -> instruct -> generate -> see it on your own image -> see the same LUT across nature / portrait / still-life / architecture / low-light. The transfer grid makes the global rule visible: one look carries, but can't fix every subject or light. Growth we actually claim = slower looking, prediction, explanation, revision ("I lifted the shadows -> the face lost contrast"), measured on new images - not aesthetic rank, not "you have a great eye." Workbench, not character: show versions, name the change, no flattery, child chooses.
- SPOV 5: Judge the contract, not the picture - and gate it in a fixed priority order before anything about "looks good." Asked for cooler shadows, warmed them -> fail, even if it's pretty. Clips / folds the lattice / drifts neutrals / distorts skin -> fail. Style words earn a slot only with a tested recipe ("cinematic" = cooler shadows + warmer highs + lower saturation + mild contrast + soft rolloff); "beautiful" never enters the validator. Headline = prompt-to-LUT pass rate, split syntax -> LUT validity -> direction -> safety. Reward is lexicographic: valid-output-or-refusal -> boundary -> direction -> LUT safety -> target fidelity -> a small style preference, dead last. Preference never buys back a broken token, wrong direction, unsafe LUT, or a missed refusal. RS/DPO before GRPO; GRPO only once SFT already emits valid tokens, follows direction, and refuses sanely.
- SPOV 6: A hard gate is only as honest as the population it's computed over - an exception is a logged artifact, not a lowered bar. (Refines SPOV 4.) A worst-case percentile means nothing pooled -> stratify by product tier (gold/served vs diagnostic-only). Our tokenizer: p5_psnr failed on the full holdout (28.37) but the failing worst-5% was 100% diagnostic-only, and gold passes (31.24). Right move -> not "block forever on rows you'll never serve," not "nudge the threshold" -> a dated, signed exception that pins the true value. Legit only if the tier is frozen before the gate, provenance-defined, independent of the gate's own failure axis, reported with N/CI, and never moved for yield. Honest flag: our own waiver only half-meets this - tier partly derived from the same fit/smoothness axis (near-circular), boundary loosened for yield, no CI -> a documented, revisit-able exception, not a clean pass. No expiry + no fix commitment -> normalization of deviance.
- SPOV 7: The dataset is the deliverable - collect broadly, train narrowly - and it's a bet, not a result yet. Corpus and training set are different objects with different size logic. Collect wide for coverage (expert LUTs from PPR10K/FiveK XMP, ~2,000 behavior-only scraped .cube, HaldCLUT packs) -> then select small (12k) by usage-prior + facility-location/MMR, distilled from a teacher under *authoritative deterministic* color gates (the LLM judge pre-filters fluency; it can't overrule the checks). 50k/100k -> demoted to a scale-up milestone. Well-precedented (LIMA, phi-1, Distilling Step-by-Step, INGENIOUS) but unproven here: no SFT run yet -> no past-tense "curation beat volume." Live tensions: ~95% of supported rows are diagnostic-tier while headline eval is gold-only; at demo scale (3,033 < 12k) the coverage selector barely bites. Counter-evidence to hold: with quality controlled, more good data still helps -> the milestone may be real, not just deferred.

## Experts
- Lois Hetland / Ellen Winner / Project Zero - Their Studio Thinking work keeps the learning claim pointed at habits: observe, envision, reflect, stretch, explore, and revise.
- Carl Hendrick - His learning-science writing is useful here because enjoyment and in-session fluency are weak evidence. The product needs transfer, explanation quality, and retention of color vocabulary.
- Dedre Gentner - Her comparison and structure-mapping research supports the side-by-side workflow. Children learn the relation by lining up cases; one result in isolation does much less.
- Alison Gopnik / Laura Schulz / Elizabeth Bonawitz - Their work on children’s exploration supports guided play: enough freedom to test ideas, enough constraint that the test means something.
- Edward Deci / Richard Ryan - Self-determination theory explains why the child needs bounded choice and control over the artifact.
- Terry Winograd / Ben Shneiderman / Allison Druin / Mitchel Resnick - Child-centered design, Cooperative Inquiry, Scratch, and construction-kit work all point toward tool-as-material.
- Pierre Bourdieu / Gloria Ladson-Billings / Ruha Benjamin-adjacent equity literature - Taste and “quality” are not neutral. The product should describe changes without ranking cultural taste.
- Tianren Ma / Mingxiang Liao / Xijin Zhang / Qixiang Ye - Their AceTone paper is the closest technical ancestor for this project: a VQ-VAE LUT tokenizer plus a vision-language model that predicts 3D LUTs from text or reference images. Treat it as the first implementation to beat or reproduce, not as proof the child-facing product works.
- CIE / Sharma / ACES / LUT researchers - Colorimetry, CIEDE2000, ACES CLF, image-adaptive LUTs, and spatial-aware LUTs ground the measurable side of the project.
- PPR10K / MIT-Adobe FiveK dataset authors - These corpora are useful input material, but their licensing, portrait bias, local-edit assumptions, and lack of prompt labels limit how they can be used.
- Selective classification and instruction-following researchers - Abstention, coverage, risk, and atomic pass rates define how <unsupported> and prompt following should be evaluated.

## DOK 3 - insights
### Product scope and UX
- A global LUT works as a teaching material because its limits are plain. It can change color globally; it cannot pick out a shirt, swap a sky, remove a person, or relight a face.
- The comparison view is the main learning loop. Original, version A, version B, and “what changed?” matter more than a gallery of filters.
- Feedback should name the edit. “Highlights are warmer and shadows are lower” gives the child something to inspect. “This looks better” does not.
- Natural language matters because it lets the child begin with intent instead of tool vocabulary. The product can then connect “make it moodier” to darker shadows, lower exposure, lower saturation, or a refusal if the request is really local or textural.
- Natural language helps only if the child can inspect, undo, compare, reject, and rename the final result.
- The final interface should be a GUI, not just a fine-tuned-model demo. The core workflow is: upload an image, create a LUT with the model, preview it on the upload, then apply that same LUT to five fixed image types so the child can see what transfers and what breaks.

### Evaluation and safety
- The first question is whether the requested color behavior happened safely.
- <unsupported> belongs in the spec and in the metrics. It needs recall, precision, over-refusal, false-support, coverage, and selective-risk reporting.
- A system that refuses too often can look safe while being useless. Over-refusal must sit next to unsupported recall in every report.
- Skin preservation needs more than a single light-dark score. Test tone, hue, lighting, and camera-pipeline variation, then review images that contain people.
- Style labels should be audited as color recipes. A label that cannot be measured should stay out of v1.
- Prompting a frontier model is treated as a real baseline, not a hypothetical. Frontier models can produce correct, safe global-color LUTs; where they fail is *completing the artifact*. So evaluation must separate "did it get the color right" from "did it emit a valid complete LUT," and the project's claim narrows to reliability, output budget, cost, and local/offline operation (ADR 0011).
- A hard gate is only honest relative to the population it is computed over. Stratify gate metrics by product tier (gold/headline-eligible vs diagnostic-only). When a worst-case percentile fails only on rows that will never be served, record an auditable, signed exception that pins the true measured value, rather than lowering the threshold or blocking the ship on non-product rows. The catch: that exception is legitimate only if the tier partition is frozen before the gate runs, defined by provenance rather than by the gate's own failure axis, and never moved to raise yield.
- The boundary risk that actually needs the model's help is over-refusal, not detection. The frontier baseline refused the mixed trap correctly 3/3 for ~$0.01, so unsupported-detection is near-free and the small model can at best match it. The refusal investment and the headline boundary metric should therefore lead with over-refusal and false-support (refusing a supported prompt to look safe), not with unsupported recall alone.

### Data and training
- PPR10K and FiveK are transform sources. They are not taste truth and they are not prompt datasets.
- Public LUT collections can add variety, but every source needs provenance, rights, duplicates, and style balance checked.
- Prompt labels need measured backing. If a tag says “warm matte film,” the fitted LUT should actually move warmth, black point, contrast, and saturation in the expected directions.
- SFT should prove basic syntax and behavior before GRPO starts. Otherwise the reward stage is cleaning up avoidable noise.
- The corpus and the training set are different objects. Collect broadly (derive expert LUTs from XMP, scrape ~2,000 web `.cube` files, add HaldCLUT packs) for coverage; then select a small active set (12k target) by usage-prior + coverage, not by raw scale. Distill instructions from a frontier teacher under deterministic color-behavior gates instead of scaling human annotation.
- Scraped web LUTs are usable at scale only *behavior-only*: measure their color behavior, never trust their authored tags, and let the canonical sRGB-domain gate reject out-of-domain log/Rec709 video LUTs. Diversity from uncontrolled sources also concentrates the hardest reconstruction tail - the tokenizer's worst cases are all scraped_web.
- A generative LUT-token warmup before instruction SFT adapts the model to the new discrete token distribution; it is not pretraining from scratch, and it is train-only relative to final eval.
- The deterministic tag→behavior color verifier, not the LLM judge, is the true quality oracle for distilled data. The judge (Opus 4.8) is a cheap pre-filter for fluency and plausibility; the non-LLM checks are authoritative and set the ceiling on data quality. Where no deterministic verifier exists (naturalness, subtle interactions), the pipeline is running on trust and should be labeled that way, not scored as if judged.
- Derived-LUT corpora have a leakage topology exact-dedup cannot see: one expert edit applied across many source images yields many near-identical derived LUTs, so exact-match splitting looks clean while the model has effectively seen the test LUT. That is why the eval contract requires simultaneous disjointness on LUT, image, source-pair, support-map, and prompt-template identity plus held-out expert ids - and why any benchmark comparison that used only exact-dedup splits is not comparable.

### Learning claim
- The defensible learning claim is near visual transfer: noticing, comparing, explaining, and revising global color treatment on similar images.
- Claims about better creativity, higher grades, executive-function gains, or improved general taste require a separate child study.
- Aesthetic scoring should stay outside the center of the product. Help kids see and name changes; let them decide what they like.

### Child development claim
- The development claim should be about process growth: slower looking, more precise language, better prediction, and more deliberate revision.
- The tool should ask for an explanation before or after a change often enough to make the child connect action to evidence: “what changed?” and “what do you see that makes you say that?”
- Comparison is what turns a filter into a learning event. One image can feel like a magic result; two versions let the child notice what actually moved.
- Agency matters because the child chooses the final version. The system can show consequences and offer another try, but it should not choose the “best” look.
- Natural language should be treated as the bridge from child vocabulary to color vocabulary. The point is not to keep kids using vague words forever; it is to help them move from “make it pop” to “higher contrast and saturation” through visible feedback.
- The product should test development with new images: can the child predict what “warmer highlights” will do, explain the result, and revise intentionally?

## DOK 2 - knowledge tree
### Category 1: Learning science - transfer, comparison, and perceptual judgment
#### Subcategory 1.1: Transfer must be narrow
Source: Barnett, S. M., & Ceci, S. J. (2002). When and where do we apply what we learn? Psychological Bulletin, 128(4), 612-637.
- DOK 1 - facts:
- Transfer depends on content, context, and distance.
- Near transfer is easier to defend than far transfer across domains.
- Link: https://doi.org/10.1037/0033-2909.128.4.612
- ! DOK 2 - summary: Claim near visual transfer. Leave general creativity and school performance out. Test whether children can explain and predict global color changes on new but similar images.

---
#### Subcategory 1.2: Comparison builds structure
Source: Gentner, D., Loewenstein, J., & Thompson, L. (2003). Learning and transfer: A general role for analogical encoding. Journal of Educational Psychology, 95(2), 393-408.
- DOK 1 - facts:
- Comparing cases helps learners abstract relational structure.
- Learners often miss the relevant relation until two examples are aligned.
- Link: https://doi.org/10.1037/0022-0663.95.2.393
- ! DOK 2 - summary: Side-by-side LUT variants should be a core workflow. Comparison is doing the teaching.

---
#### Subcategory 1.3: Interleaving helps perceptual induction
Source: Kornell, N., & Bjork, R. A. (2008). Learning concepts and categories: Is spacing the "enemy of induction"? Psychological Science, 19(6), 585-592.
- DOK 1 - facts:
- Interleaved/spaced examples beat massed examples for learning painting-style categories.
- Learners still preferred the easier massed condition, even when it produced worse learning.
- Link: https://doi.org/10.1111/j.1467-9280.2008.02127.x
- ! DOK 2 - summary: Mix warm/cool, high/low contrast, muted/saturated, and filmic/natural examples over time. Blocked filter browsing feels fluent but teaches less.

---
#### Subcategory 1.4: Perceptual learning is real but domain-bounded
Source: Kellman, P. J., & Garrigan, P. (2009). Perceptual learning and human expertise. Physics of Life Reviews, 6(2), 53-84.
- DOK 1 - facts:
- Perceptual learning improves extraction of task-relevant information.
- Expertise comes from better pickup of relevant structure, not just more exposure.
- Link: https://doi.org/10.1016/j.plrev.2008.12.001
- ! DOK 2 - summary: “Developing an eye” is defensible only as better noticing in a narrow visual domain. Measure color-cause vocabulary and prediction on similar tasks.

---
#### Subcategory 1.5: Guided discovery beats unguided browsing
Source: Alfieri, L., Brooks, P. J., Aldrich, N. J., & Tenenbaum, H. R. (2011). Does discovery-based instruction enhance learning? Journal of Educational Psychology, 103(1), 1-18.
- DOK 1 - facts:
- Unassisted discovery is weak.
- Discovery improves when learners get scaffolding, feedback, worked examples, or elicited explanations.
- Link: https://doi.org/10.1037/a0021017
- ! DOK 2 - summary: Avoid random filter surfing. Use prompts, comparisons, explanations, and revision loops that keep exploration bounded.

### Category 2: Arts education and visual literacy
#### Subcategory 2.1: Keep arts-transfer claims narrow
Source: Winner, E., Goldstein, T. R., & Vincent-Lancrin, S. (2013). Art for Art’s Sake? OECD.
- DOK 1 - facts:
- Evidence for broad arts-to-academic transfer is weak.
- Arts learning is strongest when evaluated on arts-relevant habits and practices.
- Link: https://doi.org/10.1787/9789264180789-en
- ! DOK 2 - summary: Avoid claims about math, reading, IQ, GPA, or general creativity. Evaluate looking, explaining, comparing, and revising color treatment.

---
#### Subcategory 2.2: Studio habits match the product loop
Source: Project Zero / Studio Thinking framework.
- DOK 1 - facts:
- Studio habits include observe, envision, express, reflect, stretch, explore, understand art worlds, and engage/persist.
- These habits describe process: looking, trying, reflecting, and revising.
- Link: https://pz.harvard.edu/projects/the-studio-thinking-project
- ! DOK 2 - summary: The LUT playground should teach studio-like process behaviors: look closely, try a change, compare, explain, revise.

---
#### Subcategory 2.3: Visual thinking routines fit LUT comparison
Source: Visual Thinking Strategies / Housen visual-literacy work.
- DOK 1 - facts:
- Routines like “what is going on?”, “what do you see that makes you say that?”, and “what more can we find?” scaffold evidence-backed observation.
- The method privileges noticing and explanation over expert judgment.
- Link: https://vtshome.org
- ! DOK 2 - summary: LUT comparison can borrow the same questions: what changed, what do you see that makes you say that, and what next experiment would test it?

---
#### Subcategory 2.4: Development is visible in explanation and revision
Source: Project Zero / Studio Thinking framework; Visual Thinking Strategies / Housen visual-literacy work; Shute (2008) formative feedback review.
- DOK 1 - facts:
- Studio habits frame arts learning as observable process: observe, envision, reflect, stretch, explore, and revise.
- Visual thinking routines ask children to back claims with visible evidence.
- Formative feedback works best when it is specific and tied to a next action.
- Links: https://pz.harvard.edu/projects/the-studio-thinking-project ; https://vtshome.org ; https://doi.org/10.3102/0034654307313795
- ! DOK 2 - summary: The product can claim child development only when the child shows better process: closer observation, clearer explanation, and more intentional revision.

### Category 3: Motivation, feedback, and child agency
#### Subcategory 3.1: Autonomy needs bounded choice
Source: Ryan, R. M., & Deci, E. L. (2000). Self-determination theory and the facilitation of intrinsic motivation. American Psychologist, 55(1), 68-78.
- DOK 1 - facts:
- Autonomy, competence, and relatedness support intrinsic motivation.
- Controlling feedback can undermine motivation.
- Link: https://doi.org/10.1037/0003-066X.55.1.68
- ! DOK 2 - summary: Keep the child in control of the artifact. Offer meaningful choices, reversible edits, and descriptive feedback rather than rankings.

---
#### Subcategory 3.2: Too many options can weaken agency
Source: Patall, E. A., Cooper, H., & Robinson, J. C. (2008). The effects of choice on intrinsic motivation and related outcomes. Psychological Bulletin, 134(2), 270-300.
- DOK 1 - facts:
- Choice can increase motivation.
- Effects depend on the number and meaning of options.
- Link: https://doi.org/10.1037/0033-2909.134.2.270
- ! DOK 2 - summary: A huge filter catalog can make choices worse. Expose a small set of changes that children can compare and understand.

---
#### Subcategory 3.3: Trait praise is risky
Source: Mueller, C. M., & Dweck, C. S. (1998). Praise for intelligence can undermine children's motivation and performance. Journal of Personality and Social Psychology, 75(1), 33-52.
- DOK 1 - facts:
- Trait praise can reduce persistence after failure.
- Process-oriented feedback is safer than person-level evaluation.
- Link: https://doi.org/10.1037/0022-3514.75.1.33
- ! DOK 2 - summary: Avoid “you have a great eye” and talent labels. Describe the edit and the next experiment instead.

---
#### Subcategory 3.4: Feedback should point back to the task
Source: Shute, V. J. (2008). Focus on formative feedback. Review of Educational Research, 78(1), 153-189.
- DOK 1 - facts:
- Effective feedback is specific, supportive, and tied to next action.
- Task/process feedback is safer than self-level judgment.
- Link: https://doi.org/10.3102/0034654307313795
- ! DOK 2 - summary: Feedback should name observable color movement. “Shadows lifted, saturation lowered” is useful; “better image” is not.

---
#### Subcategory 3.5: Agency makes the loop developmental
Source: Ryan, R. M., & Deci, E. L. (2000); Patall, Cooper, & Robinson (2008); Mueller & Dweck (1998).
- DOK 1 - facts:
- Autonomy supports intrinsic motivation when choices feel meaningful.
- Choice effects depend on the number and meaning of options.
- Trait praise can reduce persistence after failure.
- Links: https://doi.org/10.1037/0003-066X.55.1.68 ; https://doi.org/10.1037/0033-2909.134.2.270 ; https://doi.org/10.1037/0022-3514.75.1.33
- ! DOK 2 - summary: Kids develop more safely when the tool gives bounded choices, shows the consequence of each choice, and keeps feedback on the work instead of the child’s identity.

---
#### Subcategory 3.6: Natural language gives children a lower-floor entry point
Source: Resnick et al. (2009), Scratch: Programming for All; Druin (1999), Cooperative Inquiry; Visual Thinking Strategies / Housen visual-literacy work.
- DOK 1 - facts:
- Construction-kit design argues for a low floor, wide walls, and room for personally meaningful projects.
- Cooperative Inquiry treats children as design partners whose language and practices should shape the tool.
- Visual thinking routines start with ordinary observation language before expert vocabulary.
- Links: https://doi.org/10.1145/1592761.1592779 ; https://doi.org/10.1145/302979.303166 ; https://vtshome.org
- ! DOK 2 - summary: Natural language is the lower-floor interface for this product: the child starts with “make it warmer” or “make it pop,” then the tool connects that intent to visible color terms like contrast, saturation, shadows, and highlights.

### Category 4: Rights, equity, taste, and skin-tone risk
#### Subcategory 4.1: Child-centered technology requires limits
Source: UNICEF guidance on children and automated systems; UN CRC General Comment No. 25; UNESCO education-technology guidance; COPPA rulemaking.
- DOK 1 - facts:
- Child-facing products need safety, privacy, fairness, transparency, accountability, inclusion, and age-appropriate design.
- U.S. under-13 services face consent, minimization, security, retention, and monetization limits.
- Links: UNICEF child-rights technology guidance; UN CRC General Comment No. 25; UNESCO education-technology guidance; FTC COPPA rule.
- ! DOK 2 - summary: The product cannot profile, grade, manipulate, or make consequential claims about the child. Minimize data and keep adults in the loop where required.

---
#### Subcategory 4.2: Taste carries norms
Source: Bourdieu, P. (1984). Distinction; Ladson-Billings, G. (1995). Toward a theory of culturally relevant pedagogy.
- DOK 1 - facts:
- Taste judgments can function as classed cultural sorting.
- Culturally relevant pedagogy supports cultural competence and critique rather than assimilation into dominant norms.
- Link: https://doi.org/10.3102/00028312032003465
- ! DOK 2 - summary: Words like “professional,” “natural,” and “beautiful” can smuggle in norms. Translate style terms into visible color attributes and let the child choose.

---
#### Subcategory 4.3: Skin-tone safety needs more than one axis
Source: Buolamwini, J., & Gebru, T. (2018). Gender Shades; Thong, W., Joniak, P., & Xiang, A. (2023). Beyond Skin Tone.
- DOK 1 - facts:
- Vision systems can show large demographic performance disparities.
- Skin-color fairness requires multidimensional lightness and hue measures; a single light-dark scale is too thin.
- Links: https://proceedings.mlr.press/v81/buolamwini18a.html ; https://openaccess.thecvf.com/content/ICCV2023/html/Thong_Beyond_Skin_Tone_A_Multidimensional_Measure_of_Apparent_Skin_Color_ICCV_2023_paper.html
- ! DOK 2 - summary: Audit skin preservation across tone, hue, lighting, and camera-pipeline variation. One generic skin metric will miss problems.

### Category 5: LUT and color science
#### Subcategory 5.1: A global LUT is spatially blind
Source: NVIDIA GPU Gems 2, Ch. 24; RawPedia Film Simulation / HaldCLUT documentation.
- DOK 1 - facts:
- A 3D LUT maps input color to output color independent of spatial position.
- HaldCLUTs encode global color/tonal transforms and cannot encode local denoise, sharpen, distortion, geometry, or object-specific edits.
- Links: https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-24-using-lookup-tables-accelerate-color ; https://rawpedia.rawtherapee.com/Film_Simulation
- ! DOK 2 - summary: The support boundary follows from the medium. If a prompt needs local or semantic behavior, the correct answer is <unsupported>.

---
#### Subcategory 5.2: Color claims need a declared pipeline
Source: CIE 015:2018, Colorimetry; Sharma, G., Wu, W., & Dalal, E. N. (2005), CIEDE2000 implementation notes.
- DOK 1 - facts:
- Color measurement depends on declared spaces, observers, illuminants, and conversion assumptions.
- CIEDE2000 implementation has known edge cases and published test data.
- Links: https://doi.org/10.25039/TR.015.2018 ; https://doi.org/10.1002/col.20070
- ! DOK 2 - summary: Warmth, saturation, contrast, neutral drift, and DeltaE gates mean little until the color pipeline is fixed and tested.

---
#### Subcategory 5.3: LUT validity is a safety surface
Source: ACES Common LUT Format; Zeng et al. Learning Image-Adaptive 3D Lookup Tables; Wang et al. Spatial-Aware 3D LUTs.
- DOK 1 - facts:
- LUT interchange formats make color transforms serializable and auditable.
- Image-adaptive and spatial-aware LUT papers exist because plain global LUTs cannot adapt by local region.
- Links: https://docs.acescentral.com/specifications/clf/ ; https://doi.org/10.1109/TPAMI.2020.3026740 ; https://arxiv.org/abs/2309.15662
- ! DOK 2 - summary: Decoded LUTs need checks for smoothness, foldover, clipping, neutral drift, and exportability. Syntactically valid tokens are only the first check.

### Category 6: Technical data sources and provenance
#### Subcategory 6.1: PPR10K is useful but portrait-biased
Source: Liang et al. (2021). PPR10K: A Large-Scale Portrait Photo Retouching Dataset.
- DOK 1 - facts:
- PPR10K contains 11,161 RAW portrait photos, 1,681 groups, and three expert retouch target sets.
- It includes full-resolution human-region masks and research-only/non-commercial dataset restrictions.
- Link: https://doi.org/10.1109/CVPR46437.2021.00071
- ! DOK 2 - summary: PPR10K can provide 33,483 candidate expert targets, with caveats: portrait bias, region priorities, and licensing.

---
#### Subcategory 6.2: FiveK gives breadth but no prompt labels
Source: Bychkovsky et al. (2011). Learning Photographic Global Tonal Adjustment with a Database of Input/Output Image Pairs.
- DOK 1 - facts:
- MIT-Adobe FiveK contains 5,000 DNG photos with five expert retouched renditions.
- Canonical expert outputs are TIFF16 ProPhoto RGB; image licenses are research-only/non-commercial.
- Link: https://doi.org/10.1109/CVPR.2011.5995332
- ! DOK 2 - summary: FiveK can provide 25,000 expert renditions for LUT fitting. Natural-language labels and commercial rights still need separate handling.

---
#### Subcategory 6.3: Public LUT collections need provenance gates
Source: FreshLUTs terms; RawTherapee Film Simulation Collection.
- DOK 1 - facts:
- FreshLUTs claims uploaded LUTs are CC0, but uploader provenance may be missing.
- RawTherapee’s HaldCLUT archive is mostly sRGB 8-bit PNGs and reports CC BY-SA 4.0.
- Links: https://freshluts.com/terms ; https://rawpedia.rawtherapee.com/Film_Simulation
- ! DOK 2 - summary: Public LUTs can add corpus variety, but rights, attribution, trademarks, source balance, and duplicates need explicit tracking.

### Category 7: Model architecture, training, and rewards
#### Subcategory 7.1: Prompt-to-LUT has a direct technical analogue
Source: Ma, T., Liao, M., Zhang, X., & Ye, Q. (2026). AceTone: Bridging Words and Colors for Conditional Image Grading. arXiv 2604.00530.
- DOK 1 - facts:
- AceTone is the closest direct precedent for conditional image grading using a VQ-VAE LUT tokenizer and VLM-predicted LUT tokens.
- AceTone formulates grading as a generative color-transformation task where a model directly produces 3D LUTs conditioned on text prompts or reference images.
- The AceTone paper reports a VQ-VAE tokenizer that compresses a 3x32^3 LUT vector to 64 discrete tokens with DeltaE < 2 fidelity.
- It is new enough that reproduction and version pinning matter.
- Link: https://arxiv.org/abs/2604.00530
- ! DOK 2 - summary: AceTone is the closest reference implementation. It makes the architecture plausible; product proof still has to come from local evals.

---
#### Subcategory 7.2: Tokenization makes the contract testable
Source: VQ-VAE, arXiv 1711.00937; local v1 behavior spec.
- DOK 1 - facts:
- VQ-VAE uses a discrete latent codebook with reconstruction and commitment losses.
- The v1 target is a 4x4x4 latent grid, exactly 64 LUT code tokens, and a planned codebook size of 256.
- Link: https://arxiv.org/abs/1711.00937
- ! DOK 2 - summary: Tokenized LUT output creates a crisp syntax contract: <lut_bos> plus 64 code tokens plus <lut_eos>, or <unsupported>.

---
#### Subcategory 7.3: Fine-tuning should stay narrow
Source: Qwen2.5-VL Technical Report, arXiv 2502.13923; LoRA, arXiv 2106.09685.
- DOK 1 - facts:
- Qwen2.5-VL includes an open 3B Instruct model family.
- LoRA supports parameter-efficient fine-tuning by training low-rank adapters instead of the full model.
- Links: https://arxiv.org/abs/2502.13923 ; https://arxiv.org/abs/2106.09685
- ! DOK 2 - summary: Keep training focused on output behavior and selected modules. Avoid broad model-superiority claims.

---
#### Subcategory 7.4: Reward optimization needs hard gates
Source: DeepSeekMath / GRPO, arXiv 2402.03300; reward-overoptimization literature.
- DOK 1 - facts:
- GRPO uses group scores without a critic.
- Learned or imperfect reward models can be exploited when over-optimized.
- Link: https://arxiv.org/abs/2402.03300
- ! DOK 2 - summary: Use rule-based validators before preference scoring. Reject invalid tokens, wrong color direction, unsafe LUTs, and refusal errors before looking at style.

---
#### Subcategory 7.5: Curation and distillation can beat brute scale for one narrow behavior
Source: Zhou et al. (2023) LIMA: Less Is More for Alignment; Gunasekar et al. (2023) Textbooks Are All You Need (phi-1); Hsieh et al. (2023) Distilling Step-by-Step!; INGENIOUS informative-subset selection; counterpoint: Li et al. (2023) Self-Alignment with Instruction Backtranslation.
- DOK 1 - facts:
- LIMA reached GPT-4-competitive alignment from 1,000 diversity-curated pairs with no RLHF.
- phi-1 (1.3B) reached ~50% HumanEval from filtered “textbook-quality” plus teacher-synthesized data.
- Distilling Step-by-Step trained a 770M student that beat a 540B model using ~80% less labeled data by learning teacher rationales.
- Facility-location is the canonical submodular coverage objective for training-subset selection, with a greedy (1 − 1/e) guarantee.
- Counterpoint: once quality is controlled, scaling more high-quality instruction data keeps improving instruction-following.
- Links: https://arxiv.org/abs/2305.11206 ; https://arxiv.org/abs/2306.11644 ; https://arxiv.org/abs/2305.02301 ; https://arxiv.org/abs/2305.06677 ; https://arxiv.org/abs/2308.06259
- ! DOK 2 - summary: Grounds SPOV 7. “Collect broadly, train narrowly” is a precedented design bet, not a demonstrated result. Treat 50k/100k as a live scale-up milestone and prove the small curated set beats volume on this task with an actual eval before claiming it.

### Category 8: Evaluation and baselines
#### Subcategory 8.1: Instruction following should be atomic
Source: IFEval, arXiv 2311.07911; HELM, arXiv 2211.09110.
- DOK 1 - facts:
- IFEval separates prompt-level pass from atomic instruction pass.
- HELM argues for multiple scenarios and metrics.
- Links: https://arxiv.org/abs/2311.07911 ; https://arxiv.org/abs/2211.09110
- ! DOK 2 - summary: Break prompt-to-LUT pass rate into token validity, LUT validity, color direction, refusal, safety, speed, stability, and baseline deltas.

---
#### Subcategory 8.2: Abstention has its own metrics
Source: Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks.
- DOK 1 - facts:
- Selective classification treats abstention as a first-class prediction.
- Coverage and risk show the tradeoff between answering more and being wrong less.
- Link: https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html
- ! DOK 2 - summary: <unsupported> needs recall, precision, over-refusal, false-support, coverage, and selective-risk reporting.

---
#### Subcategory 8.3: Clear status beats silent identity
Source: Nielsen Norman Group, visibility of system status heuristic.
- DOK 1 - facts:
- Interfaces should show users what is happening.
- Recoverable, visible states reduce confusion.
- Link: https://www.nngroup.com/articles/visibility-system-status/
- ! DOK 2 - summary: A visible refusal is better than an invisible identity fallback. The child should know when the request exceeds the medium.

---
#### Subcategory 8.4: Task-specialized small models vs prompted frontier on structured outputs
Source: ServiceNow (2025) Fine-Tune an SLM or Prompt an LLM? (low-code workflows); Geng et al. (2025) JSONSchemaBench; Bai et al. (2024) LongWriter.
- DOK 1 - facts:
- A fine-tuned ~12B SLM beat prompted GPT-4o/Gemini/Llama-70B by ~10% on low-code JSON workflows; prompted GPT-4o emitted structurally invalid JSON on ~17% of test workflows.
- Across ~10K real JSON schemas, constrained-decoding coverage collapsed from ~86% (simple) to ~3% (complex), and structural validity must be measured separately from output quality.
- Maximum reliable output length is bounded by SFT example length; very long dense targets fail to complete (truncation) rather than producing wrong content.
- Links: https://arxiv.org/abs/2505.24189 ; https://arxiv.org/abs/2501.10868 ; https://arxiv.org/abs/2408.07055
- ! DOK 2 - summary: Grounds SPOV 6. On bounded structured-output tasks the frontier's failure is emitting a valid complete artifact, not the underlying judgment; the tuned model's edge is a compact reliable output contract - but it must be proven against a frontier baseline in the right format, not only raw dense emission.

---
#### Subcategory 8.5: A metric is only honest once stratified by the population it serves
Source: Oakden-Rayner et al. (2020) Hidden Stratification; Mitchell et al. (2019) Model Cards for Model Reporting; Sagawa et al. (2020) Distributionally Robust Neural Networks (Group DRO); Google SRE Workbook, Error Budget Policy.
- DOK 1 - facts:
- A model can hit near-perfect aggregate AUC while a clinically important subclass underperforms by 30+ points (hidden stratification).
- Model Cards call for disaggregated metrics reported with confidence intervals across relevant populations.
- Group DRO optimizes worst-group loss precisely because average/served-subset metrics hide tail failures.
- SRE error-budget policy sanctions documented, out-of-scope exceptions (for example, load-test traffic) as distinct from renegotiating the target.
- Links: https://arxiv.org/abs/1909.12475 ; https://arxiv.org/abs/1810.03993 ; https://arxiv.org/abs/1911.08731 ; https://sre.google/workbook/error-budget-policy/
- ! DOK 2 - summary: Grounds SPOV 5. Stratifying a gate by product tier is legitimate only when the tier is frozen, provenance-defined, independent of the gate's failure axis, and reported with N/CI; otherwise a “stratified pass” is a post-hoc subgroup rationalization (garden of forking paths) or a threshold edit relocated one layer up.

## DOK 1 - facts
- A single global LUT maps the same input RGB to the same output RGB everywhere.
- Local edits require masks, segmentation, spatial gating, multiple LUTs, local filters, or inpainting/replacement tools.
- A 17x17x17 LUT has 4,913 grid points and 14,739 channel values.
- The v1 tokenizer target is a 4x4x4 latent grid, producing 64 LUT code tokens.
- The LUT codebook size is 256, frozen in the tokenizer manifest.
- The model output is either <lut_bos> plus exactly 64 code tokens plus <lut_eos>, or <unsupported>. A raw dense 17x17x17 `.cube` is 4,913 rows (~100K output floats/tokens); the tokenized form is 64 VQ tokens (generation budget 67 = bos + 64 + eos + base-model EOS).
- Tokenizer acceptance threshold in the local behavior spec is mean reconstruction DeltaE <= 2.0 on held-out LUTs.
- AceTone: Bridging Words and Colors for Conditional Image Grading was posted to arXiv on April 1, 2026 by Tianren Ma, Mingxiang Liao, Xijin Zhang, and Qixiang Ye.
- AceTone is the original direct paper to track for this project because it connects natural-language/image-conditioned grading to discrete 3D-LUT generation.
- AceTone reports a VQ-VAE LUT tokenizer that maps a 3x32^3 LUT vector into 64 discrete tokens with DeltaE < 2 fidelity.
- The SFT checkpoint target includes at least 85% free-generation valid-token output rate, measured with confidence intervals.
- The SFT checkpoint target includes unsupported recall, unsupported precision, boundary F1, mixed-prompt recall, and an over-refusal ceiling.
- The SFT checkpoint must beat null/constant baselines and the deterministic renderer by predeclared paired-CI gates, not only a prompted-Qwen baseline.
- RS/DPO should be tried before GRPO; GRPO should improve prompt-to-LUT pass rate or safety failure rate over the best prior tuned stage outside paired confidence intervals.
- Aesthetic reward is last-priority and gets no credit when tokens are invalid, prompt direction is wrong, LUTs are unsafe, or refusals fail.
- Token meaning is bound by `vq_codebook_sha256`, `vq_decoder_sha256`, flatten order, `.cube` serialization, ICC conversion config, and deterministic environment.
- PPR10K contains 11,161 RAW portrait photos and three expert target sets, producing 33,483 candidate expert targets.
- PPR10K includes 1,681 photo groups and full-resolution human-region masks.
- PPR10K’s dataset files are non-commercial research only; the code license does not override the dataset restriction.
- MIT-Adobe FiveK contains 5,000 SLR photographs with five expert retouched renditions, producing 25,000 expert outputs.
- FiveK canonical inputs are DNG RAW; canonical individual expert outputs are TIFF16 ProPhoto RGB.
- FiveK image licenses are research-only/non-commercial.
- FreshLUTs terms claim CC0 for uploaded LUTs, but uploader provenance may be missing.
- RawTherapee Film Simulation uses HaldCLUT PNG/TIFF reference images and excludes local/detail/geometric operations.
- Child-facing products require safety, privacy, fairness, transparency, accountability, and inclusion.
- COPPA applies to many U.S. child-facing services under 13 and requires strict data collection, retention, consent, and security practices.
- Trait praise can undermine children’s persistence after failure; process feedback is safer.
- The strongest learning claim is near visual transfer: noticing, explaining, comparing, and revising on similar visual cases.
- The strongest development claim is process growth: closer observation, more precise vocabulary, better prediction, and more intentional revision.
- A child-development study should use new images, child explanations, prediction tasks, and revision traces rather than taste scores.
- Natural-language input is justified when it captures child intent that expert controls hide behind terms like curves, black point, saturation, contrast, and clarity.
- A child phrase such as “make it sharper” should be resolved into a supported color behavior such as more contrast, or refused if the child is asking for texture/detail sharpening.
- The natural-language interface should be evaluated by intent capture, correct mapping to supported color attributes, boundary handling, and whether the child can later name the visible change.
- A fine-tuned prompt-to-LUT model is not enough for the educational claim; the claim depends on a GUI where the child creates a LUT, applies it to their uploaded image, and compares the same LUT across several other image categories.
- The v1 LUT tokenizer is frozen as `vq_v2_srgbres_17to4_cb256_t64` in the canonical domain `slm_lut_v1_srgb_display_encoded_17_trilinear`, with held-out reconstruction mean ΔE00 ≈ 1.30, p95 ≈ 2.80, PSNR ≈ 36.7 dB.
- The frozen tokenizer carries a recorded gate-owner exception on `p5_psnr` (measured 28.37 vs the ≥30 gate on the full holdout); the failing worst-5% is entirely diagnostic-only, and on gold/headline-eligible LUTs it passes at p5_psnr = 31.24.
- The active SFT set targets 10k-15k examples (12k default); the current pool is a demo-scale 3,033 rows (2,761 supported + 272 unsupported ≈ 9%), with a realized family mix led by ppr10k_derived and scraped_web, max-family fraction 0.32, and ppr10k+fivek 0.49 ≤ 0.50 cap.
- Instruction data is distilled from a teacher model (`claude-sonnet-4-6`) and judged by an LLM judge (`claude-opus-4-8`); deterministic tag↔behavior color checks are authoritative and the judge cannot overrule them. Supported rows carry materialized 64-token targets.
- In the prompted-frontier baseline, every fully-completed 17³ LUT was direction- and safety-correct and every failure was truncation or refusal; the best model completed 4/5 single-attribute rows at ~$1.4-2.4 and 8-17 min each, and one route is structurally output-token-capped (~65K) and cannot emit a full 17³ regardless of prompt.
- The materialized dataset is hosted at `hf://datasets/ericrcwu/LUT_SLM`.

## Ways this could fail
1. Slider baseline risk: a slider/preset playground may support the same child learning outcomes with less model complexity.
2. Frontier structured-output risk: the free-form prompted-frontier baseline has now been run - frontier models can produce correct global-color LUTs but fail to emit the full 17³ artifact reliably (best 4/5, ~$1.4-2.4 and 8-17 min each; some routes are structurally output-token-capped). The still-open risk is a frontier *structured-output / tool / recipe-mode / smaller-grid-then-resample* baseline that de-confounds the token ceiling; if one matches the tuned model, the claim narrows to output-budget, reliability, cost, local operation, and deterministic auditable artifacts rather than "fine-tuning is required." A deterministic attribute renderer remains a required baseline for the same reason.
3. Prompt-label risk: PPR10K and FiveK have no natural-language prompts; labels must be validated against measured LUT behavior.
4. Licensing risk: PPR10K and FiveK are research-only. Any commercial or public product path needs separately licensed data or legal clearance.
5. Skin-tone risk: generic color metrics may miss harmful changes to people. Skin/face-region checks and diverse review are required.
6. Reward-hacking risk: GRPO can learn to exploit the reward tuple unless invalid outputs are hard-rejected and reward-hacking evals are held out.
7. Over-refusal risk: a model can look safe by refusing too much. Supported-prompt coverage and over-refusal rate must be reported.
8. Far-transfer risk: any claim beyond near visual discrimination requires a separate child study.
9. Natural-language ambiguity risk: if the system maps vague child language to edits without explanation, it becomes prompt magic. The output needs to show the child which color concept changed.
10. Strawman-baseline risk (SPOV 6): the frontier moat was measured only against raw-dense-17³ prompting. If a fair structured-output frontier baseline - a parametric recipe or a coarse 5³/9³ grid expanded by the deterministic renderer - completes reliably and cheaply, the output-budget and reliability-at-budget legs collapse and the claim must fall back to locality, cost/latency, and auditability. That baseline is designed (renderer/recipe modes) but deliberately deferred to a post-training stage, where it is run as a head-to-head on applied-LUT image results once the SLM exists.
11. Circular-gate risk (SPOV 5): the frozen tokenizer's `p5_psnr` exception rests on a tier partition partly derived from the same construct the gate measures, on a gold boundary loosened for yield, and on a post-hoc subgroup pass reported without a confidence interval. If a pre-registered, provenance-only, metric-independent tier still fails, the exception was unjustified and clearing the gate needs a capacity bump (new codebook/latent), which cascades a retokenization of every target.
12. Train/eval tier-mismatch risk: at demo scale ~95% of supported training rows are `diagnostic_only` while headline eval is gold-only, so the model may train mostly on the non-product LUTs the tokenizer reconstructs worst, then be scored on the thin gold slice it barely trained on.
13. Distillation capacity-gap risk: distilling one of the strongest available teachers (Sonnet 4.6) into a 3B student is the regime where a stronger teacher does not guarantee a stronger student; the reliability may come from the deterministic gates and the tiny 64-token grammar rather than from distilled teacher reasoning.

## References
### Learning science and arts education
1. Barnett, S. M., & Ceci, S. J. (2002). When and where do we apply what we learn? https://doi.org/10.1037/0033-2909.128.4.612
2. Gentner, D., Loewenstein, J., & Thompson, L. (2003). Learning and transfer: A general role for analogical encoding. https://doi.org/10.1037/0022-0663.95.2.393
3. Kornell, N., & Bjork, R. A. (2008). Learning concepts and categories. https://doi.org/10.1111/j.1467-9280.2008.02127.x
4. Kellman, P. J., & Garrigan, P. (2009). Perceptual learning and human expertise. https://doi.org/10.1016/j.plrev.2008.12.001
5. Alfieri, L., Brooks, P. J., Aldrich, N. J., & Tenenbaum, H. R. (2011). Does discovery-based instruction enhance learning? https://doi.org/10.1037/a0021017
6. Winner, E., Goldstein, T. R., & Vincent-Lancrin, S. (2013). Art for Art’s Sake? https://doi.org/10.1787/9789264180789-en

### Motivation, rights, and equity
1. Ryan, R. M., & Deci, E. L. (2000). Self-determination theory and the facilitation of intrinsic motivation. https://doi.org/10.1037/0003-066X.55.1.68
2. Patall, E. A., Cooper, H., & Robinson, J. C. (2008). The effects of choice on intrinsic motivation and related outcomes. https://doi.org/10.1037/0033-2909.134.2.270
3. Mueller, C. M., & Dweck, C. S. (1998). Praise for intelligence can undermine children's motivation and performance. https://doi.org/10.1037/0022-3514.75.1.33
4. Shute, V. J. (2008). Focus on formative feedback. https://doi.org/10.3102/0034654307313795
5. Ladson-Billings, G. (1995). Toward a theory of culturally relevant pedagogy. https://doi.org/10.3102/00028312032003465
6. Buolamwini, J., & Gebru, T. (2018). Gender Shades. https://proceedings.mlr.press/v81/buolamwini18a.html
7. Thong, W., Joniak, P., & Xiang, A. (2023). Beyond Skin Tone. https://openaccess.thecvf.com/content/ICCV2023/html/Thong_Beyond_Skin_Tone_A_Multidimensional_Measure_of_Apparent_Skin_Color_ICCV_2023_paper.html
8. Resnick et al. (2009). Scratch: Programming for All. https://doi.org/10.1145/1592761.1592779
9. Druin, A. (1999). Cooperative inquiry: developing new technologies for children with children. https://doi.org/10.1145/302979.303166

### LUTs, data, and training
1. NVIDIA GPU Gems 2, Chapter 24. https://developer.nvidia.com/gpugems/gpugems2/part-iii-high-quality-rendering/chapter-24-using-lookup-tables-accelerate-color
2. RawPedia Film Simulation / HaldCLUT. https://rawpedia.rawtherapee.com/Film_Simulation
3. CIE 015:2018, Colorimetry. https://doi.org/10.25039/TR.015.2018
4. Sharma, G., Wu, W., & Dalal, E. N. (2005). CIEDE2000 implementation notes. https://doi.org/10.1002/col.20070
5. ACES Common LUT Format. https://docs.acescentral.com/specifications/clf/
6. Zeng et al. Learning Image-Adaptive 3D Lookup Tables. https://doi.org/10.1109/TPAMI.2020.3026740
7. Wang et al. Spatial-Aware 3D LUTs. https://arxiv.org/abs/2309.15662
8. PPR10K. https://doi.org/10.1109/CVPR46437.2021.00071
9. MIT-Adobe FiveK. https://doi.org/10.1109/CVPR.2011.5995332
10. Ma, T., Liao, M., Zhang, X., & Ye, Q. (2026). AceTone: Bridging Words and Colors for Conditional Image Grading. https://arxiv.org/abs/2604.00530
11. VQ-VAE. https://arxiv.org/abs/1711.00937
12. Qwen2.5-VL Technical Report. https://arxiv.org/abs/2502.13923
13. LoRA. https://arxiv.org/abs/2106.09685
14. DeepSeekMath / GRPO. https://arxiv.org/abs/2402.03300

### Evaluation
1. IFEval. https://arxiv.org/abs/2311.07911
2. HELM. https://arxiv.org/abs/2211.09110
3. Geifman, Y., & El-Yaniv, R. (2017). Selective Classification for Deep Neural Networks. https://proceedings.neurips.cc/paper/2017/hash/4a8423d5e91fda00bb7e46540e2b0cf1-Abstract.html
4. Nielsen Norman Group, visibility of system status. https://www.nngroup.com/articles/visibility-system-status/

### Model specialization, data curation, and evaluation methodology
1. ServiceNow (2025). Fine-Tune an SLM or Prompt an LLM? The Case of Generating Low-Code Workflows. https://arxiv.org/abs/2505.24189
2. Geng et al. (2025). JSONSchemaBench: A Rigorous Benchmark of Structured Outputs for Language Models. https://arxiv.org/abs/2501.10868
3. Bai et al. (2024). LongWriter: Unleashing 10,000+ Word Generation from Long Context LLMs. https://arxiv.org/abs/2408.07055
4. Zhou et al. (2023). LIMA: Less Is More for Alignment. https://arxiv.org/abs/2305.11206
5. Gunasekar et al. (2023). Textbooks Are All You Need (phi-1). https://arxiv.org/abs/2306.11644
6. Hsieh et al. (2023). Distilling Step-by-Step! https://arxiv.org/abs/2305.02301
7. Renduchintala et al. (2023). INGENIOUS: Informative Data Subsets for Efficient Pre-Training. https://arxiv.org/abs/2305.06677
8. Li et al. (2023). Self-Alignment with Instruction Backtranslation (counterpoint to curation-over-scale). https://arxiv.org/abs/2308.06259
9. Busbridge et al. (2025). Distillation Scaling Laws (teacher-student capacity gap). https://arxiv.org/abs/2502.08606
10. Oakden-Rayner et al. (2020). Hidden Stratification Causes Clinically Meaningful Failures in ML for Medical Imaging. https://arxiv.org/abs/1909.12475
11. Mitchell et al. (2019). Model Cards for Model Reporting. https://arxiv.org/abs/1810.03993
12. Sagawa et al. (2020). Distributionally Robust Neural Networks for Group Shifts (Group DRO). https://arxiv.org/abs/1911.08731
13. Google SRE Workbook. Error Budget Policy. https://sre.google/workbook/error-budget-policy/
