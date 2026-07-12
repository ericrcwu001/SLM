# Corpus Audit — do the three fixed data-pipeline bugs require a rebuild?

**Scope:** read-only audit of the **frozen, materialized** corpus against the *fixed* pipeline code
(commit `9902f02`, branch `feat/two-stage`). No rebuild, no re-derivation. All numbers are computed
from local metadata (`data/raw_registry/provenance.jsonl`, `data/active_sft/active_rows.jsonl`) and
the on-disk `.xmp` sidecars, using the actual fixed predicates
(`sft.holdout.is_holdout_row`, `data_pipeline.sources.derive.parse_xmp`, and the verbatim
`representability.PAIR_FIT` thresholds).

**Corpus under audit:** 3033 active SFT rows; 7760 registry rows (4855 `pair_fit`: 4055 PPR10K +
800 FiveK). Baselines the audit is protecting: token-acc **0.329**, free-running behavioral fidelity
**0.159**, oracle@32 **0.30** (holdout keyed on `split_unit_id`, `frac=0.06`).

## TL;DR — **PROCEED on the current corpus. Do not rebuild.**

| Audit | What the bug would change | Impact on frozen corpus | Verdict |
|-------|---------------------------|-------------------------|---------|
| **1 — FiveK double-ingest → holdout leakage** | duplicate photos straddling train/holdout | **0** duplicate photos, **0** leaking clusters | baselines **NOT** invalidated |
| **2 — fit_val vs fit_all admission** | rows admitted on in-sample fit that fail held-out | **1** currently-accepted row flips (≤0.03% of corpus); **0** gold rows | negligible → PROCEED |
| **3 — XMP local-edit hard-reject** | PPR10K local edits pair-fit into global LUTs | **3** genuinely-local rows (0.10% of corpus) | negligible → PROCEED |

All three bugs' impact on the *existing* corpus is **< 5%** ⇒ the decision rule says **PROCEED**.
The fixes are correct improvements, but they change essentially nothing about what is already frozen.

> ⚠️ **One thing that is NOT a corpus problem but IS a landmine for any future rebuild:** the wired
> `parse_xmp` hard-reject (Audit 3), *as coded*, would reject **100% of PPR10K** (31.5% of the
> corpus) because it substring-matches Lightroom sidecar **defaults**, not active local edits. Fix
> its value-awareness before it is ever run in a rebuild. Details in Audit 3.

---

## AUDIT 1 — FiveK double-ingest + holdout leakage (highest priority)

**Fix recap.** `run_acquire.py` now lists `fivek_expert_abcde` in `FALLBACK_SOURCES`, so the HF
connector runs only if the Kaggle primary acquires nothing (previously both ran → double-ingest
risk). Holdout is unit-aware: `is_holdout_row` keys on `split_unit_id` (`sft/holdout.py`).

**(a) FiveK rows by connector.** All 800 `fivek_derived` registry rows come from a single source:

```
source_url_or_dataset:  kaggle://weipengzhang/adobe-fivek   → 800  (Kaggle primary)
                        (HF connector)                       →   0
```
A full scan for any `fivek`+`hf`/`huggingface` reference across every registry row returns **0**.
**The double-ingest never happened** — only the primary connector produced rows, exactly as the fix
intends. (All 800 are expert `c`; experts a–e were not multi-ingested.)

**(b) Duplicate base photos.** Grouping the 800 rows by base-photo key (`source_photo_id`, cross-
checked against `source_pair_id` and the `source_image_path` stem):

```
distinct base photos:            800
base photos appearing >1×:         0
duplicate source_pair_id / stem:   0
```
Every underlying FiveK photo appears **exactly once**. There is nothing to leak.

**(c) Leakage in the active corpus.** 538 `fivek_derived` rows are in the active corpus. Grouped by
base-photo key (`image_path` stem) and evaluated with `is_holdout_row`:

```
active fivek rows:                              538
distinct base photos:                           538   (each photo → exactly 1 active row)
base-photo clusters with >1 row:                  0
clusters straddling holdout + train:              0
clusters carrying >1 distinct split_unit_id:      0
active fivek rows landing in holdout:            16 / 538
```
Because each base photo maps to a single active row and a single `split_unit_id`, **no cluster can
straddle the holdout boundary**. Where photos *do* share a split unit (286 distinct units; the
largest unions 160 rows), the unit-aware holdout assigns the whole unit one train/holdout decision
by construction — the leakage-safe unioning is working as designed.

**Audit 1 verdict: NO leakage.** The FiveK double-ingest bug did not manifest, and even if it had,
the unit-aware holdout would have prevented straddling. **The 0.329 / 0.159 / 0.30 baselines are not
invalidated by this bug — no re-clustering of `split_unit_id` is needed.**

---

## AUDIT 2 — fit_val vs fit_all admission/tier flips

**Fix recap.** `assess_pair_fit` now gates admission on the **held-out** fit (`fit_val`), not the
in-sample `fit_all`. Reject if `fit_val.mean > 3.0` **or** `fit_val.p95 > 7.0` **or**
`fit_val.p99 > 10.0`; gold requires `fit_val.mean ≤ 2.5` (`PAIR_FIT` verbatim). We compare the two
**already-persisted** metric sets — `fit_deltaE00_*` (fit_all) vs `fit_validation_deltaE00`
(fit_val) — for all 4855 pair-fit rows. **All 4855 carry both metric sets; none are null.**

**(a) Admission flip — accepted under fit_all, would reject under fit_val:**

```
over all pair_fit rows:                 3   (0.06% of pair-fit)
  └ restricted to currently-ACCEPTED:   1   (the only real corpus contamination)
(context) reject_all → accept_val:     10   (would ADD rows, not contaminate)
```
The single accepted flip is a p99 hairline case (`p99: 9.952 → 10.013`, i.e. 0.013 ΔE over the 10.0
gate). ≤ **1 / 3033 = 0.03%** of the corpus.

**(b) Tier flip — gold under fit_all, not gold under fit_val (`mean` crosses 2.5):**

```
over all pair_fit rows:                 2   (0.04% of pair-fit)
  └ currently-GOLD tier rows affected:  0
```
Both are boundary cases (`2.4989 → 2.5027`, `2.4993 → 2.5014`) and neither is currently a gold row,
so **no headline-eligible eval row loses gold.**

**Why so small:** a global LUT fit on all pixels is massively over-determined, so `fit_val ≈ fit_all`
to the third decimal (the code's own rationale). The fix is the honest generalization gate, but on
this corpus it moves essentially nothing.

**Audit 2 verdict: negligible (≤ 0.03% of corpus). PROCEED.**

---

## AUDIT 3 — XMP local-edit fraction (PPR10K target quality)

**Fix recap.** `derive.parse_xmp(text)` is now wired so a PPR10K pair edited with **local** tools
(masks/brush/gradient/retouch) is hard-rejected instead of being pair-fit into a *global* LUT.
`XmpResult.accepted = (parse_status == "parsed" and local_tool_count == 0)`.

**Setup.** All 4055 PPR10K rows are `pair_fit`, all carry `raw_edit_metadata_path`, and **all 4055
`.xmp` files exist locally and parse** (`parse_status="parsed"`). In the frozen registry
`xmp_local_tool_count` is **null on every row** — confirming the hard-reject was never run when the
corpus was built (the bug).

**Literal `parse_xmp` verdict (as-coded):**

```
rows with local_tool_count > 0  →  4055 / 4055  = 100.0% of PPR10K
  └ currently accepted:              2390
  └ active-corpus PPR10K rows:        956  (31.5% of the 3033-row corpus)
```

**This 100% is a false-positive cascade, not a finding that PPR10K is all local edits.** `parse_xmp`
counts a marker by **substring presence** (`if marker in text`), and the markers below are written
as **defaults in every Lightroom/Camera-Raw sidecar** regardless of whether a local tool was used —
several at value `0`:

```
PerspectiveVertical, PerspectiveHorizontal, LensProfileEnable, AutoLateralCA,
PostCropVignetteAmount, Sharpness (=40 default), LuminanceSmoothing (=0),
ColorNoiseReduction (=25), Clarity, Dehaze          → present in all 4055
```
The function's own comment says it should "count only *active* local edits: a list bag or a non-zero
scalar," but the implementation does not check the value. Since PPR10K is the design's primary
global-LUT pair source (4055 of 4855 pair-fit rows), a rule that rejects 100% of it contradicts the
pipeline's intent — the mis-calibration is in the *fix*, not the data.

**Genuine local-edit contamination** (the spatially-masked markers the reject was actually designed
to catch — `MaskGroup`/`Masks`/`Paint*`/`*GradientBasedCorrections`/`Retouch*`/`RedEyeInfo`/
`DustSpots`):

```
PPR10K rows with a genuine local-edit marker:  10 / 4055 = 0.25%   (all PaintBasedCorrections)
  └ currently accepted:                          3
active-corpus rows in those groups:              3   (0.10% of the 3033-row corpus)
```

**Audit 3 verdict: the design-intended contamination is 0.10% of the corpus → negligible → PROCEED.**

> **Blocker for any future rebuild (not a current-corpus problem):** before `parse_xmp` is run in a
> rebuild, make it **value-aware** — count a local tool only when its scalar is non-zero or its list
> bag is non-empty, and drop the always-present global/detail/geometry defaults
> (`Sharpness`/`LuminanceSmoothing`/`ColorNoiseReduction`/`Clarity`/`Dehaze`/`LensProfileEnable`/
> `AutoLateralCA`/`PostCropVignetteAmount`/`Perspective*`) from the local-tool marker list. As
> currently coded it would delete the entire PPR10K family (31.5% of the corpus).

---

## Decision (rule applied)

- **Audit 1 (leakage):** no clusters straddle the holdout ⇒ **no FLAG**; baselines stand, no
  `split_unit_id` re-clustering needed.
- **Audit 2:** ≤ 0.03% of corpus ⇒ **< 5% ⇒ PROCEED**.
- **Audit 3:** genuine local edits 0.10% of corpus ⇒ **< 5% ⇒ PROCEED**.

**Overall: PROCEED on the frozen corpus — do not rebuild.** The three fixed bugs have negligible
impact on what is already materialized, and a rebuild would not address the collapse / exposure-bias
work that is the actual priority. Carry one non-urgent follow-up into the backlog: **fix
`parse_xmp` value-awareness before the next rebuild**, or it will discard all of PPR10K.

---

<sub>Method notes: FiveK connector split from `source_url_or_dataset`; base-photo key from
`source_photo_id` (registry) / `image_path` stem (active); holdout via `sft.holdout.is_holdout_row`
(`frac=0.06`); admission/tier flips from persisted `fit_deltaE00_*` vs `fit_validation_deltaE00`
against `representability.PAIR_FIT`; XMP via `data_pipeline.sources.derive.parse_xmp` on the 4055
on-disk `config.xmp` files; active↔registry PPR10K link via the `ppr10k/global/<group>/` id.
Read-only: no pipeline code or data modified.</sub>
