# Refuse/Clarify Boundary Including Out-Of-Gamut; Parametric Renderer Deferred

Status: Accepted. Amends ADR 0014 (unsupported prompt boundary).

Today the refuse path never trains: the 272 unsupported rows carry absolute image paths that fail
to resolve on Colab and are skipped every epoch, so the deployed model neither supports nor cleanly
refuses out-of-scope requests — it emits a muted LUT
(`docs/AUDIT_claude_codex_prompt_to_lut.md` F2). There is also no handling for globally-representable
but out-of-gamut looks (infrared, pure-primary casts, hue-rotation), which the frozen tokenizer
cannot represent (AUDIT F3: probe ΔE00 13–16; materialization writes nothing above mean 3.0 /
p95 6.0).

Decision:

- The route is three-way: **`grade` / `clarify` / `refuse`**, and the refuse path is
  **load-bearing** — it must be trained and measured (over-refusal, out-of-gamut recall,
  boundary F1).
- **`refuse:out_of_scope`** keeps the existing taxonomy (local/semantic/content/relighting/
  geometry/texture/reference).
- **`refuse:out_of_gamut`** is NEW: a global color look whose nearest representable LUT exceeds the
  materialization admission gate (`scripts/materialize_target_tokens.py:50`, mean ΔE00 ≤ 3.0 /
  p95 ≤ 6.0) is refused. The refuse route only *reads* this boundary; it never modifies the
  tokenizer.
- **`clarify`** handles under-specified color intent ("make it better") — offer supported
  directions instead of fabricating a grade.
- **The parametric recipe→LUT renderer is explicitly deferred.** Out-of-gamut looks are refused,
  not rendered. Partly-representable concepts (e.g. "Mars") are clamped to the nearest representable
  point and graded, not refused; which concepts are supported is defined empirically by which corpus
  LUTs materialize.

Consequences: the taxonomy strings must stay in sync across `data_pipeline/unsupported_gen.py`,
`scripts/generate_unsupported.py`, `eval/fixtures/make_smoke_rows.py`,
`eval/unsupported_metrics.py`, and `tests/test_unsupported_gen.py` (a sync test is added in the code
phase). Reversible later: if a renderer or a v2 tokenizer is built, the `out_of_gamut` bucket is
simply re-routed to it — no change to the interpreter or generator.
