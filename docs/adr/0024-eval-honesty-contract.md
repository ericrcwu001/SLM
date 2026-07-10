# Eval-Honesty Contract

Status: Accepted. Amends ADR 0016 (eval splits and provenance contract) and ADR 0011 (baseline
comparisons).

The optimized metric cannot see the failure it is meant to catch
(`docs/AUDIT_claude_codex_prompt_to_lut.md` F4/F5): the SFT holdout is a random per-row SHA-1 carve
that ignores the pipeline's leakage-safe `split_unit_id` (48.5% of held-out rows share a
near-duplicate LUT unit with training); it is teacher-forced, exact-match, in-distribution, and
scored on a source-biased first-48 slice; and the eval harness never scores LUT quality at all.

Decision — the following become contractual:

- **Unit-aware holdout:** `sft/holdout.py` keys on `split_unit_id`, not the row id, so
  near-duplicate LUTs cannot straddle the train/holdout boundary. Expect (and accept) a headline
  drop that quantifies prior inflation.
- **Full, stratified scoring:** score all held-out rows (drop the default `--limit 48`), report
  macro per-slice / per-family accuracy with group-bootstrap CIs.
- **Exact-64 assertion:** every scored supported row must have exactly 64 surviving code positions
  (closes the partial-truncation blind spot, AUDIT F8).
- **New eval slices** (decoder-free): unseen-wording, named-concept, nonce-concept,
  counterfactual-ranking, paraphrase-consistency, refuse (out-of-scope + out-of-gamut), and an
  in-distribution regression guard. Declared in `eval/configs/gating_slice_registry.yaml`.
- **Interpreter metrics:** attribute-F1 (requested vs measured axes), route accuracy, over-refusal.
- The two untested choke points (`sft/example.py`, `sft/score_tokens.py`) get tests before any
  retrain.

The frozen decoder stays disabled (`eval/lut_decoder.py`); perceptual scoring uses decoder-free
proxies, with any read-only decode-for-eval left owner-gated and out of v1.
