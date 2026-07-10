# Training Sequence v2

Status: Accepted. Supersedes ADR 0009 (training sequence). Related: ADR 0020, 0024.

The two-stage architecture (ADR 0020) adds an interpreter and changes the generator's input, so the
training sequence and the repo's stage numbering must be revised in lockstep across
`docs/master_plan.md` (Stages) and `docs/training_plan_colab.md` (Stages), and this ADR.

Guiding order (from `docs/AUDIT_claude_codex_prompt_to_lut.md` §10/§13): make the metric honest
before any retrain, and prove the seam is not lossy before spending on the interpreter/generator.

Current v2 sequence:

1. Build eval harness and smoke eval rows.
2. Derive, canonicalize, and representability-filter LUTs.
3. Create split units and reserve eval/diagnostic/qualitative identities.
4. Train and freeze the VQ tokenizer (unchanged; ADR 0017).
5. Resize vocabulary and run embedding/head preflight assertions.
6. **Eval honesty (ADR 0024):** unit-aware holdout, full per-slice scoring, exact-64 assertion,
   OOD/refuse slices — before any retrain.
7. **Refuse path becomes load-bearing (ADR 0023):** portable staging so unsupported rows train;
   out-of-gamut + clarify taxonomy.
8. **behavior_v2 (ADR 0022):** implement the new axes; re-measure into a new versioned artifact.
9. **AttributeSpec + captioning (ADR 0021, 0026):** schema/serializer; caption corpus LUTs.
   **Oracle `attribute_spec → codes` upper-bound gate — HARD go/no-go before step 10/11.**
10. **Interpreter distillation:** train the small LM on `(caption → AttributeSpec + route)`.
11. **Generator retrain:** same Qwen QLoRA, input = `attribute_spec_text` (locked knobs unchanged).
12. Run image-blind/shuffled-image ablations.
13. End-to-end two-stage integration + CLI; then RS/DPO before GRPO only if warranted.
14. Package the CLI.

Consequences: eval-honesty and refuse work ship before compute is spent; the oracle gate can abort
the two-stage move cheaply; the tokenizer freeze (ADR 0017) and generator locked knobs are
unchanged.
