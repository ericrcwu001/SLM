# Training Sequence

Status: Superseded by ADR 0025 (training sequence v2). Previously partially superseded by
ADR 0015 and the current `docs/training_plan_colab.md`.

The older sequence moved from tokenizer training directly into 50,000
instruction examples, SFT, and a small GRPO stage. The current sequence keeps the
tokenizer-before-SFT dependency, but supersedes the scale and rollout order.

Current v1 sequence:

1. Build eval harness and smoke eval rows.
2. Derive, canonicalize, and representability-filter LUTs.
3. Create split units and reserve eval/diagnostic/qualitative identities.
4. Train and freeze the VQ tokenizer after mean, tail, per-family, per-target,
   codebook, and roundtrip gates pass.
5. Resize vocabulary and run embedding/head preflight assertions.
6. Build the active 10k-15k instruction SFT set and freeze final eval rows.
7. Materialize 30k-100k train-only warmup rows.
8. Run generative LUT-token warmup.
9. Run QLoRA SFT.
10. Run image-blind/shuffled-image ablations.
11. Run RS/DPO before GRPO.
12. Run GRPO only if RS/DPO plateaus and reward correctness is proven.
13. Package the CLI.
