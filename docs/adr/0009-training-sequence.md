# Training Sequence

Status: Partially superseded by ADR 0015 and the current
`docs/training_plan_colab.md`.

The older sequence moved from tokenizer training directly into 50,000
instruction examples, SFT, and a small GRPO stage. The current sequence keeps the
tokenizer-before-SFT dependency, but supersedes the scale and rollout order.

Current v1 sequence:

1. Build eval harness and frozen eval rows.
2. Derive, canonicalize, and representability-filter LUTs.
3. Train and freeze the VQ tokenizer after mean, tail, per-family, per-target,
   codebook, and roundtrip gates pass.
4. Resize vocabulary and run embedding/head preflight assertions.
5. Run 30k-100k generative LUT-token warmup.
6. Build the active 10k-15k instruction SFT set.
7. Run QLoRA SFT.
8. Run image-blind/shuffled-image ablations.
9. Run RS/DPO before GRPO.
10. Run GRPO only if RS/DPO plateaus and reward correctness is proven.
11. Package the CLI.
