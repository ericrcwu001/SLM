# GRPO Reward Priority

Status: Accepted. Authoritative for RS/DPO/GRPO reward priority.

Reward priority is lexicographic:

1. valid 64-token output or valid `<unsupported>`;
2. correct support/refusal boundary;
3. prompt-direction correctness;
4. LUT safety;
5. target fidelity;
6. style discriminability;
7. small aesthetic preference.

This deliberately differs from a simple color-similarity-plus-aesthetic reward
because the project's target behavior is reliable prompt-to-LUT control.
Aesthetic preference and target fidelity must not compensate for invalid tokens,
wrong directions, unsafe global LUTs, or refusal failures.
