# Small First GRPO Stage

Status: Amended.

GRPO is no longer the first post-SFT optimization step. V1 first runs rejection
sampling over SFT completions, then DPO from winner/loser pairs if useful. GRPO
is an escalation path only after RS/DPO plateaus, invalid rollout rate is low,
reward hacking checks pass, and improvements are outside confidence intervals.

If GRPO is run, the first stage remains bounded: 1,000 to 3,000 prompts with
four sampled completions per prompt. The run must pin the reference policy, KL
configuration, rollout budget, generation backend, seeds, reward config version, and
eval config. A GRPO checkpoint ships only if it beats the best prior tuned stage
by the CI-gated eval criteria without increasing over-refusal or boundary
failures beyond allowed limits.
