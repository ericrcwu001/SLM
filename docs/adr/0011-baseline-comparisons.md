# Baseline Comparisons

Status: Amended.

Evaluation compares the tuned prompt-to-LUT model against null, constant,
deterministic, prompted, frontier, image-blind, and tuned-stage baselines.

Required baselines include always-unsupported, identity-all-prompts, train-mean
constant LUT, dev-optimized constant LUTs, a deterministic attribute renderer,
Qwen token/raw/recipe modes, prompt-only or blank/shuffled-image ablations, and a
frontier prompted baseline when available.

The SFT gate is not allowed to depend only on prompted Qwen. It must beat null
and constant baselines and the deterministic renderer by predeclared paired-CI
thresholds. If a prompted frontier baseline matches or beats the tuned model, the
project must narrow its claim to local/offline/cost/artifact advantages rather
than claiming fine-tuning is required for behavior reliability.
