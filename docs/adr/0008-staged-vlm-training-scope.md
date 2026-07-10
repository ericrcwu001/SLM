# Staged VLM Training Scope

Status: Amended by ADR 0007, ADR 0015, ADR 0020 (two-stage interpreter + generator), and the current training plan.

V1 starts by freezing the vision encoder while applying LoRA to the language
model and adapting the multimodal projector/connector. Full language-model
fine-tuning and full-model fine-tuning are reserved for later scale-up after the
tokenizer, SFT, evaluation, and simpler tuned stages show proof of concept.

RS/DPO precedes optional GRPO. GRPO runs only after the best tuned stage plateaus
and reward correctness passes.
