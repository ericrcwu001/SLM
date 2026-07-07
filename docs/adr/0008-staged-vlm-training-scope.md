# Staged VLM Training Scope

V1 starts by freezing the vision encoder while applying LoRA to the language model and adapting the multimodal projector/connector. Full language-model fine-tuning and full-model fine-tuning are reserved for later scale-up after the tokenizer, SFT, evaluation, and small GRPO loop show proof of concept.
