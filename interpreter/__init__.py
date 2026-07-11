"""Stage-1 interpreter: a separate text-only LM mapping user free-text -> attribute_spec_text + route.

Decoupled from the Stage-2 generator (Qwen2.5-VL-3B) by the frozen seam
:mod:`data_pipeline.attribute_spec`. This package is text-only (``AutoModelForCausalLM``); it never
touches the VLM stack, the VQ tokenizer, or the locked generator config (``sft.config.SFTConfig``).
"""
