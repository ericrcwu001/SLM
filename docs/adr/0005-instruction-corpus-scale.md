# Instruction Corpus Scale

The v1 SFT corpus target is 50,000 instruction examples, with a later scale-up to 100,000 examples after the 50,000-example pipeline proves the tokenizer, prompt generation, filtering, training, and evaluation loop. This scale is intentionally below AceTone's reported 800,000 instruction tuples but large enough to show a meaningful base-vs-tuned prompt-to-LUT behavior delta.
