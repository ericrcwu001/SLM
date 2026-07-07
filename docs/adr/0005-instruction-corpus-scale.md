# Instruction Corpus Scale

Status: Superseded by ADR 0015.

The older decision targeted 50,000 instruction examples for v1, with a later
scale-up to 100,000 examples. That target has been superseded.

Current v1 uses an active 10k-15k instruction SFT set, default 12k, plus a
separate 30k-100k generative LUT-token warmup. The 50k/100k instruction corpus
sizes are now scale-up milestones after the tokenizer, warmup, SFT, eval, and CLI
loop work reliably.
