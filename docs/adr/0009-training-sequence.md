# Training Sequence

The v1 pipeline runs in this order: collect and derive LUTs, normalize them to 17^3, convert to residual LUTs, train the VQ tokenizer, pass tokenizer acceptance gates, generate 50,000 instruction examples, run VLM SFT, evaluate base versus SFT, run a small GRPO stage, evaluate base versus SFT versus GRPO, and package the demo/results. VLM SFT does not begin before tokenizer quality is acceptable, and GRPO does not begin before SFT shows basic validity and direction-following gains.
