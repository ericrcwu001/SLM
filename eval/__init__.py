"""Prompt-to-LUT eval harness (Stage 1 spine).

Model- and decode-independent. The token->LUT decoder and every color-scoring layer
(L2-L8) are present as guarded, disabled interfaces until the VQ tokenizer is frozen.
See docs/eval_harness_implementation.md and the approved plan.
"""

__version__ = "0.1.0"
