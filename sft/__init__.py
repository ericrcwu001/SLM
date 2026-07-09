"""QLoRA SFT stage for the prompt-to-LUT VLM (master-plan Stage 14 / training_plan_colab.md
"Stage 5"). Vocabulary resize + preflight, the resumable QLoRA training loop, and the adapter
manifest live here; the pinned hyperparameters are in :mod:`sft.config`.

Heavy deps (transformers/peft/trl/bitsandbytes/accelerate) are the ``sft`` extra and are
imported lazily inside the functions that need them, so importing this package (and
:mod:`sft.config`) stays dependency-light.
"""
