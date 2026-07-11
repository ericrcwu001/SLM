"""Interpreter training config — a STANDALONE dataclass, deliberately not ``sft.config.SFTConfig``.

``SFTConfig.__post_init__`` *raises* unless ``epochs==2``, ``num_new_tokens==259``, and the base is
the Qwen VLM — all generator locks that make no sense for a text-only 0.5B interpreter. Overloading
it would either break those invariants or pollute the generator's locked config (AGENTS.md forbids).
So the interpreter gets its own config; only the YAML-override *loader shape* mirrors
``sft.config.load_config``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path("configs/interpreter_default.yaml")

_TUNING_MODES = ("full", "lora")
_SCHEDULERS = ("cosine", "linear", "constant")


@dataclass
class InterpreterConfig:
    # Base model (text-only CausalLM). Qwen2.5-0.5B-Instruct: zero new deps, same tokenizer family
    # as the generator, ships with a chat template. Swap to LFM2.5-350M-Base later if size demands.
    base_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tuning_mode: str = "full"          # "full" (default; a 0.5B model needs no QLoRA) | "lora"
    # LoRA knobs (only used when tuning_mode == "lora"; requires the peft dep).
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    # Optimization (NOT locked to the generator's epochs=2 / batch triple).
    epochs: int = 3
    learning_rate: float = 1e-5
    per_device_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    max_seq_len: int = 256             # caption + short spec target fit comfortably
    warmup_ratio: float = 0.03
    scheduler: str = "cosine"
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = False
    seed: int = 0
    # Data / holdout / IO.
    holdout_frac: float = 0.06         # same fraction as the generator (sft.holdout.DEFAULT_HOLDOUT_FRAC)
    corpus_path: str = "data/interpreter/interpreter_rows.jsonl"
    out_dir: str = "models/interpreter"
    # Class balance: repeat non-grade (refuse/clarify) rows N× in the train mix (grade dominates).
    upsample_nongrade: int = 4
    # Generation (scoring).
    max_new_tokens: int = 64

    def __post_init__(self) -> None:
        if self.tuning_mode not in _TUNING_MODES:
            raise ValueError(f"tuning_mode must be one of {_TUNING_MODES}, got {self.tuning_mode!r}")
        if self.scheduler not in _SCHEDULERS:
            raise ValueError(f"scheduler must be one of {_SCHEDULERS}, got {self.scheduler!r}")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.upsample_nongrade < 1:
            raise ValueError("upsample_nongrade must be >= 1")


def load_config(path: str | Path | None = None) -> InterpreterConfig:
    """Load an :class:`InterpreterConfig`, applying YAML overrides for known fields only.

    Mirrors :func:`sft.config.load_config`: filter to declared fields, coerce lists to tuples.
    Shared by the trainer and scorer so their config parsing cannot drift.
    """
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    overrides = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}
    field_names = {f.name for f in fields(InterpreterConfig)}
    kw = {k: (tuple(v) if isinstance(v, list) else v)
          for k, v in overrides.items() if k in field_names}
    return InterpreterConfig(**kw)
