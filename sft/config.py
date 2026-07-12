"""QLoRA SFT configuration for the prompt-to-LUT VLM (training_plan_colab.md "Stage 5").

Pure and dependency-light (no torch/peft/transformers import) so it is import- and test-safe
without the ``sft`` extra, mirroring :mod:`tokenizer.config`. Every hyperparameter the trainer +
adapter manifest commit to lives here. Starting values follow training_plan_colab.md "Stage 5:
SFT With QLoRA" and model_architecture.md "VLM Fine-Tuning Architecture"; the Colab throughput/
credit levers follow "Runtime And Credit Optimization". Epochs are fixed at 2 (not a lever).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

from eval.vocab import NUM_SPECIAL_TOKENS  # 259 = <lut_bos>/<lut_eos>/<unsupported> + <lut_000..255>

DEFAULT_CONFIG_PATH = Path("configs/sft_default.yaml")

# Base model + LoRA targets (model_architecture.md "Base Model" / "LoRA target modules").
BASE_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LORA_TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass(frozen=True)
class SFTConfig:
    """Immutable QLoRA SFT configuration. Geometry-free; all values are Stage-5 starting points."""

    base_model_id: str = BASE_MODEL_ID

    # -- LoRA --
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = LORA_TARGET_MODULES

    # -- quantization (4-bit NF4 QLoRA) --
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"      # -> float16 fallback if bf16 unsupported

    # -- module policy (model_architecture.md "Default module policy") --
    freeze_vision_encoder: bool = True
    projector_policy: str = "lora"                # "lora" | "full" | "frozen" (exploratory only)
    train_new_token_rows: bool = True             # row-selective embed/head for the 259 new rows
    num_new_tokens: int = NUM_SPECIAL_TOKENS

    # -- optimization (Stage 5 starting values) --
    epochs: int = 2                               # FIXED — not a lever
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 32         # pdb*accum == effective_batch_size
    effective_batch_size: int = 32
    learning_rate_lora: float = 2.0e-4
    learning_rate_projector: float = 1.0e-5       # used only when projector_policy == "full"
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    scheduler: str = "cosine"
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True           # memory-safe default; drop only with headroom
    loss_on_assistant_only: bool = True
    seed: int = 0

    # -- Colab throughput / credit levers ("Runtime And Credit Optimization") --
    max_pixels: int = 256 * 28 * 28               # cap vision tokens (dominant lever) = 200704
    min_pixels: int = 4 * 28 * 28
    max_seq_len: int = 1024

    # -- data / io --
    active_rows_path: str = "data/active_sft/active_rows.jsonl"
    out_dir: str = "models/sft_adapters"
    # Generator conditioning input (ADR 0020/0021): "instruction" (one-stage), "attribute_spec_text"
    # (two-stage, P6), or "instruction_and_spec" (hybrid: NL anchor + precise spec). NOT a locked knob
    # — it is the sanctioned input swap; all locked hyperparameters are unchanged when it flips.
    input_field: str = "instruction"
    # Render spec magnitudes as ordinal buckets (warmer=strong) on the generator INPUT instead of raw
    # floats (canonical serialize/parse unchanged). Applies whenever the input feeds a spec
    # (attribute_spec_text / instruction_and_spec). NOT a locked knob.
    spec_bucketize: bool = False

    # -- Phase 3 collapse fixes (methodology knobs; NOT in the locked bilevel search) --
    # Codebook-embedding-weighted soft-target loss: auxiliary CE on code positions whose target is a
    # softmax over codebook distances (perceptually-near codes share mass). 0.0 = OFF (identical to
    # baseline hard-label CE). tau is the (scale-free) softmax temperature.
    soft_label_weight: float = 0.0
    soft_label_tau: float = 1.0
    # Train-only spec augmentation: jitter magnitudes (sign-preserving) + shuffle axis order on the
    # INPUT (target codes unchanged) to smooth the learned function. NEVER applied at scoring.
    spec_augment: bool = False
    spec_jitter: float = 0.3

    # -- smoke overfit sizes (Stage 5 smoke tests = the FIRST run) --
    smoke_sizes: tuple[int, ...] = (50, 200)
    ckpt_every: int = 200
    keep_last: int = 3

    def __post_init__(self) -> None:
        if self.epochs != 2:
            raise ValueError("epochs is fixed at 2 (training_plan_colab.md: not a sanctioned lever)")
        eff = self.per_device_batch_size * self.gradient_accumulation_steps
        if eff != self.effective_batch_size:
            raise ValueError(
                f"per_device_batch_size*gradient_accumulation_steps={eff} "
                f"!= effective_batch_size={self.effective_batch_size}")
        if self.num_new_tokens != NUM_SPECIAL_TOKENS:
            raise ValueError(f"num_new_tokens must be {NUM_SPECIAL_TOKENS}")
        if self.projector_policy not in ("lora", "full", "frozen"):
            raise ValueError("projector_policy must be one of lora|full|frozen")
        if self.bnb_4bit_compute_dtype not in ("bfloat16", "float16"):
            raise ValueError("bnb_4bit_compute_dtype must be bfloat16 or float16")
        if self.input_field not in ("instruction", "attribute_spec_text", "instruction_and_spec"):
            raise ValueError(
                "input_field must be instruction, attribute_spec_text, or instruction_and_spec")
        if self.soft_label_weight < 0:
            raise ValueError("soft_label_weight must be >= 0 (0 disables the soft-target loss)")
        if self.soft_label_tau <= 0:
            raise ValueError("soft_label_tau must be > 0")

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_CONFIG = SFTConfig()


def load_config(path: str | Path | None = None) -> SFTConfig:
    """Load an :class:`SFTConfig`, applying YAML overrides for known fields only.

    Shared by the trainer, scorer, oracle gate, and vocab-resize entrypoints so their
    config parsing cannot drift apart (they must stay byte-aligned for train/score to
    agree). List values are coerced to tuples to match the frozen dataclass fields.
    """
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    overrides = yaml.safe_load(p.read_text(encoding="utf-8")) or {} if p.exists() else {}
    field_names = {f.name for f in fields(SFTConfig)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in overrides.items() if k in field_names}
    return SFTConfig(**kw)
