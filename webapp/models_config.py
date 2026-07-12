"""Typed, config-driven model registry for the local web demo."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_path(path_str: str | os.PathLike[str]) -> Path:
    """Resolve a possibly-relative config path against the repo root, so launch CWD never matters."""
    p = Path(path_str)
    return p if p.is_absolute() else (_REPO_ROOT / p)


def _section(dc_cls, raw: dict[str, Any], name: str):
    """Build a config dataclass, raising a clear error on unknown keys instead of a raw TypeError."""
    if not isinstance(raw, dict):
        raise ValueError(f"config section '{name}' must be a JSON object")
    valid = {f.name for f in fields(dc_cls)}
    unknown = sorted(set(raw) - valid)
    if unknown:
        raise ValueError(f"unknown key(s) in config section '{name}': {unknown} (valid: {sorted(valid)})")
    return dc_cls(**raw)


@dataclass
class InterpreterConfig:
    model_path: str = "models/interpreter/interp_full"
    base_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    tuning_mode: str = "full"
    max_new_tokens: int = 64


@dataclass
class GeneratorConfig:
    stub: bool = True
    adapter_path: str | None = "models/sft_adapters/p6_twostage_d0f9c744_smokefull"
    base_model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    resized_base_path: str = "models/base_resized"
    input_mode: str = "attribute_spec_text"
    spec_bucketize: bool = False
    load_in_4bit: bool = True
    best_of_n_N: int = 4
    chunk: int = 4
    sampling: dict[str, float] = field(default_factory=lambda: {"temperature": 1.0, "top_p": 0.9})
    max_pixels: int = 200704
    min_pixels: int = 3136


@dataclass
class VQDecoderConfig:
    final_dir: str | None = None


@dataclass
class ServerConfig:
    runs_dir: str = "webapp/_runs"
    static_dir: str = "webapp/static"
    references_dir: str = "webapp/assets/references"
    max_upload_mb: int = 20
    max_image_edge: int = 2048
    request_timeout_s: int = 600


@dataclass
class WebappConfig:
    device: str = "mps"
    interpreter: InterpreterConfig = field(default_factory=InterpreterConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    vq_decoder: VQDecoderConfig = field(default_factory=VQDecoderConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    config_path: str | None = None

    @classmethod
    def load(cls, path: str | os.PathLike[str] = "configs/webapp.json") -> "WebappConfig":
        config_path = Path(path)
        raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{config_path}: top-level config must be a JSON object")
        known_top = {"device", "interpreter", "generator", "vq_decoder", "server"}
        unknown_top = sorted(set(raw) - known_top)
        if unknown_top:
            raise ValueError(f"{config_path}: unknown top-level key(s): {unknown_top} (valid: {sorted(known_top)})")
        cfg = cls(
            device=str(raw.get("device", "mps")),
            interpreter=_section(InterpreterConfig, raw.get("interpreter", {}), "interpreter"),
            generator=_section(GeneratorConfig, raw.get("generator", {}), "generator"),
            vq_decoder=_section(VQDecoderConfig, raw.get("vq_decoder", {}), "vq_decoder"),
            server=_section(ServerConfig, raw.get("server", {}), "server"),
            config_path=str(config_path),
        )
        env_stub = os.environ.get("WEBAPP_STUB")
        if env_stub is not None:
            cfg.generator.stub = env_stub.strip().lower() not in {"0", "false", "no", "off"}
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.device not in {"cuda", "mps", "cpu"}:
            raise ValueError("device must be one of cuda, mps, or cpu")
        if self.interpreter.tuning_mode not in {"full", "lora"}:
            raise ValueError("interpreter.tuning_mode must be full or lora")
        if self.generator.input_mode not in {"attribute_spec_text", "instruction", "instruction_and_spec"}:
            raise ValueError("generator.input_mode has an unsupported value")
        if self.generator.best_of_n_N < 1 or self.generator.chunk < 1:
            raise ValueError("generator best_of_n_N and chunk must be positive")
        if self.generator.max_pixels < self.generator.min_pixels:
            raise ValueError("generator max_pixels must be >= min_pixels")
        if self.server.max_upload_mb < 1 or self.server.max_image_edge < 64:
            raise ValueError("server upload/image limits are invalid")


def load_interpreter(cfg: WebappConfig):
    """Load the full-FT or LoRA interpreter once on the configured device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ic = cfg.interpreter
    dtype = torch.float32 if cfg.device == "cpu" else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(ic.model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if ic.tuning_mode == "lora":
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(ic.base_model_id, dtype=dtype)
        model = PeftModel.from_pretrained(base, ic.model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(ic.model_path, dtype=dtype)
    return model.to(cfg.device).eval(), tokenizer, cfg.device


def _match_embedding_rows(base, tokenizer) -> None:
    """Match the model table to the adapter tokenizer, including removal of Qwen padding rows."""
    tokenizer_size = len(tokenizer)
    embedding_size = int(base.get_input_embeddings().num_embeddings)
    if tokenizer_size != embedding_size:
        base.resize_token_embeddings(tokenizer_size)


def load_generator(cfg: WebappConfig):
    """Load the resized generator without ever touching bitsandbytes off CUDA."""
    g = cfg.generator
    if cfg.device == "cuda" and g.load_in_4bit:
        from sft.config import SFTConfig
        from sft.loader import load_eval_model

        sft_cfg = SFTConfig(max_pixels=g.max_pixels, min_pixels=g.min_pixels)
        return load_eval_model(sft_cfg, g.resized_base_path, g.adapter_path)

    import torch
    from transformers import AutoProcessor, AutoTokenizer

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    except ImportError:
        from transformers import AutoModelForVision2Seq as ModelClass

    dtype = torch.float32 if cfg.device == "cpu" else torch.float16
    processor = AutoProcessor.from_pretrained(
        g.resized_base_path,
        trust_remote_code=True,
        min_pixels=g.min_pixels,
        max_pixels=g.max_pixels,
    )
    # Adapters produced by this repo carry the authoritative resized tokenizer.  Install it on the
    # base processor when present.  This supports the disk-saving local fallback where the vanilla
    # base snapshot is resized in memory before PEFT restores the trained embedding/lm-head modules.
    if g.adapter_path and Path(g.adapter_path, "tokenizer.json").is_file():
        processor.tokenizer = AutoTokenizer.from_pretrained(g.adapter_path, trust_remote_code=True)
    base = ModelClass.from_pretrained(g.resized_base_path, dtype=dtype, trust_remote_code=True)
    if g.adapter_path:
        from peft import PeftModel

        # Qwen pads its vanilla embedding table to 151,936 rows.  The training resize is the
        # authoritative adapter vocabulary (151,924 here), so matching it may legitimately shrink
        # that padded table as well as grow an unpadded vanilla base.
        _match_embedding_rows(base, processor.tokenizer)
        model = PeftModel.from_pretrained(base, g.adapter_path).merge_and_unload()
    else:
        model = base
    return model.to(cfg.device).eval(), processor
