"""SFT adapter / vocab-resize manifests (mirror tokenizer/manifest.py + eval.schemas identity).

Pure dict assembly (no torch import) — callers pass already-computed hashes (adapter_sha256 via
tokenizer.manifest.hash_state_dict over the trainable state). Binds the adapter to the exact
frozen tokenizer + resized-vocab identity it commits to, so a checkpoint is reproducible and
self-check-able (model_architecture.md "Version Manifest And Startup Assertions").
"""

from __future__ import annotations

import json
from pathlib import Path


def build_vocab_resize_manifest(*, base_model_id: str, base_vocab_size: int,
                                vocab_size_after_resize: int, added_special_token_ids: dict,
                                tied_embedding_status: str, tokenizer_manifest: dict,
                                preflight: dict) -> dict:
    """Identity + preflight record written next to the resized base (Stage 3)."""
    return {
        "stage": "vocab_resize_preflight",
        "base_model_id": base_model_id,
        "base_vocab_size": base_vocab_size,
        "vocab_size_after_resize": vocab_size_after_resize,
        "num_added_tokens": len(added_special_token_ids),
        "added_special_token_ids": added_special_token_ids,
        "tied_embedding_status": tied_embedding_status,
        "token_suffix_to_codebook_index": "identity",
        # bind the frozen tokenizer this vocab is for
        "tokenizer_version": tokenizer_manifest.get("tokenizer_version"),
        "vq_codebook_sha256": tokenizer_manifest.get("vq_codebook_sha256"),
        "vq_decoder_sha256": tokenizer_manifest.get("vq_decoder_sha256"),
        "preflight": preflight,
    }


def build_adapter_manifest(*, run_id: str, adapter_step: int, adapter_sha256: str,
                           cfg: dict, tokenizer_manifest: dict, vocab_resize_manifest: dict,
                           train_report: dict) -> dict:
    """Adapter identity for a checkpoint under models/sft_adapters/."""
    return {
        "stage": "sft_qlora",
        "run_id": run_id,
        "adapter_step": adapter_step,
        "adapter_sha256": adapter_sha256,
        "base_model_id": cfg.get("base_model_id"),
        "lora": {k: cfg.get(k) for k in ("lora_r", "lora_alpha", "lora_dropout",
                                         "lora_target_modules")},
        "quantization": {k: cfg.get(k) for k in ("load_in_4bit", "bnb_4bit_quant_type",
                                                 "bnb_4bit_use_double_quant", "bnb_4bit_compute_dtype")},
        "optim": {k: cfg.get(k) for k in ("epochs", "per_device_batch_size",
                                          "gradient_accumulation_steps", "effective_batch_size",
                                          "learning_rate_lora", "warmup_ratio", "scheduler",
                                          "max_grad_norm", "gradient_checkpointing", "seed")},
        "levers": {k: cfg.get(k) for k in ("max_pixels", "min_pixels", "max_seq_len")},
        # identity this adapter is only valid under
        "tokenizer_version": tokenizer_manifest.get("tokenizer_version"),
        "vq_codebook_sha256": tokenizer_manifest.get("vq_codebook_sha256"),
        "vocab_size_after_resize": vocab_resize_manifest.get("vocab_size_after_resize"),
        "tied_embedding_status": vocab_resize_manifest.get("tied_embedding_status"),
        "train_report": train_report,
    }


def write_manifest(path: str | Path, manifest: dict) -> None:
    Path(path).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
