"""Shared eval-time model loader for the prompt->LUT VLM (4-bit base + trained LoRA adapter).

Extracted from :func:`sft.score_tokens.score` so the scorer, :mod:`eval.oracle_at_n`,
:mod:`eval.best_of_n`, and the distillation harvest all load the adapter IDENTICALLY (one place to
keep the QLoRA/dtype config correct). Heavy deps (torch/transformers/peft) are imported lazily, so
this module imports cleanly without the ``sft`` extra.
"""

from __future__ import annotations

from data_pipeline.errors import SFTError


def load_eval_model(cfg, resized_model: str, adapter: str):
    """Load the resized 4-bit base + a trained LoRA adapter for INFERENCE. Returns ``(model, processor)``.

    Mirrors the QLoRA + compute-dtype config carried on ``cfg`` (bf16 on A100, auto fp16 fallback on
    Turing/Volta e.g. the Colab T4). The returned model is in ``.eval()`` mode. Raises
    :class:`~data_pipeline.errors.SFTError` if the QLoRA stack is not installed.
    """
    try:
        from transformers import AutoProcessor, BitsAndBytesConfig
        from peft import PeftModel
    except Exception as exc:  # noqa: BLE001
        raise SFTError(f"QLoRA stack unavailable (install the `sft` extra): {exc}") from exc
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _ModelCls
    except Exception:  # noqa: BLE001
        from transformers import AutoModelForVision2Seq as _ModelCls  # type: ignore

    from sft.example import resolve_compute_dtype
    compute_dtype = resolve_compute_dtype(cfg)   # bf16 on A100; auto fp16 on T4/Volta (no hw bf16)
    processor = AutoProcessor.from_pretrained(resized_model, trust_remote_code=True,
                                              min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels)
    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit, bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant, bnb_4bit_compute_dtype=compute_dtype)
    base = _ModelCls.from_pretrained(resized_model, quantization_config=bnb, torch_dtype=compute_dtype,
                                     device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return model, processor
