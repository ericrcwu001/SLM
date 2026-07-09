"""QLoRA SFT for the prompt-to-LUT VLM (training_plan_colab.md "Stage 5"; master-plan Stage 14).

Resumable-ish QLoRA loop over ``data/active_sft/active_rows.jsonl``: 4-bit NF4 base +
LoRA adapters on the LM projections, row-embeddings trained via ``modules_to_save`` for the
259 new tokens (smoke default — row-selective masking is the documented follow-up), assistant-only
loss masking, cosine schedule with warmup. The FIRST run is the 50/200-row overfit smoke
(``--smoke-size``), single-seed and labeled exploratory.

Heavy deps (transformers/peft/bitsandbytes/accelerate/qwen-vl-utils) are the ``sft`` extra and are
imported lazily. Image paths in the dataset are RELATIVE and resolve against ``$SLM_ARTIFACT_ROOT``
(the staged corpus root on Colab), falling back to cwd.

⚠ First-draft trainer — validate on the Colab A100 (transformers/Qwen-VL API + memory) before the
full run. Runs nothing on import.

Usage (from repo root, on Colab):
    python -m sft.train --resized-model models/base_resized --smoke-size 50 --max-steps 300
    python -m sft.train --resized-model models/base_resized --smoke-size 200
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
from pathlib import Path

import yaml

from data_pipeline.errors import RequiresTokenizer, SFTError
from sft.config import DEFAULT_CONFIG, SFTConfig
from sft.manifest import build_adapter_manifest, write_manifest

_DEFAULT_CFG_PATH = Path("configs/sft_default.yaml")


def _load_config(path: str | None) -> SFTConfig:
    p = Path(path) if path else _DEFAULT_CFG_PATH
    overrides = yaml.safe_load(p.read_text(encoding="utf-8")) or {} if p.exists() else {}
    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in overrides.items() if k in fields}
    return SFTConfig(**kw)


def _artifact_root() -> Path:
    return Path(os.environ.get("SLM_ARTIFACT_ROOT", os.getcwd()))


def _resolve_image(path: str) -> str:
    return path if os.path.isabs(path) else str(_artifact_root() / path)


def _load_rows(active_rows_path: str, smoke_size: int | None):
    """Load supported (materialized) + unsupported rows; take a balanced smoke subset."""
    rows = [json.loads(l) for l in Path(active_rows_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    sup = [r for r in rows if r.get("is_supported") and isinstance(r.get("target_tokens"), list)
           and len(r["target_tokens"]) == 64 and r.get("image_path") and r.get("instruction")]
    unsup = [r for r in rows if not r.get("is_supported") and r.get("image_path") and r.get("instruction")]
    if not sup:
        raise SFTError("no materialized supported rows (run scripts.materialize_target_tokens first)")
    if smoke_size:
        # ~15% unsupported in the smoke subset (mirrors the corpus refusal fraction)
        n_unsup = max(1, int(round(smoke_size * 0.15))) if unsup else 0
        subset = sup[: smoke_size - n_unsup] + unsup[:n_unsup]
        return subset
    return sup + unsup


def train(cfg: SFTConfig, resized_model: str, smoke_size: int | None, max_steps: int | None,
          out_dir: str, run_id: str) -> int:
    try:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except Exception as exc:  # noqa: BLE001
        raise SFTError(f"QLoRA stack unavailable (install the `sft` extra): {exc}") from exc
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _ModelCls
    except Exception:  # noqa: BLE001
        from transformers import AutoModelForVision2Seq as _ModelCls  # type: ignore

    torch.manual_seed(cfg.seed)
    compute_dtype = torch.bfloat16 if cfg.bnb_4bit_compute_dtype == "bfloat16" else torch.float16

    processor = AutoProcessor.from_pretrained(resized_model, trust_remote_code=True,
                                              min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels)
    tok = processor.tokenizer

    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit, bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant, bnb_4bit_compute_dtype=compute_dtype)
    model = _ModelCls.from_pretrained(resized_model, quantization_config=bnb, torch_dtype=compute_dtype,
                                      device_map="auto", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=cfg.gradient_checkpointing)

    # LoRA on the LM projections; new-token rows trained via modules_to_save (smoke default).
    modules_to_save = ["embed_tokens", "lm_head"] if cfg.train_new_token_rows else None
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
                      target_modules=list(cfg.lora_target_modules), bias="none",
                      task_type="CAUSAL_LM", modules_to_save=modules_to_save)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    rows = _load_rows(cfg.active_rows_path, smoke_size)
    print(f"[sft] training rows={len(rows)} (smoke_size={smoke_size})")

    def _example(row):
        """Build (inputs, labels) with the assistant target masked-in, prompt masked-out."""
        from qwen_vl_utils import process_vision_info
        user = {"role": "user", "content": [
            {"type": "image", "image": _resolve_image(row["image_path"])},
            {"type": "text", "text": row["instruction"]}]}
        target = row["assistant_target"] if row.get("is_supported") else "<unsupported>"
        assistant = {"role": "assistant", "content": [{"type": "text", "text": target}]}
        prompt_text = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
        full_text = processor.apply_chat_template([user, assistant], tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = process_vision_info([user])
        full = processor(text=[full_text], images=image_inputs, videos=video_inputs,
                         padding=True, return_tensors="pt", max_length=cfg.max_seq_len, truncation=True)
        prompt = processor(text=[prompt_text], images=image_inputs, videos=video_inputs,
                           return_tensors="pt")
        labels = full["input_ids"].clone()
        n_prompt = prompt["input_ids"].shape[1]
        labels[:, :n_prompt] = -100                      # assistant-only loss
        full["labels"] = labels
        return {k: v.to(model.device) for k, v in full.items()}

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=cfg.learning_rate_lora, weight_decay=cfg.weight_decay)
    steps_per_epoch = math.ceil(len(rows) / cfg.effective_batch_size)
    total_steps = max_steps or steps_per_epoch * cfg.epochs
    warmup = max(1, int(cfg.warmup_ratio * total_steps))

    def _lr(step):
        if step < warmup:
            return cfg.learning_rate_lora * step / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * cfg.learning_rate_lora * (1 + math.cos(math.pi * prog))

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.train()
    step = micro = 0
    running = 0.0
    for epoch in range(cfg.epochs):
        for row in rows:
            try:
                batch = _example(row)
            except Exception as exc:  # noqa: BLE001 — skip a bad row rather than kill the run
                print(f"[sft][skip] {row.get('id')}: {type(exc).__name__}: {exc}")
                continue
            loss = model(**batch).loss / cfg.gradient_accumulation_steps
            loss.backward()
            running += float(loss) * cfg.gradient_accumulation_steps
            micro += 1
            if micro % cfg.gradient_accumulation_steps == 0:
                for g in opt.param_groups:
                    g["lr"] = _lr(step)
                torch.nn.utils.clip_grad_norm_(trainable, cfg.max_grad_norm)
                opt.step(); opt.zero_grad()
                step += 1
                if step % 10 == 0:
                    print(f"[sft] epoch{epoch} step{step}/{total_steps} lr={_lr(step):.2e} "
                          f"loss={running / (10 * cfg.gradient_accumulation_steps):.4f}")
                    running = 0.0
                if max_steps and step >= max_steps:
                    break
        if max_steps and step >= max_steps:
            break

    ckpt = Path(out_dir) / f"{run_id}_smoke{smoke_size or 'full'}"
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    # Adapter manifest (bind frozen-tokenizer + resized-vocab identity).
    from tokenizer.manifest import hash_state_dict
    trainable_sd = {n: p for n, p in model.named_parameters() if p.requires_grad}
    adapter_sha = hash_state_dict({n: p.detach().float().cpu() for n, p in trainable_sd.items()})
    tok_manifest = _read_json(_artifact_root() / "tokenizer" / "final" / "manifest.json")
    vr_manifest = _read_json(Path(resized_model) / "vocab_resize_manifest.json")
    manifest = build_adapter_manifest(
        run_id=run_id, adapter_step=step, adapter_sha256=adapter_sha, cfg=cfg.to_dict(),
        tokenizer_manifest=tok_manifest, vocab_resize_manifest=vr_manifest,
        train_report={"rows": len(rows), "steps": step, "smoke_size": smoke_size, "epochs": cfg.epochs})
    write_manifest(ckpt / "adapter_manifest.json", manifest)
    print(f"[sft][OK] adapter -> {ckpt}  steps={step} adapter_sha256={adapter_sha[:16]}…")
    return 0


def _read_json(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(_DEFAULT_CFG_PATH))
    ap.add_argument("--resized-model", default="models/base_resized",
                    help="output dir of sft.vocab_resize (base with the 259 tokens added)")
    ap.add_argument("--smoke-size", type=int, default=50, help="overfit subset size (50/200); 0=full")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--out", default="models/sft_adapters")
    ap.add_argument("--run-id", default="sft_run", help="output subdir stem (no wall-clock in-script)")
    args = ap.parse_args(argv)
    cfg = _load_config(args.config)
    smoke = args.smoke_size or None
    try:
        return train(cfg, args.resized_model, smoke, args.max_steps, args.out, args.run_id)
    except (SFTError, RequiresTokenizer) as exc:
        print(f"[sft][ABORT] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
