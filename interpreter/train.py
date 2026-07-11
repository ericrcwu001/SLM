"""Train the Stage-1 interpreter (text-only CausalLM; full fine-tuning by default).

Reads the unified corpus (``scripts.build_interpreter_corpus``), EXCLUDES holdout units from
training (the leakage contract), upsamples the sparse non-grade routes, and runs a plain AdamW +
cosine loop (LR schedule / accumulation lifted from ``sft.train``). No image, no VQ tokens, no vocab
resize, no bitsandbytes for the default full-FT path.

    python -m interpreter.train --config configs/interpreter_default.yaml --run-id interp_r1
    python -m interpreter.train --config configs/candidate_interpreter.json --smoke-size 200 --max-steps 20
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Optional

import torch

from interpreter.config import InterpreterConfig, load_config
from interpreter.corpus import load_interpreter_rows, split_train_holdout
from interpreter.example import build_supervised_example, resolve_eos_and_pad
from eval.refuse_taxonomy import ROUTE_GRADE


def _resolve_dtype() -> torch.dtype:
    # For TRAINING use bf16 (A100) or fp32 (T4/CPU) — never raw fp16: full fine-tuning in pure
    # fp16 (no GradScaler/autocast) diverges to NaN. fp32 full-FT of a 0.5B model fits a 16GB T4.
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def _lr(step: int, total: int, base: float, warmup: int, scheduler: str) -> float:
    if warmup and step < warmup:
        return base * step / max(1, warmup)
    if scheduler == "constant":
        return base
    prog = (step - warmup) / max(1, total - warmup)
    if scheduler == "linear":
        return base * max(0.0, 1.0 - prog)
    return base * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))  # cosine


def _upsample(train_rows: list[dict], factor: int) -> list[dict]:
    """Repeat non-grade (refuse/clarify) rows ``factor``× so the ~grade-dominated mix routes at all."""
    if factor <= 1:
        return list(train_rows)
    out = list(train_rows)
    extra = [r for r in train_rows if r.get("route") != ROUTE_GRADE]
    for _ in range(factor - 1):
        out.extend(extra)
    return out


def _collate(batch: list[dict], pad_id: int) -> dict:
    width = max(len(ex["input_ids"]) for ex in batch)
    input_ids, labels, attn = [], [], []
    for ex in batch:
        ids, lab = ex["input_ids"], ex["labels"]
        pad = width - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        labels.append(lab + [-100] * pad)
        attn.append([1] * len(ids) + [0] * pad)
    return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}


def train(cfg: InterpreterConfig, *, run_id: str, smoke_size: int = 0,
          max_steps: int = 0, gradient_checkpointing: Optional[bool] = None) -> int:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = load_interpreter_rows(cfg.corpus_path)
    train_rows, _holdout = split_train_holdout(rows, cfg.holdout_frac)  # holdout EXCLUDED from train
    if smoke_size:
        train_rows = train_rows[:smoke_size]
    train_rows = _upsample(train_rows, cfg.upsample_nongrade)
    if not train_rows:
        print(f"[interp][ABORT] 0 training rows from {cfg.corpus_path} "
              f"(built the corpus? holdout too large?)")
        print(json.dumps({"interpreter_summary": {"rows_trained": 0, "status": "abort"}}))
        return 1

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model_id)
    resolve_eos_and_pad(tokenizer)
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model_id, torch_dtype=_resolve_dtype())
    gc_on = cfg.gradient_checkpointing if gradient_checkpointing is None else gradient_checkpointing
    if gc_on:
        model.gradient_checkpointing_enable()
    if cfg.tuning_mode == "lora":
        from peft import LoraConfig, get_peft_model  # lazy: only the LoRA path needs peft
        model = get_peft_model(model, LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules), task_type="CAUSAL_LM"))
    model.to(device)
    model.train()

    examples = [build_supervised_example(tokenizer, r, cfg.max_seq_len) for r in train_rows]
    pad_id = tokenizer.pad_token_id
    bs = cfg.per_device_batch_size
    accum = cfg.gradient_accumulation_steps
    steps_per_epoch = math.ceil(len(examples) / bs)
    total_optim = max_steps or (math.ceil(steps_per_epoch / accum) * cfg.epochs)
    warmup = int(round(cfg.warmup_ratio * total_optim))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    optim_step, micro = 0, 0
    losses: list[float] = []
    for epoch in range(cfg.epochs):
        random.shuffle(examples)
        for i in range(0, len(examples), bs):
            batch = _collate(examples[i:i + bs], pad_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / accum
            loss.backward()
            losses.append(float(loss) * accum)
            micro += 1
            if micro % accum == 0:
                for g in opt.param_groups:
                    g["lr"] = _lr(optim_step, total_optim, cfg.learning_rate, warmup, cfg.scheduler)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()
                opt.zero_grad(set_to_none=True)
                optim_step += 1
                if optim_step % 20 == 0:
                    print(f"[interp] epoch {epoch} optim_step {optim_step}/{total_optim} "
                          f"loss {sum(losses[-accum:]) / accum:.4f}")
                if max_steps and optim_step >= max_steps:
                    break
        if max_steps and optim_step >= max_steps:
            break

    ckpt = Path(cfg.out_dir) / f"{run_id}_smoke{smoke_size or 'full'}"
    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tokenizer.save_pretrained(ckpt)
    summary = {"rows_trained": len(train_rows), "optim_steps": optim_step,
               "mean_loss": (sum(losses) / len(losses)) if losses else None,
               "base_model_id": cfg.base_model_id, "tuning_mode": cfg.tuning_mode,
               "adapter": str(ckpt), "status": "ok"}
    print(f"[interp][OK] {summary}")
    print(json.dumps({"interpreter_summary": summary}))
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train the Stage-1 interpreter (text-only).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--run-id", default="interp")
    ap.add_argument("--smoke-size", type=int, default=0, help="first N train rows (0 = full)")
    ap.add_argument("--max-steps", type=int, default=0, help="cap optimizer steps (smoke)")
    ap.add_argument("--gradient-checkpointing", dest="gc", action="store_true", default=None)
    ap.add_argument("--no-gradient-checkpointing", dest="gc", action="store_false")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    return train(cfg, run_id=args.run_id, smoke_size=args.smoke_size,
                 max_steps=args.max_steps, gradient_checkpointing=args.gc)


if __name__ == "__main__":
    raise SystemExit(main())
