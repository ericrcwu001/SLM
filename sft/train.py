"""QLoRA SFT for the prompt-to-LUT VLM (training_plan_colab.md "Stage 5"; master-plan Stage 14).

Resumable-ish QLoRA loop over ``data/active_sft/active_rows.jsonl``: 4-bit NF4 base +
LoRA adapters on the LM projections, row-embeddings trained via ``modules_to_save`` for the
259 new tokens (smoke default — row-selective masking is the documented follow-up), assistant-only
loss masking, cosine schedule with warmup. The FIRST run is the 50/200-row overfit smoke
(``--smoke-size``), single-seed and labeled exploratory.

Heavy deps (transformers/peft/bitsandbytes/accelerate/qwen-vl-utils) are the ``sft`` extra and are
imported lazily. Example construction + row loading live in :mod:`sft.example` (shared with the
token-accuracy scorer). Image paths resolve against ``$SLM_ARTIFACT_ROOT`` (the staged corpus root
on Colab), falling back to cwd.

Held-out rows (:mod:`sft.holdout`) are EXCLUDED from training so :mod:`sft.score_tokens` can score
generalization on them. The random seed is taken from the ``BILEVEL_SEED`` env var when set (so the
bilevel engine's ``repeats`` are genuinely independent draws), else ``cfg.seed``; the training row
order is shuffled with that seed.

Machine-readable output: a single ``{"sft_summary": {...}}`` JSON line is printed before the final
``[sft][OK]``; a run that trains ZERO rows/steps prints ``[sft][ABORT]`` and returns non-zero
(never a misleading ``[sft][OK]`` on an untrained adapter — the silent-success trap).

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
import random
from pathlib import Path

import yaml

from data_pipeline.errors import RequiresTokenizer, SFTError
from sft.config import DEFAULT_CONFIG, SFTConfig
from sft.example import artifact_root, build_supervised_example, load_rows, supported_rows
from sft.manifest import build_adapter_manifest, write_manifest

_DEFAULT_CFG_PATH = Path("configs/sft_default.yaml")


def _load_config(path: str | None) -> SFTConfig:
    p = Path(path) if path else _DEFAULT_CFG_PATH
    overrides = yaml.safe_load(p.read_text(encoding="utf-8")) or {} if p.exists() else {}
    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in overrides.items() if k in fields}
    return SFTConfig(**kw)


def _effective_seed(cfg: SFTConfig) -> int:
    """Seed from BILEVEL_SEED (bilevel repeats) when set, else cfg.seed. Distinct seeds -> distinct
    new-token init, dropout masks, and data order, so the engine's variance gate is not defeated."""
    env = os.environ.get("BILEVEL_SEED")
    if env is not None:
        try:
            return int(env)
        except ValueError:
            pass
    return cfg.seed


def _load_rows(active_rows_path: str, smoke_size: int | None):
    """Load supported (materialized, holdout-EXCLUDED) + unsupported rows; take a balanced smoke subset."""
    rows = load_rows(active_rows_path)
    sup = supported_rows(rows, holdout=False)  # exclude the scored holdout slice from training
    # Refusal rows (route=refuse -> target <unsupported>). ``clarify`` rows (ADR 0023) are an
    # INTERPRETER route, never a generator target, so they are excluded from the generator's pool.
    unsup = [r for r in rows if not r.get("is_supported") and r.get("image_path")
             and r.get("instruction") and r.get("route") != "clarify"]
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

    seed = _effective_seed(cfg)
    torch.manual_seed(seed)
    from sft.example import resolve_compute_dtype
    compute_dtype = resolve_compute_dtype(cfg)   # bf16 on A100; auto fp16 on T4/Volta (no hw bf16)

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
    random.Random(seed).shuffle(rows)  # seeded order so BILEVEL_SEED repeats differ
    print(f"[sft] RUN_BEGIN run_id={run_id} rows={len(rows)} smoke_size={smoke_size} "
          f"seed={seed} artifact_root={artifact_root()}")
    print(f"[sft] training rows={len(rows)} (smoke_size={smoke_size})")

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
    loss_sum = 0.0        # cumulative mean-loss numerator over every micro-step (for the summary line)
    n_seen = 0            # micro-steps that actually produced a loss (rows trained)
    skipped = 0           # rows dropped by _example (bad image path / truncation)
    for epoch in range(cfg.epochs):
        for row in rows:
            try:
                batch = build_supervised_example(processor, row, cfg, device=model.device)
            except Exception as exc:  # noqa: BLE001 — skip a bad row rather than kill the run
                skipped += 1
                print(f"[sft][skip] {row.get('id')}: {type(exc).__name__}: {exc}")
                continue
            loss = model(**batch).loss / cfg.gradient_accumulation_steps
            loss.backward()
            running += float(loss) * cfg.gradient_accumulation_steps
            loss_sum += float(loss) * cfg.gradient_accumulation_steps
            micro += 1
            n_seen += 1
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

    mean_loss = (loss_sum / n_seen) if n_seen else None
    # Fail loud on a no-op: 0 optimizer steps / 0 rows trained means every image was skipped
    # (wrong SLM_ARTIFACT_ROOT / case trap) or the subset was empty. Never write an [sft][OK]
    # adapter that was not actually trained (the silent-success trap).
    if step == 0 or n_seen == 0:
        print(json.dumps({"sft_summary": {"run_id": run_id, "steps": step, "rows_trained": n_seen,
                                          "skipped": skipped, "mean_loss": mean_loss}}))
        print(f"[sft][ABORT] 0 rows trained (steps={step} rows_trained={n_seen} skipped={skipped}) — "
              f"check SLM_ARTIFACT_ROOT ({artifact_root()}) / image paths")
        return 1

    ckpt = Path(out_dir) / f"{run_id}_smoke{smoke_size or 'full'}"
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    # Adapter manifest (bind frozen-tokenizer + resized-vocab identity).
    from tokenizer.manifest import hash_state_dict
    trainable_sd = {n: p for n, p in model.named_parameters() if p.requires_grad}
    adapter_sha = hash_state_dict({n: p.detach().float().cpu() for n, p in trainable_sd.items()})
    tok_manifest = _read_json(artifact_root() / "tokenizer" / "final" / "manifest.json")
    vr_manifest = _read_json(Path(resized_model) / "vocab_resize_manifest.json")
    manifest = build_adapter_manifest(
        run_id=run_id, adapter_step=step, adapter_sha256=adapter_sha, cfg=cfg.to_dict(),
        tokenizer_manifest=tok_manifest, vocab_resize_manifest=vr_manifest,
        train_report={"rows": len(rows), "steps": step, "rows_trained": n_seen, "skipped": skipped,
                      "mean_loss": mean_loss, "smoke_size": smoke_size, "epochs": cfg.epochs,
                      "seed": seed})
    write_manifest(ckpt / "adapter_manifest.json", manifest)

    # Single machine-readable summary line (parsed by sft.bilevel_bridge to guard steps>0 and read
    # the training loss); printed on its own line before the human-facing [sft][OK].
    print(json.dumps({"sft_summary": {"run_id": run_id, "steps": step, "rows_trained": n_seen,
                                      "skipped": skipped, "mean_loss": mean_loss,
                                      "adapter": str(ckpt), "adapter_sha256": adapter_sha}}))
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
