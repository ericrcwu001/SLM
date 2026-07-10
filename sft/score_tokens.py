"""Decoder-free held-out token-accuracy metric for an SFT adapter (the bilevel objective).

Loads the resized base + a trained LoRA adapter, and for each HELD-OUT supported row
(:mod:`sft.holdout`) runs ONE teacher-forced forward pass, argmaxes the logits over the assistant
span, and measures how often the model predicts the correct ``<lut_NNN>`` code token. This is:

  * FAITHFUL-ish — the 64 code tokens deterministically decode to the target residual LUT under the
    frozen VQ encoder (predicting the exact codes ≈ predicting the target LUT), so it is the best
    quality proxy obtainable WITHOUT enabling the frozen decoder (eval/lut_decoder.py stays disabled);
  * DETERMINISTIC — argmax, no sampling, so it is not polluted by generation noise (the bilevel
    variance gate stays meaningful);
  * CHEAP — one forward pass per held-out row (no autoregressive generation loop).

Prints exactly one ``METRIC=<accuracy>`` sentinel line (direction=MAX) that the bilevel objective
parser takes verbatim, plus a ``{"score_summary": {...}}`` JSON line. Exits non-zero (no METRIC) if
no held-out row could be scored — so a mis-rooted corpus can never yield a bogus metric.

Heavy deps (torch/transformers/peft) are imported lazily. Runs on the Colab GPU stack.

Usage (on Colab, after training an adapter):
    python -m sft.score_tokens --resized-model models/base_resized \
        --adapter models/sft_adapters/bl_smoke200 --limit 48
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import yaml

from data_pipeline.errors import SFTError
from eval.vocab import code_token
from sft.config import SFTConfig
from sft.example import build_supervised_example, load_rows, supported_rows

_DEFAULT_CFG_PATH = Path("configs/sft_default.yaml")


def _load_config(path: str | None) -> SFTConfig:
    p = Path(path) if path else _DEFAULT_CFG_PATH
    overrides = yaml.safe_load(p.read_text(encoding="utf-8")) or {} if p.exists() else {}
    fields = {f.name for f in dataclasses.fields(SFTConfig)}
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in overrides.items() if k in fields}
    return SFTConfig(**kw)


def score(cfg: SFTConfig, resized_model: str, adapter: str, limit: int) -> dict:
    try:
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig
        from peft import PeftModel
    except Exception as exc:  # noqa: BLE001
        raise SFTError(f"QLoRA stack unavailable (install the `sft` extra): {exc}") from exc
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _ModelCls
    except Exception:  # noqa: BLE001
        from transformers import AutoModelForVision2Seq as _ModelCls  # type: ignore

    compute_dtype = torch.bfloat16 if cfg.bnb_4bit_compute_dtype == "bfloat16" else torch.float16
    processor = AutoProcessor.from_pretrained(resized_model, trust_remote_code=True,
                                              min_pixels=cfg.min_pixels, max_pixels=cfg.max_pixels)
    tok = processor.tokenizer

    bnb = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit, bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant, bnb_4bit_compute_dtype=compute_dtype)
    base = _ModelCls.from_pretrained(resized_model, quantization_config=bnb, torch_dtype=compute_dtype,
                                     device_map="auto", trust_remote_code=True)
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()

    # The 256 code-token ids in the resized vocab; accuracy is measured ONLY over these positions
    # (ignore <lut_bos>/<lut_eos>/chat-template tokens the model gets trivially).
    code_ids = torch.tensor([tok.convert_tokens_to_ids(code_token(k)) for k in range(256)],
                            device=model.device)

    rows = supported_rows(load_rows(cfg.active_rows_path), holdout=True)
    if limit:
        rows = rows[:limit]
    if not rows:
        raise SFTError("no held-out supported rows to score (empty holdout slice)")

    correct = total = scored_rows = skipped = exact = 0
    for row in rows:
        try:
            batch = build_supervised_example(processor, row, cfg, device=model.device)
        except Exception as exc:  # noqa: BLE001 — skip a bad/missing-image row
            skipped += 1
            print(f"[score][skip] {row.get('id')}: {type(exc).__name__}: {exc}")
            continue
        labels = batch.pop("labels")
        with torch.no_grad():
            logits = model(**batch).logits
        pred = logits[:, :-1, :].argmax(dim=-1)          # predicts token t+1
        gold = batch["input_ids"][:, 1:]
        mask = labels[:, 1:] != -100                      # assistant span only
        is_code = torch.isin(gold, code_ids)
        sel = mask & is_code
        n_sel = int(sel.sum())
        if n_sel == 0:
            skipped += 1
            continue
        n_hit = int(((pred == gold) & sel).sum())
        correct += n_hit
        total += n_sel
        scored_rows += 1
        if n_hit == n_sel:
            exact += 1

    if total == 0:
        raise SFTError("scored 0 code positions (all held-out rows skipped — check SLM_ARTIFACT_ROOT / images)")

    accuracy = correct / total
    exact_match = exact / scored_rows if scored_rows else 0.0
    return {"metric": accuracy, "token_accuracy": accuracy, "exact_match_rate": exact_match,
            "code_positions": total, "correct": correct, "scored_rows": scored_rows, "skipped": skipped}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(_DEFAULT_CFG_PATH))
    ap.add_argument("--resized-model", default="models/base_resized")
    ap.add_argument("--adapter", required=True, help="trained adapter dir (models/sft_adapters/<run>)")
    ap.add_argument("--limit", type=int, default=48, help="max held-out rows to score (cost lever)")
    args = ap.parse_args(argv)
    cfg = _load_config(args.config)
    try:
        rep = score(cfg, args.resized_model, args.adapter, args.limit)
    except SFTError as exc:
        print(json.dumps({"score_summary": {"error": str(exc)}}))
        print(f"[score][ABORT] {exc}")
        return 1
    print(json.dumps({"score_summary": rep}))
    print(f"METRIC={rep['metric']:.6f}")   # bilevel sentinel (direction=max), wins over all parsing
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
