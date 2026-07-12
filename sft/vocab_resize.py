"""Vocabulary resize + embedding preflight (training_plan_colab.md "Stage 3";
model_architecture.md "Vocabulary Resize And Embedding Preflight").

Adds the 259 special tokens (eval.vocab: <lut_bos>/<lut_eos>/<unsupported> + <lut_000..255>) to the
base Qwen2.5-VL tokenizer, resizes the model embeddings/head, mean-initializes the new rows, and runs
the preflight assertion suite. Writes the resized base + the FULL processor (tokenizer +
image-processor config + chat template) to ``--out`` and a ``vocab_resize_manifest.json`` (identity +
preflight report). The processor is required because ``sft/train.py`` loads the dir with
``AutoProcessor.from_pretrained`` — without ``preprocessor_config.json`` that call raises OSError.

Heavy deps (torch/transformers) are imported lazily; a missing runtime raises the honest guard so no
result is fabricated. Runs on the Colab GPU stack (or CPU for a small preflight-only check).

Usage (from repo root):
    python -m sft.vocab_resize --config configs/sft_default.yaml --out models/base_resized
    python -m sft.vocab_resize --preflight-only            # asserts only, writes nothing
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from data_pipeline.errors import SFTError
from eval.vocab import DEFAULT_VOCAB, NUM_SPECIAL_TOKENS, code_token
from sft.config import DEFAULT_CONFIG, SFTConfig, load_config as _load_config
from sft.manifest import build_vocab_resize_manifest, write_manifest

_DEFAULT_CFG_PATH = Path("configs/sft_default.yaml")


def preflight_checks(*, base_len: int, n_tok: int, n_in: int, n_out: int,
                     code_ids: list[int], new_ids: list[int], tied: bool,
                     num_added_expected: int = NUM_SPECIAL_TOKENS) -> dict:
    """Pure vocab-resize preflight gate (model_architecture.md "Vocabulary Resize" list).

    Extracted so the gate logic is unit-testable without loading a multi-GB model.
    """
    report = {
        "base_vocab_size": base_len,
        "vocab_size_after_resize": n_tok,
        "num_added": len(new_ids),
        "tied_embedding_status": "tied" if tied else "untied",
        "len_tok_eq_embed_eq_head": (n_tok == n_in == n_out),
        "code_tokens_contiguous": (len(code_ids) == 256
                                   and code_ids == list(range(code_ids[0], code_ids[0] + 256))),
        "special_ids_unique": (len(set(new_ids)) == num_added_expected),
        "count_ok": (n_tok == base_len + num_added_expected),
    }
    report["all_pass"] = bool(report["len_tok_eq_embed_eq_head"] and report["code_tokens_contiguous"]
                              and report["special_ids_unique"] and report["count_ok"])
    return report


def _write_artifacts(out_dir: str, model, processor, tok, manifest: dict) -> Path:
    """Persist the resized model + FULL processor + tokenizer + manifest to ``out_dir``.

    Order matters: the processor is written first (``preprocessor_config.json`` + chat template),
    then the resized tokenizer is written last so its 259 added tokens overwrite the processor's
    base-tokenizer copy and are authoritative on disk. ``sft/train.py`` loads this dir with
    ``AutoProcessor.from_pretrained`` and reads ``preprocessor_config.json``; omitting the processor
    save leaves the artifact unloadable. Extracted so the save contract is unit-testable without a
    multi-GB model (see tests/test_vocab_resize.py).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    processor.save_pretrained(out)   # writes preprocessor_config.json + chat template (the missing files)
    tok.save_pretrained(out)         # authoritative resized tokenizer (overwrites the processor's copy)
    # Defensive: some transformers versions don't emit preprocessor_config.json from
    # processor.save_pretrained; write it explicitly from the image processor so train.py's
    # AutoProcessor.from_pretrained(out) can never fail on a missing image-processor config.
    if not (out / "preprocessor_config.json").is_file() and getattr(processor, "image_processor", None) is not None:
        processor.image_processor.save_pretrained(out)
    write_manifest(out / "vocab_resize_manifest.json", manifest)
    return out


def resize_and_preflight(cfg: SFTConfig, out_dir: str | None, preflight_only: bool = False) -> dict:
    """Resize the base model's vocab and run preflight. Returns the preflight report dict."""
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # noqa: BLE001
        raise SFTError(f"transformers/torch unavailable (install the `sft`/`ml` extra): {exc}") from exc

    try:  # Qwen2.5-VL has a dedicated class; fall back to the generic vision2seq loader.
        from transformers import Qwen2_5_VLForConditionalGeneration as _ModelCls
    except Exception:  # noqa: BLE001
        from transformers import AutoModelForVision2Seq as _ModelCls  # type: ignore

    import torch

    tok = AutoTokenizer.from_pretrained(cfg.base_model_id, trust_remote_code=True)
    base_len = len(tok)

    new_tokens = DEFAULT_VOCAB.all_tokens  # 259 in canonical order
    added = tok.add_special_tokens({"additional_special_tokens": new_tokens})
    if added != NUM_SPECIAL_TOKENS:
        # some may already exist; assert the full set is now present and contiguous at the tail
        missing = [t for t in new_tokens if t not in tok.get_vocab()]
        if missing:
            raise SFTError(f"expected {NUM_SPECIAL_TOKENS} new tokens; missing {len(missing)}")

    # low_cpu_mem_usage streams the shards in (lower peak RAM), so the fp32 3B load fits a smaller
    # runtime (e.g. a standard Colab T4 with ~12.7GB RAM), not just the A100 high-RAM box.
    model = _ModelCls.from_pretrained(cfg.base_model_id, torch_dtype=torch.float32,
                                      trust_remote_code=True, low_cpu_mem_usage=True)
    model.resize_token_embeddings(len(tok))

    # Mean/stat-init the new rows from the existing embedding distribution (avoid random init).
    emb = model.get_input_embeddings().weight.data
    new_ids = [tok.convert_tokens_to_ids(t) for t in new_tokens]
    old_mean = emb[:base_len].mean(dim=0)
    old_std = emb[:base_len].std(dim=0)
    for i in new_ids:
        emb[i] = old_mean + 0.02 * old_std * torch.randn_like(old_std)
    out_emb = model.get_output_embeddings()
    tied = out_emb is not None and out_emb.weight.data_ptr() == emb.data_ptr()
    if out_emb is not None and not tied:
        ow = out_emb.weight.data
        om, os_ = ow[:base_len].mean(0), ow[:base_len].std(0)
        for i in new_ids:
            ow[i] = om + 0.02 * os_ * torch.randn_like(os_)

    # --- preflight assertions (model_architecture.md list) ---
    n_tok = len(tok)
    n_in = model.get_input_embeddings().num_embeddings
    n_out = out_emb.out_features if out_emb is not None else n_in
    code_ids = [tok.convert_tokens_to_ids(code_token(k)) for k in range(256)]
    report = preflight_checks(base_len=base_len, n_tok=n_tok, n_in=n_in, n_out=n_out,
                              code_ids=code_ids, new_ids=new_ids, tied=tied)
    if not report["all_pass"]:
        raise SFTError(f"vocab-resize preflight FAILED: {report}")

    added_ids = {t: tok.convert_tokens_to_ids(t) for t in new_tokens}
    if preflight_only:
        print(f"[vocab-resize][preflight-only] PASS  base={base_len} -> {n_tok} (+{len(new_ids)}) "
              f"tied={report['tied_embedding_status']}")
        return report

    # Load the frozen tokenizer manifest to bind identity (best-effort; may be staged elsewhere).
    tok_manifest = {}
    try:
        from tokenizer.frozen import frozen_final_dir
        mp = frozen_final_dir() / "manifest.json"
        if mp.is_file():
            tok_manifest = json.loads(mp.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        pass

    # Load the FULL base processor and swap in the resized tokenizer, so the saved artifact carries
    # preprocessor_config.json + chat template. Fatal on failure: a processor-less dir would only
    # blow up later at train.py's AutoProcessor.from_pretrained (OSError before step 1).
    try:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(cfg.base_model_id, trust_remote_code=True)
        processor.tokenizer = tok  # the resized tokenizer (with the 259 added tokens)
    except Exception as exc:  # noqa: BLE001
        raise SFTError(
            f"could not load base processor for {cfg.base_model_id!r}; the resized model would be "
            f"missing preprocessor_config.json and train.py could not load it: {exc}") from exc

    manifest = build_vocab_resize_manifest(
        base_model_id=cfg.base_model_id, base_vocab_size=base_len,
        vocab_size_after_resize=n_tok, added_special_token_ids=added_ids,
        tied_embedding_status=report["tied_embedding_status"], tokenizer_manifest=tok_manifest,
        preflight=report)
    out = _write_artifacts(out_dir or "models/base_resized", model, processor, tok, manifest)
    print(f"[vocab-resize][OK] {out}  base={base_len} -> {n_tok} (+{len(new_ids)}) "
          f"tied={report['tied_embedding_status']}")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(_DEFAULT_CFG_PATH))
    ap.add_argument("--out", default="models/base_resized")
    ap.add_argument("--preflight-only", action="store_true", help="assert only; write nothing")
    args = ap.parse_args(argv)
    cfg = _load_config(args.config)
    try:
        resize_and_preflight(cfg, args.out, preflight_only=args.preflight_only)
    except SFTError as exc:
        print(f"[vocab-resize][ABORT] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
