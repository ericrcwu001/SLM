"""Minimal prompt_to_lut CLI (model_architecture.md "Runtime Inference";
detailed_behavior_spec.md "CLI-First Behavior").

Stage-1 spine behavior (decoder disabled, no trained model):
  * There is no trained VLM, so the "model output" comes from ``--mock-output`` (raw
    text) or defaults to ``<unsupported>``. The model is recorded as ``mock`` in the
    manifest/metrics; gated ``--model qwen``/``checkpoint`` raise cleanly.
  * Runtime-constrained decoding projects the output onto the grammar (FSM), so a
    supported output is always syntactically valid.
  * Refusal path is complete: writes input.png, output_tokens.txt (``<unsupported>``),
    metrics.json (kind=unsupported), version_manifest.json — no LUT applied.
  * A valid LUT-token output cleanly BLOCKS at decode (``block_reason:
    decoder_disabled``): no graded.png / output.cube / preview, and NO silent identity
    LUT (spec: "must not silently replace an invalid LUT with identity").
  * ``--self-check`` validates the version-manifest fields available in this build and
    reports decoder/codebook fields as pending.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

from eval.baseline_adapters import ids_to_text
from eval.constrained_decoding import LutGrammarFSM
from eval.output_parsers import parse_output
from eval.schemas import (
    CANONICAL_DOMAIN_ID,
    CUBE_SERIALIZATION_VERSION,
    DECODER_DISABLED_REASON,
    FSM_VERSION,
    ICC_CONVERSION_CONFIG,
    PARSER_VERSION,
    SCHEMA_VERSION,
    build_version_manifest,
)
from eval.vocab import DEFAULT_VOCAB, NUM_SPECIAL_TOKENS, UNSUPPORTED

RUNTIME_CONSTRAINED = "runtime_constrained"
FREE_GENERATION = "free_generation"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _constrain(text: str, vocab, fsm: LutGrammarFSM) -> str:
    parsed = parse_output(text)
    if parsed.kind == "unsupported":
        return UNSUPPORTED
    cand = [vocab.bos_id] + [vocab.code_id(i) for i in parsed.token_ids] + [vocab.eos_id]
    return ids_to_text(fsm.project(cand), vocab)


def _save_input_png(image_path: str, out_path: str) -> None:
    """Write canonical input.png. Falls back to copying bytes if PIL can't open it."""
    try:
        from PIL import Image

        with Image.open(image_path) as im:
            im.convert("RGB").save(out_path, format="PNG")
    except Exception:
        with open(image_path, "rb") as src, open(out_path, "wb") as dst:
            dst.write(src.read())


def _version_manifest() -> dict:
    return build_version_manifest(
        vocab_added_special_token_ids=DEFAULT_VOCAB.added_special_token_ids,
        vocab_size_after_resize=None,
    )


def run_self_check() -> int:
    """Validate the version-manifest fields available in the decode-disabled spine."""
    manifest = _version_manifest()
    checks: list[tuple[str, bool, str]] = []
    checks.append(("parser_version", manifest["parser_version"] == PARSER_VERSION, PARSER_VERSION))
    checks.append(("fsm_version", manifest["fsm_version"] == FSM_VERSION, FSM_VERSION))
    checks.append(("canonical_domain_id", manifest["canonical_domain_id"] == CANONICAL_DOMAIN_ID,
                   CANONICAL_DOMAIN_ID))
    checks.append(("cube_serialization_version",
                   manifest["cube_serialization_version"] == CUBE_SERIALIZATION_VERSION,
                   CUBE_SERIALIZATION_VERSION))
    checks.append(("icc_conversion_config",
                   manifest["icc_conversion_config"] == ICC_CONVERSION_CONFIG, ICC_CONVERSION_CONFIG))
    checks.append(("added_special_token_ids==259",
                   len(manifest["added_special_token_ids"]) == NUM_SPECIAL_TOKENS,
                   str(NUM_SPECIAL_TOKENS)))
    checks.append(("token_suffix_to_codebook_index==identity",
                   manifest["token_suffix_to_codebook_index"] == "identity", "identity"))

    pending = [k for k in ("vq_codebook_sha256", "vq_decoder_sha256", "flatten_order",
                           "adapter_sha256", "vocab_size_after_resize", "base_model_revision")
               if str(manifest.get(k)).startswith("pending")]

    print("prompt_to_lut --self-check (Stage 1 spine)")
    ok = True
    for name, passed, expected in checks:
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {name} = {expected}")
    print(f"  [PENDING] decoder/model fields (require frozen VQ tokenizer): {', '.join(pending)}")
    print(f"  decoder_enabled = {manifest['decoder_enabled']}")
    return 0 if ok else 1


def run(image: str, prompt: str, out_dir: str, mock_output: str | None,
        model: str, mode: str) -> int:
    if model in ("qwen", "checkpoint"):
        from eval.baseline_adapters import RequiresModel

        raise RequiresModel(f"--model {model} requires a trained checkpoint + GPU (not in the spine)")

    os.makedirs(out_dir, exist_ok=True)
    run_id = os.path.basename(os.path.normpath(out_dir))

    if not os.path.exists(image):
        print(f"error: image not found: {image}", file=sys.stderr)
        return 2
    with open(image, "rb") as fh:
        image_sha = _sha256_bytes(fh.read())

    vocab = DEFAULT_VOCAB
    fsm = LutGrammarFSM(vocab)

    raw_output = mock_output if mock_output is not None else UNSUPPORTED
    if mode == RUNTIME_CONSTRAINED:
        raw_output = _constrain(raw_output, vocab, fsm)
    parsed = parse_output(raw_output)

    # always-written artifacts
    input_png = os.path.join(out_dir, "input.png")
    _save_input_png(image, input_png)

    # output_tokens.txt
    tokens_path = os.path.join(out_dir, "output_tokens.txt")
    if parsed.kind == "unsupported":
        token_text = UNSUPPORTED
    else:
        token_text = raw_output
    with open(tokens_path, "w", encoding="utf-8") as fh:
        fh.write(token_text + "\n")

    # version manifest
    manifest = _version_manifest()
    manifest_path = os.path.join(out_dir, "version_manifest.json")
    manifest_bytes = json.dumps(manifest, sort_keys=True, indent=2).encode("utf-8")
    with open(manifest_path, "wb") as fh:
        fh.write(manifest_bytes)
    manifest_sha = _sha256_bytes(manifest_bytes)

    # decide output kind + status
    if parsed.kind == "unsupported":
        kind, blocked, block_reason = "unsupported", False, None
    elif parsed.kind == "lut_tokens":
        # supported token sequence: decode is disabled -> BLOCK (no silent identity LUT)
        kind, blocked, block_reason = "lut_tokens", True, DECODER_DISABLED_REASON
    else:
        kind, blocked, block_reason = "invalid", True, "invalid_syntax"

    metrics = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "model": model,
        "input": {"image_path": image, "image_sha256": image_sha, "prompt": prompt},
        "output": {
            "kind": kind,
            "syntax_pass": parsed.syntax_pass,
            "token_count": parsed.token_count,
            "token_ids": parsed.token_ids,
            "parser_errors": parsed.parser_errors,
        },
        "decoding": {
            "mode": mode, "grammar_mask": mode == RUNTIME_CONSTRAINED,
            "fsm_version": FSM_VERSION, "do_sample": False, "num_beams": 1,
            "seed": 1234, "max_new_tokens": 67, "precision": "mock",
        },
        "lut": {
            "canonical_domain_id": CANONICAL_DOMAIN_ID, "grid_size": [17, 17, 17],
            "latent_shape": [4, 4, 4], "codebook_size": 256,
            "flatten_order": "pending:decoder_disabled", "interpolation": "trilinear",
            "vq_codebook_sha256": "pending:decoder_disabled",
            "vq_decoder_sha256": "pending:decoder_disabled",
            "cube_serialization_version": CUBE_SERIALIZATION_VERSION,
            "icc_conversion_config": ICC_CONVERSION_CONFIG,
        },
        "measured_behavior": {k: None for k in (
            "temperature_delta_b", "tint_delta_a", "mean_l_delta", "contrast_l_spread_delta",
            "highlight_l_delta", "shadow_l_delta", "chroma_delta", "neutral_drift_deltaE",
            "skin_locus_deltaE00_p95", "clip_rate", "smoothness", "foldover_rate")},
        "direction_checks": {"expected_attributes_source": "none", "checks": []},
        "status": {"blocked": blocked, "block_reason": block_reason},
        "version_manifest_sha256": manifest_sha,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, sort_keys=True)

    # supported-only artifacts (graded.png, preview, output.cube) require a decoded LUT.
    # Decoder disabled -> intentionally NOT written; no silent identity LUT.
    print(f"[prompt_to_lut] out={out_dir} kind={kind} "
          f"{'blocked:' + block_reason if blocked else 'ok'}")
    if kind == "lut_tokens":
        print("  supported tokens parsed & syntax-valid; decode blocked (decoder_disabled): "
              "no output.cube / graded.png written.")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="prompt_to_lut", description="Prompt-to-LUT CLI (Stage 1 spine).")
    ap.add_argument("--image")
    ap.add_argument("--prompt")
    ap.add_argument("--out")
    ap.add_argument("--mock-output", default=None,
                    help="raw model output text (mock model); defaults to <unsupported>")
    ap.add_argument("--model", default="mock", choices=["mock", "qwen", "checkpoint"])
    ap.add_argument("--mode", default=RUNTIME_CONSTRAINED,
                    choices=[RUNTIME_CONSTRAINED, FREE_GENERATION])
    ap.add_argument("--self-check", action="store_true")
    args = ap.parse_args(argv)

    if args.self_check:
        return run_self_check()

    if not (args.image and args.prompt and args.out):
        ap.error("--image, --prompt, and --out are required unless --self-check is given")
    from eval.baseline_adapters import RequiresDecoder, RequiresFrozenConfig, RequiresModel

    try:
        return run(args.image, args.prompt, args.out, args.mock_output, args.model, args.mode)
    except (RequiresModel, RequiresDecoder, RequiresFrozenConfig) as exc:
        print(f"prompt_to_lut: unavailable in the Stage-1 spine: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
