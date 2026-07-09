"""Load the FROZEN VQ tokenizer for inference (encode / decode).

``tokenizer/freeze.py`` writes the frozen artifacts to ``tokenizer/final/`` — ``model.pt``
is a **bare** ``state_dict`` (not a training checkpoint), alongside split
``encoder.pt``/``decoder.pt``/``codebook.npy`` and ``manifest.json`` — and they ship via HF
staging under ``$SLM_ARTIFACT_ROOT/tokenizer/final/``. This module loads ``model.pt`` into a
:class:`tokenizer.model.VQVAE`, verifies the loaded weights reproduce the manifest hashes,
and caches the instance.

It NEVER retrains, re-gates, or modifies the tokenizer — load-and-use only (the frozen
tokenizer is immutable per the Canonical LUT Contract). Importing this module imports torch;
callers that must stay dependency-light (the gated stubs) import it lazily.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

import torch

from .config import DEFAULT_CONFIG
from .manifest import hash_state_dict, hash_tensor
from .model import VQVAE


class FrozenTokenizerError(RuntimeError):
    """Frozen tokenizer weights are missing, or do not match the manifest identity."""


def frozen_final_dir() -> Path:
    """Resolve the frozen ``tokenizer/final`` directory.

    Prefers ``$SLM_ARTIFACT_ROOT/tokenizer/final`` — the staged corpus (where the real
    weights live; a plain ``git clone`` ships only ``manifest.json`` because the ``.pt``
    files are gitignored) — and falls back to the repo-relative ``tokenizer/final``.
    """
    root = os.environ.get("SLM_ARTIFACT_ROOT")
    if root:
        cand = Path(root) / "tokenizer" / "final"
        if (cand / "model.pt").is_file():
            return cand
    return Path("tokenizer") / "final"


@lru_cache(maxsize=None)
def load_frozen_vqvae(final_dir: str | None = None):
    """Load + integrity-verify the frozen ``VQVAE``. Cached per resolved directory.

    Returns ``(model, manifest)``. Raises :class:`FrozenTokenizerError` when the weights are
    absent or their hashes disagree with the manifest
    (``vq_codebook_sha256`` / ``vq_encoder_sha256`` / ``vq_decoder_sha256``).
    """
    d = Path(final_dir) if final_dir else frozen_final_dir()
    model_pt, manifest_p = d / "model.pt", d / "manifest.json"
    if not model_pt.is_file() or not manifest_p.is_file():
        raise FrozenTokenizerError(
            f"frozen tokenizer not found at {d} (need model.pt + manifest.json). "
            "Stage the corpus (slm_stage) or set SLM_ARTIFACT_ROOT to the staged root."
        )
    manifest = json.loads(manifest_p.read_text(encoding="utf-8"))

    if manifest.get("arch_version") != DEFAULT_CONFIG.arch_version:
        raise FrozenTokenizerError(
            f"manifest arch_version {manifest.get('arch_version')!r} != code "
            f"DEFAULT_CONFIG.arch_version {DEFAULT_CONFIG.arch_version!r} — refusing to load "
            "a mismatched architecture."
        )

    model = VQVAE(DEFAULT_CONFIG)
    model.load_state_dict(torch.load(model_pt, map_location="cpu"))
    model.eval()

    # Integrity: the loaded weights must reproduce the manifest hashes (freeze.py wrote them
    # with these same helpers). This is the load-time analogue of the runbook Step-5 gate.
    got = {
        "vq_codebook_sha256": hash_tensor(model.vq.codebook),
        "vq_encoder_sha256": hash_state_dict(model.encoder.state_dict()),
        "vq_decoder_sha256": hash_state_dict(model.decoder.state_dict()),
    }
    mism = {k: (v, manifest.get(k)) for k, v in got.items() if manifest.get(k) and v != manifest[k]}
    if mism:
        raise FrozenTokenizerError(f"frozen weight hashes disagree with manifest: {mism}")

    return model, manifest
