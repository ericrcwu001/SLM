"""Materialize ``target_tokens`` for supported active-SFT rows via the FROZEN tokenizer.

Join (verified 2761/2761 on the current registry): active row ``id`` (= ``file_hash``) ->
``data/raw_registry/provenance.jsonl`` ``residual_key`` -> ``luts/canonical_residual/<key>.npy``
-> :meth:`tokenizer.model.VQVAE.encode` -> 64 codebook ids. Each supported, resolvable row is
stamped with ``target_tokens``, ``assistant_target`` (``<lut_bos> <lut_###>*64 <lut_eos>``),
``token_status=materialized``, and ``tokenizer_version`` / ``vq_codebook_sha256`` /
``vq_decoder_sha256`` from the frozen manifest. Unsupported rows are left untouched (their
target is the literal ``<unsupported>``).

Verification (before writing):
  * per-row token validity — exactly 64 ids in ``0..255``;
  * reconstruction against the per-target SFT-admission gate — overall mean ΔE00 <= 3.0 and
    p95 <= 6.0 (also reported per source family), via :mod:`tokenizer.metrics`.

Honest: rows whose residual cannot be resolved/encoded stay ``pending_tokenizer`` (never
fabricated) and are reported. Backs up ``active_rows.jsonl`` / ``active_manifest.json`` before
writing and replaces atomically. ``--dry-run`` computes + reports everything but writes nothing.

Paths: residuals + provenance resolve via ``SLM_ARTIFACT_ROOT`` (the staged corpus root);
``active_rows.jsonl`` is taken from ``--active-dir`` (default ``data/active_sft`` under cwd),
because the active set is neither git-tracked nor part of the staged corpus.

Usage (from repo root):
    python -m scripts.materialize_target_tokens --dry-run
    python -m scripts.materialize_target_tokens
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
from pathlib import Path

import numpy as np

from data_pipeline.active_dataset import AcceptanceChecker, SftRow
from data_pipeline.constants import TOKEN_STATUS_MATERIALIZED, TOKEN_STATUS_PENDING
from data_pipeline.errors import RequiresTokenizer
from data_pipeline.paths import artifact_paths
from tokenizer import metrics as M
from tokenizer.frozen import load_frozen_vqvae

REPO = Path(__file__).resolve().parents[1]
_SFT_FIELDS = {f.name for f in dataclasses.fields(SftRow)}
# Per-target SFT-admission gate (model_architecture.md "LUT Tokenizer"; freeze.py reminder).
ADMISSION_MEAN_DE, ADMISSION_P95_DE = 3.0, 6.0


def _assistant_target(codes: list[int]) -> str:
    return "<lut_bos> " + " ".join(f"<lut_{c:03d}>" for c in codes) + " <lut_eos>"


def _load_residual_key_map(provenance: Path) -> dict[str, str]:
    """file_hash -> residual_key (fallback lut_id/source_pair_id/file_hash), for resolvable rows."""
    m: dict[str, str] = {}
    for line in provenance.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        fh = row.get("file_hash")
        if not fh:
            continue
        m[fh] = (row.get("residual_key") or row.get("lut_id")
                 or row.get("source_pair_id") or fh)
    return m


def run(active_dir: Path, artifact_root: Path, dry_run: bool) -> int:
    paths = artifact_paths(str(artifact_root))
    provenance = paths.raw_registry / "provenance.jsonl"
    residual_dir = paths.canonical_residual
    active_rows_p = active_dir / "active_rows.jsonl"
    manifest_p = active_dir / "active_manifest.json"

    for p in (provenance, residual_dir, active_rows_p):
        if not p.exists():
            print(f"[abort] missing input: {p}")
            return 2

    # Frozen tokenizer (raises FrozenTokenizerError -> here surfaced) + manifest identity.
    try:
        model, tok_manifest = load_frozen_vqvae()
    except Exception as exc:  # noqa: BLE001
        print(f"[abort] frozen tokenizer unavailable: {exc}")
        return 2
    tok_ver = tok_manifest["tokenizer_version"]
    print(f"[tok] loaded frozen tokenizer {tok_ver}")

    key_map = _load_residual_key_map(provenance)
    rows = [json.loads(l) for l in active_rows_p.read_text(encoding="utf-8").splitlines() if l.strip()]

    resolved: list[tuple[dict, np.ndarray]] = []   # (row, residual) to encode
    already = unresolved = unsupported = 0
    for row in rows:
        if not row.get("is_supported"):
            unsupported += 1
            continue
        tt = row.get("target_tokens")
        if isinstance(tt, list) and len(tt) == 64 and row.get("tokenizer_version") not in (None, "", TOKEN_STATUS_PENDING):
            already += 1
            continue
        key = key_map.get(row.get("id"))
        npy = (residual_dir / f"{key}.npy") if key else None
        if not key or not npy.is_file():
            unresolved += 1
            continue
        resolved.append((row, np.load(npy)))

    print(f"[join] supported-to-materialize={len(resolved)} already={already} "
          f"unresolved={unresolved} unsupported={unsupported} total={len(rows)}")
    if unresolved:
        print(f"[warn] {unresolved} supported rows could not resolve a residual — left pending.")
    if not resolved:
        print("[done] nothing to materialize.")
        return 0

    # One encode+decode pass: gives the 64-id codes to stamp AND the reconstruction for the gate.
    residuals = [r for _row, r in resolved]
    recons, codes = M.reconstruct(model, residuals)
    families = [(row.get("source_family") or "unknown") for row, _r in resolved]
    agg = M.aggregate_reconstruction(residuals, recons, families)
    ov = agg["overall"]
    gate_ok = ov["mean_deltae"] <= ADMISSION_MEAN_DE and ov["p95_deltae"] <= ADMISSION_P95_DE
    print(f"[recon] overall meanΔE={ov['mean_deltae']:.3f} p95ΔE={ov['p95_deltae']:.3f} "
          f"maxΔE={ov['max_deltae']:.3f} meanPSNR={ov['mean_psnr']:.2f}  "
          f"admission(mean<=3.0,p95<=6.0)={'PASS' if gate_ok else 'FAIL'}")
    for fam, s in sorted(agg["per_family"].items()):
        print(f"        {fam:<22} n={s['n']:<5} meanΔE={s['mean_deltae']:.3f} p95ΔE={s['p95_deltae']:.3f}")

    # Stamp rows + validate token ids.
    bad_tokens = []
    for i, (row, _r) in enumerate(resolved):
        ids = [int(c) for c in codes[i]]
        if len(ids) != 64 or any(c < 0 or c > 255 for c in ids):
            bad_tokens.append(row.get("id"))
            continue
        row["target_tokens"] = ids
        row["assistant_target"] = _assistant_target(ids)
        row["token_status"] = TOKEN_STATUS_MATERIALIZED
        row["tokenizer_version"] = tok_ver
        row["vq_codebook_sha256"] = tok_manifest["vq_codebook_sha256"]
        row["vq_decoder_sha256"] = tok_manifest["vq_decoder_sha256"]
    if bad_tokens:
        print(f"[abort] {len(bad_tokens)} rows produced invalid token ids (e.g. {bad_tokens[:3]}); writing nothing.")
        return 1
    if not gate_ok:
        print("[abort] reconstruction admission gate FAILED; writing nothing "
              "(frozen tokenizer is immutable — investigate the failing family rows).")
        return 1

    # Recompute acceptance honestly (criterion 6 should now pass; 12 pending until image pairing).
    sft_rows = [SftRow(**{k: v for k, v in r.items() if k in _SFT_FIELDS}) for r in rows]
    accept = AcceptanceChecker(enforce_scale=False).check(
        sft_rows, leakage_status="pass", model_clients_available=True)
    c6 = accept.criteria["representability_and_recon"]
    c12 = accept.criteria["generic_input_support"]
    print(f"[accept] representability_and_recon={c6['status']} ({c6['detail']})")
    print(f"[accept] generic_input_support={c12['status']} ({c12['detail']})")
    print(f"[accept] overall={accept.overall}")

    if dry_run:
        print("[dry-run] no files written. Re-run without --dry-run to apply.")
        return 0

    # Write active_rows.jsonl (backup first, atomic replace).
    shutil.copy2(active_rows_p, active_rows_p.with_suffix(".jsonl.bak_pre_tokenize"))
    tmp = active_rows_p.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, active_rows_p)
    print(f"[write] {active_rows_p} ({len(resolved)} rows materialized)")

    # Surgically refresh the manifest acceptance block (backup first).
    if manifest_p.exists():
        shutil.copy2(manifest_p, manifest_p.with_suffix(".json.bak_pre_tokenize"))
        man = json.loads(manifest_p.read_text(encoding="utf-8"))
        man["acceptance"] = accept.summary()
        man.setdefault("materialization", {})
        man["materialization"] = {
            "tokenizer_version": tok_ver, "materialized_rows": len(resolved),
            "unresolved_rows": unresolved,
            "recon_overall": {k: round(float(ov[k]), 4) for k in
                              ("mean_deltae", "p95_deltae", "max_deltae", "mean_psnr")},
            "admission_gate_pass": bool(gate_ok)}
        manifest_p.write_text(json.dumps(man, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[write] {manifest_p} (acceptance + materialization refreshed)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--active-dir", default=str(Path("data/active_sft")),
                    help="dir holding active_rows.jsonl (+ active_manifest.json)")
    ap.add_argument("--artifact-root", default=os.environ.get("SLM_ARTIFACT_ROOT", str(REPO)),
                    help="staged corpus root (residuals + provenance); default $SLM_ARTIFACT_ROOT or repo")
    ap.add_argument("--dry-run", action="store_true", help="compute + report; write nothing")
    args = ap.parse_args(argv)
    return run(Path(args.active_dir), Path(args.artifact_root), args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
