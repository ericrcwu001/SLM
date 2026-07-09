"""Pair a leakage-safe generic input image to each LUT-only supported active-SFT row.

The supported rows from LUT-only families (scraped_web / fresh_luts / gmic_rawtherapee /
smaller_public_packs / controlled_procedural) have no natural source image. This assigns each a
unique FREE input image — the ppr10k/fivek pool minus every image already used by an active row
or reserved for eval — so every supported row is image-conditioned (Active Dataset criterion 12,
``generic_input_support``). It also normalizes ALL supported rows' ``image_path`` to REPO-RELATIVE
(``luts/raw/.../*.jpg``) so they resolve under the artifact root on Colab as well as locally.

Conditioning note: for a LUT-only row the target LUT is independent of the attached image, so these
rows teach prompt->LUT but give weak per-image signal. They are tagged ``image_pairing="generic"``
so the Stage 6/7 image-conditioning ablation can account for them; this does not affect the first
(smoke) SFT run.

Deterministic (seeded), backs up ``active_rows.jsonl`` / ``active_manifest.json`` before writing,
replaces atomically. ``--dry-run`` computes + reports but writes nothing.

Usage (from repo root):
    python -m scripts.pair_generic_images --dry-run
    python -m scripts.pair_generic_images
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import glob
import json
import os
import random
import shutil
from pathlib import Path

from data_pipeline.active_dataset import AcceptanceChecker, SftRow

REPO = Path(__file__).resolve().parents[1]
_SFT_FIELDS = {f.name for f in dataclasses.fields(SftRow)}
_SEED = 20260709
_LUT_ONLY_FAMILIES = {"scraped_web", "fresh_luts", "gmic_rawtherapee",
                      "smaller_public_packs", "controlled_procedural"}


def _rel(path: str) -> str:
    """Absolute or messy path -> repo/artifact-root-relative ``luts/...`` (root-independent)."""
    s = str(path)
    i = s.find("luts/")
    return s[i:] if i != -1 else s


def _pool(root: Path) -> list[str]:
    pats = ["luts/raw/ppr10k/**/before.jpg", "luts/raw/fivek*/**/*.jpg", "luts/raw/fivek*/**/*.png"]
    found: set[str] = set()
    for pat in pats:
        for f in glob.glob(str(root / pat), recursive=True):
            if os.path.isfile(f):
                found.add(os.path.abspath(f))
    return sorted(found)


def _excluded_images(active_dir: Path) -> set[str]:
    """Every image already used by an active row or reserved for eval (abs paths)."""
    excl: set[str] = set()
    scan = [active_dir / "active_rows.jsonl", active_dir / "unsupported_eval_rows.jsonl"]
    scan += [Path(p) for p in glob.glob("data/eval/*.jsonl")]
    for p in scan:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            ip = json.loads(line).get("image_path")
            if ip:
                # normalize to abs under repo for set membership vs the abs pool
                ip = ip if os.path.isabs(ip) else str(REPO / _rel(ip))
                excl.add(os.path.abspath(ip))
    return excl


def run(active_dir: Path, dry_run: bool) -> int:
    active_rows_p = active_dir / "active_rows.jsonl"
    manifest_p = active_dir / "active_manifest.json"
    if not active_rows_p.exists():
        print(f"[abort] no active_rows.jsonl at {active_rows_p}")
        return 2

    rows = [json.loads(l) for l in active_rows_p.read_text(encoding="utf-8").splitlines() if l.strip()]
    need_rows = sorted(
        [r for r in rows if r.get("is_supported") and not r.get("image_path")],
        key=lambda r: r.get("id", ""))
    fams = collections.Counter(r.get("source_family") for r in need_rows)
    print(f"[pair] supported rows needing an image: {len(need_rows)}  families={dict(fams)}")

    pool = _pool(REPO)
    excluded = _excluded_images(active_dir)
    free = [f for f in pool if f not in excluded]
    rng = random.Random(_SEED)
    rng.shuffle(free)
    print(f"[pool] total={len(pool)} excluded(used+eval)={len(excluded)} FREE={len(free)} need={len(need_rows)}")
    if len(free) < len(need_rows):
        print(f"[abort] only {len(free)} free images for {len(need_rows)} rows.")
        return 1

    # Assign unique free images to the LUT-only rows.
    assigned = 0
    for row, img in zip(need_rows, free):
        row["image_path"] = _rel(img)
        row["image_pairing"] = "generic"          # honest provenance (not a natural pair)
        assigned += 1

    # Normalize ALL supported rows' image_path to repo-relative (Colab portability).
    normalized = 0
    for r in rows:
        if r.get("is_supported") and r.get("image_path"):
            rel = _rel(r["image_path"])
            if rel != r["image_path"]:
                r["image_path"] = rel
                normalized += 1
    print(f"[pair] assigned generic images={assigned}  normalized existing paths={normalized}")

    # Recompute acceptance (criterion 12 should now pass; overall should reach pass).
    sft_rows = [SftRow(**{k: v for k, v in r.items() if k in _SFT_FIELDS}) for r in rows]
    accept = AcceptanceChecker(enforce_scale=False).check(
        sft_rows, leakage_status="pass", model_clients_available=True)
    for name in ("representability_and_recon", "generic_input_support"):
        c = accept.criteria[name]
        print(f"[accept] {name}={c['status']} ({c['detail']})")
    print(f"[accept] overall={accept.overall}")

    if dry_run:
        print("[dry-run] no files written. Re-run without --dry-run to apply.")
        return 0

    shutil.copy2(active_rows_p, active_rows_p.with_suffix(".jsonl.bak_pre_pairing"))
    tmp = active_rows_p.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, active_rows_p)
    print(f"[write] {active_rows_p}")

    if manifest_p.exists():
        shutil.copy2(manifest_p, manifest_p.with_suffix(".json.bak_pre_pairing"))
        man = json.loads(manifest_p.read_text(encoding="utf-8"))
        man["acceptance"] = accept.summary()
        man["image_pairing"] = {"generic_assigned": assigned, "paths_normalized": normalized,
                                "seed": _SEED, "pool_free_after": len(free) - assigned}
        manifest_p.write_text(json.dumps(man, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[write] {manifest_p} (acceptance + image_pairing refreshed)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--active-dir", default=str(Path("data/active_sft")))
    ap.add_argument("--dry-run", action="store_true", help="compute + report; write nothing")
    args = ap.parse_args(argv)
    return run(Path(args.active_dir), args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
