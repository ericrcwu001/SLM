"""Re-measure ``measured_behavior`` for the active corpus at ``behavior_v2`` (ADR 0022).

Re-runs :func:`data_pipeline.behavior_vector.measure_behavior` (now behavior_v2) over each supported
active row's canonical LUT and writes the new versioned vector. This is the NEW versioned artifact
ADR 0026 calls for — the derived rows layer is regenerated, the frozen LUT/image corpus and the
tokenizer are never touched.

Join (identical to ``scripts.materialize_target_tokens``): active row ``id`` (= ``file_hash``) ->
``data/raw_registry/provenance.jsonl`` ``residual_key`` -> ``luts/canonical_residual/<key>.npy``;
absolute = residual + identity (``eval.cube_io.residual_to_absolute``). Deterministic + idempotent.

Follows the repo convention: back up ``active_rows.jsonl`` / ``active_manifest.json`` to
``*.bak_pre_behavior_v2``, write atomically, and record the re-measurement + version in the manifest.
Unsupported rows carry no LUT, so their (absent) ``measured_behavior`` is left untouched.

Usage (from repo root):
    python -m scripts.remeasure_behavior_v2 --dry-run
    python -m scripts.remeasure_behavior_v2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

from data_pipeline.behavior_vector import measure_behavior
from data_pipeline.constants import BEHAVIOR_VECTOR_VERSION
from data_pipeline.paths import artifact_paths
from eval.cube_io import residual_to_absolute

_ACTIVE_DIR = Path("data/active_sft")
_BAK = ".bak_pre_behavior_v2"


def _load_residual_key_map(provenance: Path) -> dict[str, str]:
    m: dict[str, str] = {}
    for line in provenance.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        fh = row.get("file_hash")
        if fh:
            m[fh] = row.get("residual_key") or row.get("lut_id") or row.get("source_pair_id") or fh
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

    key_map = _load_residual_key_map(provenance)
    rows = [json.loads(l) for l in active_rows_p.read_text(encoding="utf-8").splitlines() if l.strip()]

    remeasured = unresolved = unsupported = 0
    for row in rows:
        if not row.get("is_supported"):
            unsupported += 1
            continue
        key = key_map.get(row.get("id"))
        npy = (residual_dir / f"{key}.npy") if key else None
        if not key or not npy.is_file():
            unresolved += 1
            continue
        absolute = residual_to_absolute(np.load(npy))
        behavior = measure_behavior(absolute)
        row["measured_behavior"] = behavior
        row["behavior_vector_version"] = behavior["behavior_vector_version"]
        remeasured += 1

    print(f"[remeasure] supported_remeasured={remeasured} unresolved={unresolved} "
          f"unsupported={unsupported} total={len(rows)} -> {BEHAVIOR_VECTOR_VERSION}")
    if unresolved:
        print(f"[abort] {unresolved} supported rows could not resolve a residual — not writing "
              f"(check SLM_ARTIFACT_ROOT / the canonical_residual cache).")
        return 1
    if dry_run:
        print("[remeasure] DRY-RUN (no writes)")
        return 0

    shutil.copy2(active_rows_p, active_rows_p.with_name(active_rows_p.name + _BAK))
    tmp = active_rows_p.with_suffix(active_rows_p.suffix + ".tmp")
    tmp.write_text("\n".join(json.dumps(r, sort_keys=True) for r in rows) + "\n", encoding="utf-8")
    os.replace(tmp, active_rows_p)

    if manifest_p.exists():
        shutil.copy2(manifest_p, manifest_p.with_name(manifest_p.name + _BAK))
        man = json.loads(manifest_p.read_text(encoding="utf-8"))
        man["behavior_vector_version"] = BEHAVIOR_VECTOR_VERSION
        man["behavior_v2_remeasure"] = {
            "adr": "0022", "behavior_vector_version": BEHAVIOR_VECTOR_VERSION,
            "supported_remeasured": remeasured,
            "note": "measured_behavior recomputed at behavior_v2 from luts/canonical_residual",
        }
        manifest_p.write_text(json.dumps(man, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[remeasure] wrote {remeasured} behavior_v2 vectors -> {active_rows_p}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--active-dir", default=str(_ACTIVE_DIR))
    ap.add_argument("--artifact-root", default=os.environ.get("SLM_ARTIFACT_ROOT", os.getcwd()))
    ap.add_argument("--dry-run", action="store_true", help="report coverage, write nothing")
    args = ap.parse_args(argv)
    return run(Path(args.active_dir), Path(args.artifact_root), args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
