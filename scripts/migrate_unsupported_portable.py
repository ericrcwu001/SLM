"""Make the unsupported/refusal rows portable + route-tagged (ADR 0023; AUDIT F2 fix).

The 272 unsupported rows shipped with ABSOLUTE image paths (``/Users/.../luts/raw/...``) that do not
resolve on Colab, so ``sft.train`` skipped every refusal row every epoch and the refuse path never
trained. This migration rewrites those rows in place as a NEW VERSIONED artifact (per ADR 0026 data
governance — a regenerable derived layer, never the frozen corpus):

  * ``image_path`` -> corpus-relative (``luts/raw/...``), resolving against ``$SLM_ARTIFACT_ROOT``
    exactly like the supported rows (0 skips);
  * ``route`` = ``refuse`` and ``refuse_kind`` backfilled from the category
    (:mod:`eval.refuse_taxonomy`; the pre-existing 272 are all ``out_of_scope``).

Follows the repo convention (see ``scripts/pair_generic_images.py``): back up each file to
``*.bak_pre_portable_unsup``, rewrite atomically, and bump ``active_set_version`` + record the
migration in ``active_manifest.json``. Deterministic and IDEMPOTENT: re-running on already-portable
rows changes nothing but the backups. The frozen LUT/image corpus and ``luts/`` are never touched.

Usage:
    python -m scripts.migrate_unsupported_portable --dry-run
    python -m scripts.migrate_unsupported_portable
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from eval.refuse_taxonomy import ROUTE_REFUSE, refuse_kind_for_category
from scripts.generate_unsupported import to_portable_image_path

_ACTIVE_DIR = Path("data/active_sft")
_NEW_VERSION = "active_set_v2_portable_unsup"
_BAK = ".bak_pre_portable_unsup"
# Files that may carry unsupported rows with absolute paths.
_ROW_FILES = ("active_rows.jsonl", "unsupported_rows.jsonl", "unsupported_eval_rows.jsonl")


def _is_unsupported(row: dict) -> bool:
    return (row.get("is_supported") is False) or str(row.get("id", "")).startswith("unsup_")


def migrate_row(row: dict) -> tuple[dict, bool]:
    """Return (row, changed). Portabilizes image_path and backfills route/refuse_kind on refuse rows."""
    changed = False
    ip = row.get("image_path")
    if ip:
        portable = to_portable_image_path(ip)
        if portable != ip:
            row["image_path"] = portable
            changed = True
    if _is_unsupported(row):
        if not row.get("route"):
            row["route"] = ROUTE_REFUSE
            changed = True
        if not row.get("refuse_kind"):
            rk = refuse_kind_for_category(row.get("unsupported_category"))
            if rk:
                row["refuse_kind"] = rk
                changed = True
    return row, changed


def _process_file(path: Path, dry_run: bool) -> dict:
    if not path.exists():
        return {"path": str(path), "present": False}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    n_changed = 0
    unresolved: list[str] = []
    out_lines: list[str] = []
    for row in rows:
        row, changed = migrate_row(row)
        n_changed += int(changed)
        if _is_unsupported(row):
            ip = row.get("image_path") or ""
            resolved = ip if os.path.isabs(ip) else str(Path(os.environ.get("SLM_ARTIFACT_ROOT", os.getcwd())) / ip)
            if not os.path.exists(resolved):
                unresolved.append(f"{row.get('id')}::{ip}")
        out_lines.append(json.dumps(row, sort_keys=True))
    summary = {"path": str(path), "present": True, "rows": len(rows),
               "changed": n_changed, "unresolved": len(unresolved),
               "unresolved_examples": unresolved[:5]}
    if not dry_run and n_changed:
        shutil.copy2(path, path.with_name(path.name + _BAK))
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    return summary


def _update_manifest(summaries: list[dict], dry_run: bool) -> None:
    manifest_p = _ACTIVE_DIR / "active_manifest.json"
    if not manifest_p.exists():
        return
    man = json.loads(manifest_p.read_text(encoding="utf-8"))
    prior = man.get("active_set_version")
    man["active_set_version"] = _NEW_VERSION
    man["portable_unsupported_migration"] = {
        "prior_active_set_version": prior,
        "adr": "0023",
        "audit_finding": "F2",
        "note": "unsupported rows: absolute -> corpus-relative image_path; route/refuse_kind backfilled",
        "files": {s["path"]: {"rows": s.get("rows"), "changed": s.get("changed"),
                              "unresolved": s.get("unresolved")} for s in summaries if s.get("present")},
    }
    if not dry_run:
        shutil.copy2(manifest_p, manifest_p.with_name(manifest_p.name + _BAK))
        manifest_p.write_text(json.dumps(man, indent=2, sort_keys=True), encoding="utf-8")


def run(dry_run: bool) -> int:
    summaries = [_process_file(_ACTIVE_DIR / name, dry_run) for name in _ROW_FILES]
    total_unresolved = sum(s.get("unresolved", 0) for s in summaries if s.get("present"))
    for s in summaries:
        if not s.get("present"):
            print(f"[migrate][skip] {s['path']} (absent)")
            continue
        print(f"[migrate] {s['path']}: rows={s['rows']} changed={s['changed']} "
              f"unresolved={s['unresolved']} {s['unresolved_examples'] or ''}")
    if total_unresolved:
        print(f"[migrate][ABORT] {total_unresolved} unsupported rows still unresolvable locally — "
              f"NOT writing manifest version bump (check SLM_ARTIFACT_ROOT / corpus).")
        return 1
    _update_manifest(summaries, dry_run)
    print(f"[migrate] {'DRY-RUN (no writes)' if dry_run else f'active_set_version -> {_NEW_VERSION}'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true", help="report changes, write nothing")
    args = ap.parse_args(argv)
    return run(args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
