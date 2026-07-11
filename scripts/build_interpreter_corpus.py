"""Unify the Stage-1 interpreter training corpus + apply the leakage fix (ADR 0021/0024).

Joins three sources into one leakage-safe ``interpreter_rows.jsonl`` (input text → target
``attribute_spec_text`` + route):

  1. **grade** — ``caption_rows.jsonl`` (teacher captions of real LUTs). These carry ``source_lut_id``
     but NO ``split_unit_id``; we stamp each with the source LUT's ``split_unit_id`` from
     ``active_rows.jsonl``. **This is the load-bearing leakage fix:** without it every one of a LUT's
     5 style-captions would get an independent holdout coin-flip (the exact row-id-carve leak that
     inflated the generator holdout 48.5%, ADR 0024). All captions of a LUT must share one unit.
  2. **refuse / out_of_scope** — the ``route=="refuse"`` rows already in ``active_rows.jsonl`` (input =
     ``instruction_natural``). Keep their ``split_unit_id``.
  3. **clarify + refuse / out_of_gamut** — ``route_supplement_rows.jsonl`` (scripts.generate_route_supplement).

Targets are canonical ``attribute_spec_text``: grade rows reuse the caption's grounded target;
refuse/clarify rows are ``serialize(AttributeSpec(route=..., refuse_reason=...))``.

Pure/local (seconds); never trains, never touches the frozen corpus. Output:
``data/interpreter/interpreter_rows.jsonl`` + ``interpreter_corpus_manifest.json``.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import shutil
from pathlib import Path
from typing import Optional

from data_pipeline.attribute_spec import AttributeSpec, serialize
from eval.refuse_taxonomy import ROUTE_CLARIFY, ROUTE_GRADE, ROUTE_REFUSE

_ACTIVE_ROWS = "data/active_sft/active_rows.jsonl"
_CAPTION_ROWS = "data/active_sft/caption_rows.jsonl"
_SUPPLEMENT = "data/active_sft/route_supplement_rows.jsonl"
_OUT = "data/interpreter/interpreter_rows.jsonl"
_CORPUS_VERSION = "interpreter_v1"


def _read_jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def _lut_to_unit(active_rows: list[dict]) -> dict[str, str]:
    """Map the captioner's LUT key (``source_lut_id or id``) → the LUT's ``split_unit_id``.

    The captioner sets a caption row's ``source_lut_id`` to exactly ``active.source_lut_id or
    active.id`` (generate_captions.py), so keying the map the same way guarantees the join lands.
    """
    m: dict[str, str] = {}
    for r in active_rows:
        if not r.get("is_supported"):
            continue
        key = r.get("source_lut_id") or r.get("id")
        unit = r.get("split_unit_id")
        if key and unit:
            m[key] = unit
    return m


def _refuse_target(refuse_kind: Optional[str]) -> str:
    return serialize(AttributeSpec(route=ROUTE_REFUSE, refuse_reason=refuse_kind or "out_of_scope"))


def _row(id_, text, target, route, refuse_kind, source_lut_id, unit, style, family) -> dict:
    return {"id": id_, "text": text, "attribute_spec_text": target,
            "route": route, "refuse_kind": refuse_kind,
            "source_lut_id": source_lut_id, "split_unit_id": unit,
            "style": style, "source_family": family,
            "interpreter_corpus_version": _CORPUS_VERSION}


def build_rows(caption_rows: list[dict], active_rows: list[dict],
               supplement_rows: list[dict]) -> tuple[list[dict], dict]:
    """Return (unified rows, stats). Pure — no IO. Grade rows with no matching unit are dropped."""
    lut_to_unit = _lut_to_unit(active_rows)
    out: list[dict] = []
    dropped_missing_unit = 0

    # 1. grade (captions) — stamp the source LUT's split_unit_id.
    for r in caption_rows:
        slut = r.get("source_lut_id")
        unit = lut_to_unit.get(slut) if slut else None
        if not unit:
            dropped_missing_unit += 1
            continue
        out.append(_row(r["id"], r["caption"], r["attribute_spec_text"], ROUTE_GRADE, None,
                        slut, unit, r.get("style"), "caption"))

    # 2. refuse / out_of_scope — the route=="refuse" rows already in active_rows.
    for r in active_rows:
        if r.get("route") != ROUTE_REFUSE:
            continue
        text = r.get("instruction_natural") or r.get("instruction")
        kind = r.get("refuse_kind") or "out_of_scope"
        out.append(_row(r["id"], text, _refuse_target(kind), ROUTE_REFUSE, kind,
                        None, r.get("split_unit_id"), None, r.get("source_family") or "unsupported_teacher"))

    # 3. clarify + refuse/out_of_gamut — the supplement.
    for r in supplement_rows:
        text = r.get("instruction_natural") or r.get("instruction")
        route = r.get("route")
        kind = r.get("refuse_kind")
        target = serialize(AttributeSpec(route=ROUTE_CLARIFY)) if route == ROUTE_CLARIFY \
            else _refuse_target(kind)
        out.append(_row(r["id"], text, target, route, kind,
                        None, r.get("split_unit_id"), None, r.get("source_family") or "unsupported_teacher"))

    # A missing split_unit_id anywhere would silently fall back to the row id at holdout time
    # (leak). Every unified row must carry one; report the count that don't.
    fallback_key_count = sum(1 for r in out if not r.get("split_unit_id"))
    stats = {
        "total_rows": len(out),
        "dropped_missing_unit": dropped_missing_unit,
        "fallback_key_count": fallback_key_count,
        "by_route": dict(collections.Counter(r["route"] for r in out)),
        "by_refuse_kind": dict(collections.Counter(r["refuse_kind"] for r in out if r["refuse_kind"])),
        "by_source_family": dict(collections.Counter(r["source_family"] for r in out)),
        "by_style": dict(collections.Counter(r["style"] for r in out if r["style"])),
        "distinct_units": len({r["split_unit_id"] for r in out if r["split_unit_id"]}),
        "units_per_route": {rt: len({r["split_unit_id"] for r in out
                                     if r["route"] == rt and r["split_unit_id"]})
                            for rt in (ROUTE_GRADE, ROUTE_CLARIFY, ROUTE_REFUSE)},
    }
    return out, stats


def _write_atomic(path: str, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        shutil.copy2(p, p.with_suffix(p.suffix + ".bak_pre_build"))
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, p)


def run(caption_rows_path: str, active_rows_path: str, supplement_path: str, out: str) -> int:
    caption_rows = _read_jsonl(caption_rows_path)
    active_rows = _read_jsonl(active_rows_path)
    supplement_rows = _read_jsonl(supplement_path)
    if not caption_rows:
        print(f"[interp-corpus] no caption rows at {caption_rows_path} — run "
              f"scripts.generate_captions first. Nothing written.")
        return 2
    if not supplement_rows:
        print(f"[interp-corpus] WARNING: no route supplement at {supplement_path} — clarify + "
              f"out_of_gamut routes will be ABSENT (run scripts.generate_route_supplement for 3-way).")

    rows, stats = build_rows(caption_rows, active_rows, supplement_rows)
    if stats["fallback_key_count"]:
        raise SystemExit(f"[interp-corpus] {stats['fallback_key_count']} rows lack split_unit_id "
                         f"(would leak at holdout) — aborting.")

    _write_atomic(out, rows)
    manifest = {"interpreter_corpus_version": _CORPUS_VERSION, "out": out,
                "sources": {"caption_rows": caption_rows_path, "active_rows": active_rows_path,
                            "route_supplement": supplement_path}, **stats}
    # Manifest lives beside the corpus (so a redirected --out keeps them together); the parent dir
    # was already created by _write_atomic.
    manifest_path = Path(out).parent / "interpreter_corpus_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[interp-corpus] wrote {len(rows)} rows -> {out}")
    print(f"[interp-corpus] {json.dumps(stats, sort_keys=True)}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Unify the interpreter training corpus (leakage-safe).")
    ap.add_argument("--caption-rows", default=_CAPTION_ROWS)
    ap.add_argument("--active-rows", default=_ACTIVE_ROWS)
    ap.add_argument("--supplement", default=_SUPPLEMENT)
    ap.add_argument("--out", default=_OUT)
    args = ap.parse_args(argv)
    return run(args.caption_rows, args.active_rows, args.supplement, args.out)


if __name__ == "__main__":
    raise SystemExit(main())
