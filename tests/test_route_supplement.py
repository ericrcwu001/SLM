"""Pure (teacher-free) tests for the clarify + out_of_gamut route supplement (ADR 0023)."""

from __future__ import annotations

import importlib

grs = importlib.import_module("scripts.generate_route_supplement")


def test_plan_routes_and_kinds_and_fresh_ids():
    plan = grs.build_supplement_plan(n_clarify=3, n_gamut=6)
    clarify = [p for p in plan if p["id"].startswith("unsup_clarify_")]
    gamut = [p for p in plan if p["id"].startswith("unsup_gamut_")]
    assert len(clarify) == 3 and len(gamut) == 6
    assert all(p["route"] == "clarify" and p["refuse_kind"] is None for p in clarify)
    assert all(p["route"] == "refuse" and p["refuse_kind"] == "out_of_gamut" for p in gamut)
    # Each synthetic prompt is its own leakage unit (no shared identity), unique across the plan.
    units = [p["split_unit_id"] for p in plan]
    assert len(set(units)) == len(units)
    assert all(u.startswith("unsup:") for u in units)


def test_load_done_only_generated(tmp_path):
    cache = tmp_path / "supp_cache.jsonl"
    cache.write_text(
        '{"id": "unsup_gamut_000001", "status": "generated", "prompt": "make it infrared false color"}\n'
        '{"id": "unsup_clarify_000001", "status": "error", "error": "boom"}\n'
        '{"id": "unsup_clarify_000002", "status": "rejected", "prompt": "make it warmer"}\n',
        encoding="utf-8")
    done = grs._load_done(str(cache))
    assert set(done) == {"unsup_gamut_000001"}  # error + rejected retry


def test_assemble_builds_interpreter_rows():
    plan = grs.build_supplement_plan(n_clarify=1, n_gamut=1)
    cid, gid = "unsup_clarify_000001", "unsup_gamut_000001"
    cache = {
        cid: {"id": cid, "status": "generated", "prompt": "make it look nicer"},
        gid: {"id": gid, "status": "generated", "prompt": "rotate every hue around the wheel"},
        "unsup_gamut_000002": {"id": "unsup_gamut_000002", "status": "rejected", "prompt": "x"},
    }
    rows, counts = grs._assemble(cache, plan)
    assert counts == {"clarify": 1, "out_of_gamut": 1}
    by_id = {r["id"]: r for r in rows}
    assert by_id[cid]["route"] == "clarify" and by_id[cid]["refuse_kind"] is None
    assert by_id[cid]["instruction_natural"] == "make it look nicer"
    assert by_id[gid]["route"] == "refuse" and by_id[gid]["refuse_kind"] == "out_of_gamut"
    assert all(r["split_unit_id"].startswith("unsup:") for r in rows)
