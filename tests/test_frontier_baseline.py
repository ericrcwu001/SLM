"""Tests for the prompted-frontier raw-.cube baseline (parser + scoring + orchestrator).

Runs fully offline — no Anthropic API. Generation (eval.frontier_client) is exercised
only for its pure helpers (system prompt); the network path is not called here.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from eval import cube_io
from eval.cube_parser import INVALID, RAW_LUT, UNSUPPORTED_KIND, parse_frontier_cube
from eval.frontier_scoring import NOT_EVALUATED, evaluate_direction, score_lut
from eval.schemas import load_rows


# --- helpers ---------------------------------------------------------------------
def _lut(tag: str) -> np.ndarray:
    lut = cube_io.identity_grid(17).copy()
    if tag == "warmer":
        lut[..., 0] = np.clip(lut[..., 0] + 0.06, 0, 1)
        lut[..., 2] = np.clip(lut[..., 2] - 0.06, 0, 1)
    elif tag == "cooler":
        lut[..., 0] = np.clip(lut[..., 0] - 0.06, 0, 1)
        lut[..., 2] = np.clip(lut[..., 2] + 0.06, 0, 1)
    elif tag == "brighter":
        lut = np.clip(lut + 0.06, 0, 1)
    elif tag == "darker":
        lut = np.clip(lut - 0.06, 0, 1)
    return lut


def _cube_text(tag: str) -> str:
    return cube_io.serialize_cube(_lut(tag)).decode("utf-8")


# --- parser ----------------------------------------------------------------------
def test_parse_valid_cube():
    p = parse_frontier_cube(_cube_text("warmer"))
    assert p.kind == RAW_LUT and p.size == 17 and p.lut_abs.shape == (17, 17, 17, 3)


def test_parse_unsupported():
    assert parse_frontier_cube("<unsupported>").kind == UNSUPPORTED_KIND
    assert parse_frontier_cube("  <unsupported>\n").kind == UNSUPPORTED_KIND


def test_parse_markdown_fence_tolerated():
    fenced = "```cube\n" + _cube_text("warmer") + "\n```"
    assert parse_frontier_cube(fenced).kind == RAW_LUT


def test_parse_trailing_prose_tolerated():
    p = parse_frontier_cube(_cube_text("warmer") + "\nDone! Enjoy your LUT.")
    assert p.kind == RAW_LUT


@pytest.mark.parametrize("text", ["", None, "Sure, here is your LUT!"])
def test_parse_invalid_no_header(text):
    assert parse_frontier_cube(text).kind == INVALID


def test_parse_wrong_size():
    body = "\n".join("0 0 0" for _ in range(8))
    p = parse_frontier_cube(f"LUT_3D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n{body}")
    assert p.kind == INVALID and "size_2_not_17" in p.errors[0]


def test_parse_truncated():
    lines = cube_io.serialize_cube(_lut("warmer")).decode().splitlines()
    p = parse_frontier_cube("\n".join(lines[:-100]))  # drop 100 data rows
    assert p.kind == INVALID and p.errors[0].startswith("truncated_")


def test_parse_out_of_range():
    lut = _lut("warmer")
    lut[0, 0, 0] = [5.0, 0.0, 0.0]  # wildly out of [0,1]
    p = parse_frontier_cube(cube_io.serialize_cube(lut).decode())
    assert p.kind == INVALID and p.errors[0].startswith("out_of_range")


def test_parse_mixed_refusal_and_lut():
    p = parse_frontier_cube("<unsupported>\n" + _cube_text("warmer"))
    assert p.kind == INVALID


# --- scoring ---------------------------------------------------------------------
def test_direction_pass_and_sign():
    warm = parse_frontier_cube(_cube_text("warmer")).lut_abs
    assert score_lut(warm, ["warmer"]).direction.status == "pass"
    assert score_lut(warm, ["cooler"]).direction.status == "fail"  # wrong sign


def test_identity_is_degenerate():
    ident = cube_io.identity_grid(17)
    s = score_lut(ident, ["warmer"])
    assert s.safety.status == "fail"  # residual_norm ~ 0
    assert s.direction.status == "fail"  # no movement
    assert s.lut_quality_pass is False


def test_style_only_tag_not_evaluated():
    warm = parse_frontier_cube(_cube_text("warmer")).lut_abs
    s = score_lut(warm, ["sepia"])  # non-directional style bundle
    assert s.direction.status == NOT_EVALUATED
    assert s.lut_quality_pass is None


def test_empty_tags_not_evaluated():
    assert evaluate_direction({}, []).status == NOT_EVALUATED


# --- orchestrator (end-to-end, offline) ------------------------------------------
def test_run_frontier_eval_offline(tmp_path):
    from eval import run_frontier_eval as rfe

    src = load_rows("data/eval/smoke_rows.jsonl")
    by_id = {r.id: r for r in src}
    picks = ["eval_sup_000001", "eval_sup_000002", "eval_sup_000005", "eval_unsup_000001"]
    rows = [by_id[i] for i in picks]

    rows_path = tmp_path / "rows.jsonl"
    with open(rows_path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r.to_dict()) + "\n")

    # crafted outputs: correct warm LUT; a WRONG-direction LUT (warm for a 'cooler' row);
    # a correct brighter LUT; and a correct refusal on the unsupported row.
    cube_dir = tmp_path / "cubes"
    cube_dir.mkdir()
    outs = {
        "eval_sup_000001": _cube_text("warmer"),   # warmer  -> pass
        "eval_sup_000002": _cube_text("warmer"),   # cooler row, warm LUT -> direction fail
        "eval_sup_000005": _cube_text("brighter"), # brighter -> pass
        "eval_unsup_000001": "<unsupported>",      # correct refusal
    }
    with open(cube_dir / "frontier_verify.jsonl", "w", encoding="utf-8") as fh:
        for rid, txt in outs.items():
            fh.write(json.dumps({"row_id": rid, "model": "verify",
                                 "model_id": "test", "text": txt,
                                 "provenance": {"output_tokens": 1}}) + "\n")

    run_dir = rfe.run(str(rows_path), str(tmp_path / "eval_runs"), str(cube_dir),
                      limit=4, model_names=["verify"], run_id="verify_run")

    import csv
    with open(os.path.join(run_dir, "frontier_overall.csv")) as fh:
        row = next(csv.DictReader(fh))

    assert int(row["N"]) == 4
    assert float(row["raw_cube_valid_rate"]) == 0.75          # 3 of 4 parsed as LUT
    assert float(row["unsupported_recall"]) == 1.0            # the 1 unsupported was refused
    assert float(row["over_refusal_rate"]) == 0.0             # no supported row refused
    assert float(row["boundary_accuracy"]) == 1.0             # all 4 boundary decisions correct
    assert int(row["direction_N"]) == 3                       # 3 supported directional LUTs
    assert float(row["direction_pass_rate"]) == pytest.approx(2 / 3, abs=1e-3)  # warmer+brighter pass, cooler fails
    assert row["target_fidelity_status"].startswith("not_evaluated")
