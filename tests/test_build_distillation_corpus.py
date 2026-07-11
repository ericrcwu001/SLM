"""Tests for the self-distillation corpus builder (scripts.build_distillation_corpus).

The row transform `distill_row` is pure (no model, no torch) — tested directly. A separate
torch-guarded test pins the local `_assistant_target` against the materializer's format.
"""

from __future__ import annotations

import argparse

import pytest

from scripts.build_distillation_corpus import _assistant_target, distill_row
from sft.example import is_supported_materialized


def _train_row():
    return {
        "id": "row1", "is_supported": True, "image_path": "img/x.jpg",
        "instruction": "make it warmer", "assistant_target": "<lut_bos> <lut_000> <lut_eos>",
        "target_tokens": [1] * 64, "token_status": "materialized",
        "split_unit_id": "unit_x", "source_family": "fivek_derived",
        "measured_behavior": {"temperature_delta_b": 3.0},
        "tokenizer_version": "vq_v2_srgbres_17to4_cb256_t64__w91cffdd2c82f",
    }


def test_distill_replaces_when_winner_clears_tau():
    row = _train_row()
    winner = [i % 256 for i in range(64)]
    out = distill_row(row, winner, best_fid=0.45, tau=0.30)
    assert out is not row
    assert out["target_tokens"] == winner
    assert out["assistant_target"] == _assistant_target(winner)
    assert out["token_status"] == "distilled"
    # everything else preserved
    for k in ("id", "image_path", "instruction", "split_unit_id", "source_family", "measured_behavior"):
        assert out[k] == row[k]
    assert is_supported_materialized(out)   # still a valid training row


def test_distill_keeps_gold_below_tau():
    row = _train_row()
    assert distill_row(row, [i % 256 for i in range(64)], best_fid=0.20, tau=0.30) is row  # identity


def test_distill_keeps_gold_on_non_64():
    row = _train_row()
    assert distill_row(row, [1, 2, 3], best_fid=0.9, tau=0.30) is row       # 64-guard
    assert distill_row(row, None, best_fid=0.9, tau=0.30) is row            # all-refused


def test_distill_none_fidelity_keeps_gold():
    row = _train_row()
    assert distill_row(row, [i % 256 for i in range(64)], best_fid=None, tau=0.30) is row


def test_assistant_target_matches_materializer():
    torch = pytest.importorskip("torch")  # materializer imports torch via tokenizer.frozen  # noqa: F841
    from scripts.materialize_target_tokens import _assistant_target as canonical
    codes = [0, 1, 42, 255] + [7] * 60
    assert _assistant_target(codes) == canonical(codes)


def test_run_routing_holdout_and_unsupported_never_harvested(tmp_path, monkeypatch):
    """Critical safety invariant: the model is called ONLY on training supported rows; holdout and
    unsupported rows are copied unchanged (no generation)."""
    import json as _json

    import eval.best_of_n as BN
    import scripts.build_distillation_corpus as D
    import sft.loader as L

    train = {**_train_row(), "id": "train1"}
    hold = {**_train_row(), "id": "hold1", "split_unit_id": "unit_hold"}
    unsup = {"id": "unsup1", "is_supported": False, "assistant_target": "<unsupported>",
             "target_tokens": [], "image_path": "img/u.jpg", "instruction": "grass greener"}
    src = tmp_path / "src.jsonl"
    src.write_text("\n".join(_json.dumps(r) for r in (train, hold, unsup)) + "\n", encoding="utf-8")

    monkeypatch.setattr(D, "is_holdout_row", lambda row: row.get("id") == "hold1")

    class _Dummy:
        device = "cpu"

    monkeypatch.setattr(L, "load_eval_model", lambda *a, **k: (_Dummy(), None))
    called: list = []

    def _fake_best_of_n(model, processor, row, **kw):
        called.append(row["id"])
        return [i % 256 for i in range(64)], {"behavioral_fidelity": 0.45}

    monkeypatch.setattr(BN, "best_of_n_for_row", _fake_best_of_n)

    args = argparse.Namespace(
        config="configs/candidate_two_stage.json", source_rows=str(src), out_dir=str(tmp_path / "out"),
        resized_model="x", adapter="y", n=4, chunk=4, temperature=1.0, top_p=0.9, tau=0.30,
        limit=0, dry_run=False)
    assert D.run(args) == 0

    assert called == ["train1"]      # model NEVER ran on holdout or unsupported
    out = [_json.loads(l) for l in (tmp_path / "out" / "active_rows.jsonl").read_text().splitlines() if l.strip()]
    by_id = {r["id"]: r for r in out}
    assert by_id["hold1"]["target_tokens"] == [1] * 64            # holdout untouched
    assert by_id["hold1"]["token_status"] == "materialized"
    assert by_id["unsup1"]["assistant_target"] == "<unsupported>"  # unsupported untouched
    assert by_id["train1"]["token_status"] == "distilled"          # training row distilled
    assert by_id["train1"]["target_tokens"] == [i % 256 for i in range(64)]
