"""End-to-end pipeline smoke test — offline (procedural raw pool only, no network)."""

import json

from data_pipeline.run_pipeline import run_pipeline


def test_pipeline_offline_procedural(tmp_path):
    summary = run_pipeline(out_root=str(tmp_path), acquire=True,
                           only_sources=["procedural_fillers_v1"])
    root = tmp_path

    # every stage artifact exists
    assert (root / "data" / "raw_registry" / "provenance.jsonl").exists()
    assert (root / "data" / "raw_registry" / "derivation_attrition.json").exists()
    assert (root / "data" / "splits" / "split_manifest.json").exists()
    assert (root / "data" / "splits" / "leakage_report.json").exists()
    assert (root / "data" / "active_sft" / "active_manifest.json").exists()
    assert (root / "data" / "active_sft" / "active_rows.jsonl").exists()
    assert (root / "data" / "eval" / "eval_manifest.json").exists()
    assert list((root / "data" / "warmup").glob("*/manifest.json"))
    assert (root / "data" / "run_summary.json").exists()

    # real metrics for the computable stages
    af = summary["stages"]["4_5_derive_filter"]
    assert af["canonicalized"] >= 1
    assert af["gold"] >= 1                       # real representability tiers assigned
    assert summary["stages"]["6_splits_leakage"]["leakage_status"] == "pass"

    # canonical residual tensors written
    assert list((root / "luts" / "canonical_residual").glob("*.npy"))

    # gated steps are honest, never fabricated
    rows = [json.loads(l) for l in
            (root / "data" / "active_sft" / "active_rows.jsonl").read_text().splitlines() if l.strip()]
    assert rows
    assert all(r["token_status"] == "pending_tokenizer" for r in rows)
    assert all(r["target_tokens"] is None for r in rows)
    assert all(r["instruction_status"] == "pending_teacher" for r in rows)
    assert all(r["procedural_filler"] for r in rows)      # procedural -> train-only
    assert all(not r["headline_eligible"] for r in rows)  # procedural never headline

    # warmup token materialization pending
    wm = summary["stages"]["11_warmup"]
    assert wm["token_status"] == "pending_tokenizer"


def test_pipeline_reuses_existing_registry(tmp_path):
    run_pipeline(out_root=str(tmp_path), acquire=True, only_sources=["procedural_fillers_v1"])
    # second pass without acquisition still completes over the existing raw registry
    summary = run_pipeline(out_root=str(tmp_path), acquire=False)
    assert summary["stages"]["4_5_derive_filter"]["canonicalized"] >= 1
