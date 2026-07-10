"""Oracle-gate pure logic (GPU-free): attribute_spec_text stamping + PASS/FAIL comparison."""

from __future__ import annotations

from sft import oracle_gate


def test_stamp_attribute_spec_text_from_measured_behavior():
    row = {"measured_behavior": {"temperature_delta_b": 2.4, "chroma_delta": -3.0}}
    oracle_gate._stamp_attribute_spec_text(row)
    text = row["attribute_spec_text"]
    assert text.startswith("route=grade")
    assert "warmer=+2.4" in text and "muted=+3.0" in text


def test_stamp_handles_missing_measured_behavior():
    row = {}
    oracle_gate._stamp_attribute_spec_text(row)
    assert row["attribute_spec_text"].startswith("route=grade")   # empty grade, no crash


def test_run_recommendation_pass_when_oracle_ge_baseline(monkeypatch):
    calls = {}

    def fake_score(cfg, resized, adapter, limit, *, input_field="instruction", prep_row=None):
        calls[input_field] = True
        acc = 0.55 if input_field == "attribute_spec_text" else 0.50
        return {"metric": acc, "overall_ci_low": acc - 0.02, "overall_ci_high": acc + 0.02,
                "scored_rows": 120, "scored_units": 120}

    monkeypatch.setattr(oracle_gate, "score", fake_score)
    rep = oracle_gate.run(cfg=object(), resized_model="m", adapter="a", limit=0)
    assert rep["recommendation"] == "PASS"
    assert rep["delta"] > 0
    assert calls == {"instruction": True, "attribute_spec_text": True}   # scored both ways


def test_run_recommendation_fail_when_oracle_lt_baseline(monkeypatch):
    def fake_score(cfg, resized, adapter, limit, *, input_field="instruction", prep_row=None):
        acc = 0.40 if input_field == "attribute_spec_text" else 0.50
        return {"metric": acc, "overall_ci_low": None, "overall_ci_high": None,
                "scored_rows": 120, "scored_units": 120}

    monkeypatch.setattr(oracle_gate, "score", fake_score)
    rep = oracle_gate.run(cfg=object(), resized_model="m", adapter="a", limit=0)
    assert rep["recommendation"] == "FAIL"
    assert rep["delta"] < 0
