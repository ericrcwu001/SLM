"""Unit tests for the bilevel/SFT glue that needs NO GPU or `sft` extra:
holdout split, the bridge's config merge/validation, and the notebook config-write / metric-read
helper scripts. These are the correctness-critical guards (silent-success trap, invariant rejection,
metric read-back) exercised without training.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sft import bilevel_bridge as bb
from sft.holdout import DEFAULT_HOLDOUT_FRAC, is_holdout

_SKILL = Path(__file__).resolve().parent.parent / "scripts"


def test_holdout_deterministic_and_fraction():
    ids = [f"row{i:06d}" for i in range(6000)]
    frac = sum(is_holdout(x) for x in ids) / len(ids)
    assert abs(frac - DEFAULT_HOLDOUT_FRAC) < 0.02
    assert is_holdout("abc") == is_holdout("abc")   # stable
    assert not is_holdout("")                        # empty id never holdout
    assert not is_holdout("abc", frac=0.0)
    assert is_holdout("abc", frac=1.0)


def test_bridge_merge_and_validate_ok(tmp_path):
    f = tmp_path / "cand.json"
    f.write_text(json.dumps({"lora_r": 24, "learning_rate_lora": 3e-4, "max_pixels": 100352}))
    merged = bb._merged_config("configs/sft_default.yaml", bb._candidate_params(str(f)))
    assert merged["lora_r"] == 24 and merged["learning_rate_lora"] == 3e-4
    bb._validate(merged)  # must not raise


def test_bridge_accepts_params_wrapper(tmp_path):
    f = tmp_path / "p.json"
    f.write_text(json.dumps({"params": {"lora_r": 8}, "hypothesis": "small"}))
    assert bb._candidate_params(str(f)) == {"lora_r": 8}


def test_bridge_rejects_unknown_knob():
    with pytest.raises(ValueError):
        bb._merged_config("configs/sft_default.yaml", {"not_a_field": 1})


def test_bridge_accepts_two_stage_input_field():
    # P6: the sanctioned input swap flows through the bridge (input_field is a real SFTConfig field,
    # not an unknown knob) and validates; the committed two-stage candidate is well-formed.
    cand = bb._candidate_params("configs/candidate_two_stage.json")
    merged = bb._merged_config("configs/sft_default.yaml", cand)
    assert merged["input_field"] == "attribute_spec_text"
    bb._validate(merged)  # must not raise
    # locked knobs are untouched by the two-stage candidate
    assert "epochs" not in cand and "max_seq_len" not in cand and "seed" not in cand


def test_bridge_rejects_bad_input_field():
    merged = bb._merged_config("configs/sft_default.yaml", {})
    merged["input_field"] = "something_else"
    with pytest.raises(ValueError):
        bb._validate(merged)


def test_bridge_rejects_batch_triple_violation():
    merged = bb._merged_config("configs/sft_default.yaml", {})
    merged["gradient_accumulation_steps"] = 5  # 1*5 != effective_batch_size (32) -> __post_init__ raises
    with pytest.raises(ValueError):
        bb._validate(merged)


def _run(script: str, *args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(_SKILL / script), *args],
                          capture_output=True, text=True)


def test_write_ipynb_config_targets_tagged_cell(tmp_path):
    nb = {"cells": [{"cell_type": "code", "id": "cand",
                     "metadata": {"tags": ["candidate-config"]},
                     "execution_count": None, "outputs": [], "source": "# @candidate-config\n"}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    p = tmp_path / "nb.ipynb"
    p.write_text(json.dumps(nb))
    r = _run("write_ipynb_config.py", "--notebook", str(p), "--params", json.dumps({"lora_r": 24}))
    assert r.returncode == 0, r.stdout + r.stderr
    doc = json.loads(p.read_text())
    src = doc["cells"][0]["source"]
    src = "".join(src) if isinstance(src, list) else src
    assert "candidate.json" in src and "24" in src and doc["cells"][0]["outputs"] == []


def _nb_with_stdout(tmp_path, text: str) -> Path:
    nb = {"cells": [{"cell_type": "code", "id": "e", "metadata": {}, "execution_count": 1,
                     "outputs": [{"output_type": "stream", "name": "stdout", "text": text}]}],
          "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
    p = tmp_path / "o.ipynb"
    p.write_text(json.dumps(nb))
    return p


def test_read_metric_ok(tmp_path):
    nb = _nb_with_stdout(tmp_path, '{"bridge_summary": {"metric": 0.42}}\nMETRIC=0.420000\n')
    out = json.loads(_run("read_ipynb_metric.py", "--notebook", str(nb)).stdout)
    assert out["status"] == "ok" and abs(out["metric"] - 0.42) < 1e-6


def test_read_metric_failed_on_abort(tmp_path):
    nb = _nb_with_stdout(tmp_path, "[bridge][ABORT] training did nothing\n")
    out = json.loads(_run("read_ipynb_metric.py", "--notebook", str(nb)).stdout)
    assert out["metric"] is None and out["status"] in ("failed", "no_metric")


def test_read_metric_none_when_absent(tmp_path):
    nb = _nb_with_stdout(tmp_path, "just some logs, no sentinel\n")
    out = json.loads(_run("read_ipynb_metric.py", "--notebook", str(nb)).stdout)
    assert out["status"] == "no_metric" and out["metric"] is None
