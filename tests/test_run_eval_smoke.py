"""End-to-end smoke test: fixtures -> run_eval -> report file set."""

import csv
import os

from eval import run_eval
from eval.fixtures import make_smoke_rows


def _read_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_run_eval_end_to_end(tmp_path):
    data_dir = tmp_path / "data_eval"
    data_dir.mkdir()
    rows_path, mock_path = make_smoke_rows.generate(str(data_dir))

    out_root = str(tmp_path / "eval_runs")
    run_dir = run_eval.run(
        config_path=None,
        rows_path=rows_path,
        out_root=out_root,
        mock_outputs_path=mock_path,
        seeds=[1234, 1235],
        modes=["free_generation", "runtime_constrained"],
        run_id="t",
    )

    # required artifacts exist
    for name in ("overall_results.csv", "unsupported_results.csv", "gate_results.csv",
                 "baseline_delta.csv", "seed_summary.csv", "config.yaml", "rows.jsonl",
                 "raw_model_outputs.jsonl", "parsed_outputs.jsonl", "failure_manifest.jsonl",
                 "target_fidelity_results.csv", "safety_results.csv", "style_results.csv"):
        assert os.path.exists(os.path.join(run_dir, name)), name
    # metrics_by_row parquet (or csv fallback)
    assert (os.path.exists(os.path.join(run_dir, "metrics_by_row.parquet"))
            or os.path.exists(os.path.join(run_dir, "metrics_by_row.csv")))

    overall = _read_csv(os.path.join(run_dir, "overall_results.csv"))
    mock_rows = [r for r in overall if r["model"] == "mock_replay"]
    assert mock_rows, "mock_replay adapter should have run"

    # runtime-constrained mode must be 100% syntax-valid
    constrained = [r for r in mock_rows if r["mode"] == "runtime_constrained"]
    assert constrained
    for r in constrained:
        assert float(r["constrained_syntax_valid_rate"]) == 1.0

    # free-generation valid-token rate is < 1.0 (fixtures inject invalid outputs)
    free = [r for r in mock_rows if r["mode"] == "free_generation"]
    assert 0.0 < float(free[0]["free_generation_valid_token_rate"]) < 1.0

    # supported pass rate is not evaluated (decoder disabled)
    assert all("decoder_disabled" in r["supported_pass_status"] for r in mock_rows)


def test_gates_and_disabled_tables(tmp_path):
    data_dir = tmp_path / "d"
    data_dir.mkdir()
    rows_path, mock_path = make_smoke_rows.generate(str(data_dir))
    run_dir = run_eval.run(None, rows_path, str(tmp_path / "runs"), mock_path,
                           seeds=[1234], modes=["free_generation"], run_id="g")

    gates = _read_csv(os.path.join(run_dir, "gate_results.csv"))
    by_metric = {g["metric"]: g for g in gates}
    # smoke N is far below min_N -> not evaluable
    assert by_metric["unsupported_recall"]["status"] == "not-evaluable-below-min_N"
    # supported pass rate blocked by decoder
    assert "decoder_disabled" in by_metric["supported_prompt_to_lut_pass_rate"]["status"]

    # disabled color tables carry the decoder_disabled note
    tf = _read_csv(os.path.join(run_dir, "target_fidelity_results.csv"))
    assert tf and "decoder_disabled" in tf[0]["status"]
