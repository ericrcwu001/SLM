PY ?= python3
ARTIFACT_ROOT ?= .

.PHONY: help install fixtures test smoke cli-demo acquire data data-offline clean clean-data

help:
	@echo "Targets:"
	@echo "  install       editable install with dev + data extras"
	@echo "  fixtures      generate 50/20 smoke eval rows + synthetic images + mock outputs"
	@echo "  test          run the pytest suite (fully offline)"
	@echo "  smoke         generate fixtures then run_eval across decoder-free baselines"
	@echo "  cli-demo      run the prompt_to_lut CLI on one supported + one unsupported prompt"
	@echo "  acquire       Stage 2: autonomously download bounded real corpora (network)"
	@echo "  data          full data-gen pipeline over acquired + procedural LUTs (network)"
	@echo "  data-offline  full data-gen pipeline over procedural LUTs only (no network)"
	@echo "  clean         remove eval_runs + generated eval fixtures"
	@echo "  clean-data    remove acquired/derived data-gen artifacts (luts/, data/*)"

install:
	$(PY) -m pip install -e ".[dev,data]"

fixtures:
	$(PY) -m eval.fixtures.make_smoke_rows --out data/eval

test:
	$(PY) -m pytest

smoke: fixtures
	$(PY) -m eval.run_eval \
		--config eval/configs/eval_default.yaml \
		--rows data/eval/smoke_rows.jsonl \
		--mock-outputs data/eval/mock_outputs.jsonl \
		--out eval_runs

cli-demo: fixtures
	$(PY) -m cli.prompt_to_lut --self-check || true
	$(PY) -m cli.prompt_to_lut --image data/eval/images/eval_unsup_000001.png \
		--prompt "make only the sky bluer" --out eval_runs/cli_demo_unsupported
	$(PY) -m cli.prompt_to_lut --image data/eval/images/eval_sup_000001.png \
		--prompt "make it warmer" --out eval_runs/cli_demo_supported \
		--mock-output "<lut_bos> $$(for i in $$(seq 1 64); do printf '<lut_042> '; done)<lut_eos>"

acquire:
	$(PY) -m data_pipeline.acquire.run_acquire \
		--config data_pipeline/configs/pipeline_default.yaml --out $(ARTIFACT_ROOT)

data:
	$(PY) -m data_pipeline.run_pipeline \
		--config data_pipeline/configs/pipeline_default.yaml --out $(ARTIFACT_ROOT)

data-offline:
	$(PY) -m data_pipeline.run_pipeline \
		--config data_pipeline/configs/pipeline_default.yaml --out $(ARTIFACT_ROOT) \
		--sources procedural_fillers_v1

clean:
	rm -rf eval_runs
	rm -rf data/eval/images data/eval/smoke_rows.jsonl data/eval/mock_outputs.jsonl

clean-data:
	rm -rf luts data/raw_registry data/splits data/active_sft data/warmup data/run_summary.json
