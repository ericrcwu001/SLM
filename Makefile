PY ?= python3
ARTIFACT_ROOT ?= .
DURABLE_ROOT ?=
LOCAL_ROOT ?= /content/slm

.PHONY: help install fixtures test smoke cli-demo acquire data data-offline pack stage push clean clean-data \
	materialize-tokens pair-images vocab-resize sft-smoke

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
	@echo "  pack          slm_stage: corpus -> tar shards in DURABLE_ROOT (Drive/local/gs://)"
	@echo "  stage         slm_stage: shards -> LOCAL_ROOT (verified, resumable); set SLM_ARTIFACT_ROOT"
	@echo "  push          slm_stage: local checkpoints/outputs -> DURABLE_ROOT"
	@echo "  clean         remove eval_runs + generated eval fixtures"
	@echo "  clean-data    remove acquired/derived data-gen artifacts (luts/, data/*)"
	@echo "  materialize-tokens  encode canonical residuals -> 64 target_tokens (frozen tokenizer)"
	@echo "  pair-images   attach leakage-safe generic input images to LUT-only supported rows"
	@echo "  vocab-resize  Stage 3: add 259 LUT tokens to Qwen2.5-VL + embedding preflight"
	@echo "  sft-smoke     Stage 5: 50/200-row QLoRA overfit smoke run (needs sft extra + GPU)"

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

pack:
	$(PY) -m data_pipeline.staging.run_staging pack \
		--config configs/staging_default.yaml --root $(ARTIFACT_ROOT) --durable-root $(DURABLE_ROOT)

stage:
	$(PY) -m data_pipeline.staging.run_staging stage \
		--config configs/staging_default.yaml --durable-root $(DURABLE_ROOT) --local-root $(LOCAL_ROOT)

push:
	$(PY) -m data_pipeline.staging.run_staging push \
		--config configs/staging_default.yaml --durable-root $(DURABLE_ROOT) --local-root $(LOCAL_ROOT)

materialize-tokens:
	$(PY) -m scripts.materialize_target_tokens

pair-images:
	$(PY) -m scripts.pair_generic_images

vocab-resize:
	$(PY) -m sft.vocab_resize --config configs/sft_default.yaml --out models/base_resized

sft-smoke:
	$(PY) -m sft.train --config configs/sft_default.yaml --resized-model models/base_resized --smoke-size 50

clean:
	rm -rf eval_runs
	rm -rf data/eval/images data/eval/smoke_rows.jsonl data/eval/mock_outputs.jsonl

clean-data:
	rm -rf luts data/raw_registry data/splits data/active_sft data/warmup data/run_summary.json
