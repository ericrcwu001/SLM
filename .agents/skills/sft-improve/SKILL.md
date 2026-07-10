---
name: sft-improve
description: >-
  Greedy, budget-capped SFT improvement loop on a Colab A100, driven by Codex + Computer Use. Trains
  the LUT-SLM QLoRA adapter up to N times (default 5), changing ONE hyperparameter per run to beat the
  best held-out token accuracy so far, keeping the best. No bilevel engine — Codex reasons about the
  next config from a small on-disk ledger. Use when the user wants to "train SFT a few times and
  improve each run", hill-climb SFT hyperparameters, or run a short manual sweep on Colab. NOT for
  editing the frozen tokenizer/decoder or the corpus (immutable / read-only).
---

# SFT greedy improve loop (Codex + Computer Use, ≤N runs)

Same eval unit as the notebook — `sft.bilevel_bridge` trains + scores **held-out token accuracy** and
prints one `METRIC=` line into the LOCAL `.ipynb`. You (Codex) keep a small ledger, change ONE knob
per run, keep the best, stop after **N (default 5)**. No `spec.json` / engine scripts. Read
`AGENTS.md` for paths, invariants, and the case trap.

## Preconditions (verify, don't assume)
- A100 connected in Cursor (**Auto Connect**); `notebooks/sft_stage7_run.ipynb` open; Computer Use on.
- `models/base_resized` built once (`vocab_resize`) and **Gate 0** passed. If not, run the notebook
  through Gate 0 first and STOP if it fails (e.g. `[sft][skip]` ≈ row count = the `/content/slm`
  case trap).
- **HF upload set up:** run the notebook's "HF UPLOAD SETUP" cell so `PUSH_HF_REPO` + `HF_WRITE_TOKEN`
  are in the env (read-only `HF_TOKEN` 403s on upload). Every eval then saves its adapter locally
  (`models/sft_adapters/<id>_smoke<N>`) AND uploads it to `PUSH_HF_REPO`.

## Objective + knobs
- MAXIMIZE the bridge's `METRIC=` (held-out token accuracy; deterministic argmax).
- Change ONLY these invariant-safe knobs, **one per run**:
  `learning_rate_lora` 5e-5..5e-4 (dominant), `lora_r` {8,16,24,32}, `lora_alpha` {16,32,64},
  `lora_dropout` 0..0.1, `warmup_ratio` 0..0.1, `max_grad_norm` 0.3..2.0,
  `max_pixels` {50176,100352,200704,401408}.
- NEVER touch `epochs`, the batch triple, `num_new_tokens`, `base_model_id`, quant scheme,
  `max_seq_len` (locked; the bridge rejects violations). **B=1** always.

## Ledger (state survives context/compaction)
`models/sft_adapters/improve_ledger.json`. Read it at the start of each step; append after each run:
`python .agents/skills/sft-improve/scripts/ledger.py append --ledger <L> --iter <i> --params '<json>'
--metric <float|nan> --adapter <path> --note "<hypothesis/result>"` — it prints the current best.

## Loop (repeat until iter == N or the user stops)
1. **Research the next change — launch a subagent.** Before every run (iter > 1), spawn ONE subagent
   whose task is: read the full ledger (all prior `{params, metric, note}`); reason about which SINGLE
   invariant-safe knob to change and BY HOW MUCH to most likely raise held-out token accuracy; it may
   WebSearch LoRA/QLoRA tuning guidance (rank/alpha/LR interplay, warmup, vision-token budget). It
   must return STRICT JSON:
   `{"knob": "<name>", "from": <current>, "to": <new>, "rationale": "<one line>"}`.
   - iter 1: skip the subagent; use the baseline config (the `configs/sft_default.yaml` values).
   - VALIDATE the subagent's answer: `knob` is in the allowed set, `to` is in range, exactly ONE knob
     changes off the BEST-so-far config, and it never touches a locked knob (batch triple / epochs /
     vocab / dtype). If invalid, re-ask once, else fall back to the greedy default order
     (`learning_rate_lora` → `lora_r` → `warmup_ratio` → `max_pixels`; push a helping knob further,
     revert + switch on a regression). Apply the change to BEST → the candidate; keep the rationale as
     the ledger note.
2. **Write it:** `python .agents/skills/bilevel-colab-loop/scripts/write_ipynb_config.py
   --notebook notebooks/sft_stage7_run.ipynb --params '<json>'`
3. Record the notebook mtime, then **Computer Use → "Run All"** (config cell writes
   `/content/SLM/candidate.json`; the eval cell runs `sft.bilevel_bridge` → `METRIC=`). Wait until the
   eval cell finishes AND the file is saved (autosave / Save).
4. **Read it:** `python .agents/skills/bilevel-colab-loop/scripts/read_ipynb_metric.py
   --notebook notebooks/sft_stage7_run.ipynb --min-mtime <recorded>`.
   - `status ok` → `ledger.py append` with the metric + adapter; update best.
   - `status failed/no_metric/stale` → `ledger.py append` with `--metric nan` + the reason; do NOT
     count it as an improvement; next run reverts to best and tries a different knob (retry a failed
     knob at most once).
5. **Save + upload the model.** The bridge saves the adapter locally and uploads it to `PUSH_HF_REPO`
   (a per-config subfolder) with `HF_WRITE_TOKEN`. Confirm `hf_push.pushed == true` in the run's
   `bridge_summary` (surfaced by `read_ipynb_metric.py`). If `hf_push` shows an error (missing write
   token / 403), **STOP and tell the user** — do not silently lose models. Track the HF path of the
   best adapter for the final report.

## Stop + report
After N runs (or on user stop): `ledger.py show` → present the table (iter, knob changed, metric,
Δ vs best), name the winning config + its adapter path. Be honest: with ≤N runs and CUDA
nondeterminism, small deltas can be within run-to-run noise — treat only clear gains as real, and
suggest a 2× repeat of the winner if they want to confirm.
