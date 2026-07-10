# Bilevel objective — external metric injection (NOT a cognitive recipe)

`spec.json` declares `objective.kind: "recipe"` **only** so `init_run.py` accepts an injected
baseline `--metric` instead of trying to run a shell command locally (a `command` objective would
execute on the Mac/Codex host, which cannot reach the remote Colab A100). **The recipe judges are
never invoked** — this loop is driven by the plain engine scripts under Codex control, and every
candidate's metric is measured on Colab and injected via
`run_iteration.py --pre-shaped --metric <value>`.

## What the metric is
Held-out, teacher-forced **LUT-code token accuracy** of the trained QLoRA adapter, computed by
`python -m sft.score_tokens` (decoder-free; the frozen VQ decoder stays disabled). Direction = **max**.

## How one candidate is evaluated (the actuator loop)
1. Codex proposes a candidate (`params` over the `param_space` in `spec.json`).
2. Codex writes the candidate into the Colab notebook's config cell (or `/content/SLM/candidate.json`).
3. Computer Use runs the eval cell, which calls
   `python -m sft.bilevel_bridge --mode colab --config candidate.json` on the A100. The bridge
   trains, scores the holdout, and prints one `METRIC=<accuracy>` line.
4. That line lands in the **local** `.ipynb`; Codex reads it and calls
   `run_iteration.py --run-dir <dir> --proposal <proposal.json> --pre-shaped --metric <accuracy>`.
5. Codex reads `status.py`; stop when `t>=T` or `no_improve>=patience`, else loop.

## Invariants (do not violate)
- Only the `param_space` knobs are varied; the batch triple, `epochs=2`, `num_new_tokens=259`,
  the quant scheme, `max_seq_len`, and `base_model_id` are **locked by omission** (a bad value would
  trip `SFTConfig.__post_init__`; the bridge pre-validates and rejects such candidates).
- `B=1` always (one A100). The frozen tokenizer is immutable; the corpus is read-only.
