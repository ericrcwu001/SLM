# Doc 01 — Measure `oracle@N` (the gate)

**Prereq:** read [`README.md`](README.md). This is the first thing to build and run; its result
decides whether Docs 02/03 are worth doing.

## Goal

Quantify whether **sampling ever covers a high-fidelity trajectory**. For each holdout row, draw N
samples, score each by behavioral fidelity, and report:
- **`oracle@k`** = mean over rows of `max` fidelity among the first `k` samples (the ceiling a
  perfect reranker could reach), for `k ∈ {1,4,8,16,32,64}`.
- **`best_of_N`** = mean over rows of the fidelity of the sample the **reranker** actually picks
  (primary `behavioral_fidelity`, tie-breaks per README) — the realistic deploy number.

Compare against the free-running greedy baseline (**0.159**) and the metric ceiling (**~0.89**).

**Gate:** `oracle@N ≳ 0.6` at feasible N → coverage is good → do Docs 02 + 03. `oracle@N` stalls
≲ 0.3 → coverage gap → reranking/distillation are capped; escalate to RL (out of scope here). Report
the full `oracle@k` curve so the shape (not just one number) informs the call.

## Files (Doc 01 also builds the two shared helpers Docs 02/03 import)

- **New:** `sft/generate.py` → add `generate_codes_batch(...)` + `generate_codes_for_row_batch(...)`
  (batched sampling; reused by Docs 02/03).
- **New:** `sft/loader.py` → `load_eval_model(cfg, resized_model, adapter) -> (model, processor)`,
  extracted verbatim from the loader block in `sft/score_tokens.score` (processor + `BitsAndBytesConfig`
  + `PeftModel.from_pretrained`). Update `sft/score_tokens.py` to import it; keep `tests/test_score_tokens.py`
  green (that file tests pure `summarize_scores`, so the refactor should not touch it — verify).
- **New:** `eval/behavioral_fidelity.py` → add `rerank_key(rec)` (the canonical reranker from the
  README; ΔE tie-break used only when `decoded_delta_e` is present). Docs 01 and 02 import it.
- **New:** `eval/oracle_at_n.py` (module + `main()` CLI).
- **New:** `tests/test_oracle_at_n.py` (pure-logic tests, no GPU).

## Step 1 — batched generation helper (`sft/generate.py`)

Single-sample generation is too slow for N×rows. Add a batched variant using
`num_return_sequences`. `make_prefix_fn` already ignores `batch_id` and slices on a fixed
`prompt_len`, so it is correct under expansion.

```python
def generate_codes_batch(model, processor, *, image, text, n: int, sampling: dict, chunk: int = 16,
                         max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, device=None) -> list[list[int] | None]:
    """Free-running generate N samples for one (image,text) prompt, in chunks of `chunk`.

    Returns a list of length n; each element is 64 codebook indices, or None for a refusal
    (<unsupported>). `sampling` MUST enable sampling (e.g. {"temperature":0.7,"top_p":0.9}); greedy
    with n>1 would return n identical rows. `chunk` bounds peak memory: ceil(n/chunk) .generate calls,
    each with num_return_sequences<=chunk.
    """
    import torch
    from qwen_vl_utils import process_vision_info
    tok = processor.tokenizer
    ids = SpecialIds(tok)
    user = {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": text}]}
    prompt_text = processor.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info([user])
    inp = processor(text=[prompt_text], images=image_inputs, videos=video_inputs, return_tensors="pt")
    dev = device if device is not None else getattr(model, "device", None)
    if dev is not None:
        inp = inp.to(dev)
    plen = inp["input_ids"].shape[1]
    prefix_fn = make_prefix_fn(plen, ids)
    results, remaining = [], n
    with torch.no_grad():
        while remaining > 0:
            k = min(chunk, remaining)
            out = model.generate(**inp, do_sample=True, num_return_sequences=k,
                                 prefix_allowed_tokens_fn=prefix_fn,
                                 max_new_tokens=max_new_tokens, **sampling)
            results.extend(codes_from_output(out[i], plen, ids) for i in range(out.shape[0]))
            remaining -= k
    return results
```

Notes / pitfalls:
- `codes_from_output` already maps ids→codebook indices and returns `None` on `<unsupported>`.
- **Memory / chunking (implement, don't leave as a choice):** `num_return_sequences=n` multiplies the
  decode batch by `n`. Give `generate_codes_batch` a `chunk: int = 16` parameter and internally loop
  `ceil(n/chunk)` `.generate` calls (each with `num_return_sequences=min(chunk, remaining)`),
  concatenating results — so any `n` works within a fixed memory ceiling. `run()` (Step 2) passes
  `chunk` through. This makes "feasible N" a function of `chunk`, not of a single-call OOM.
- **`generate_codes_for_row_batch(model, processor, row, *, input_field, bucketize=False, n, sampling,
  chunk=16, device=None)`** mirrors `generate_codes_for_row`: `text = input_text_for(row, input_field,
  bucketize=bucketize)` (the CONDITIONING — matches training), `image = resolve_image(row["image_path"])`,
  then `generate_codes_batch`. (Scoring uses the canonical spec separately — see Step 2.)

## Step 2 — `eval/oracle_at_n.py`

Pure orchestration over the existing scorers. Skeleton:

```python
"""Measure oracle@N / best-of-N behavioral fidelity for an adapter (the coverage gate; docs/collapse_fix)."""
import numpy as np
from data_pipeline.attribute_spec import ground_truth_attribute_spec_text
from eval.behavioral_fidelity import rerank_key, score_generation   # rerank_key added in Doc 01 Files
from sft.example import load_rows, supported_rows

DEFAULT_KS = (1, 4, 8, 16, 32, 64)

def score_row_samples(codes_list, spec_text, target_codes):
    """Score each of the N sampled code lists; None (refusal) -> fidelity 0.0 record."""
    recs = []
    for codes in codes_list:
        if codes is None or len(codes) != 64:
            recs.append({"behavioral_fidelity": 0.0, "collapsed": True, "refused": codes is None})
        else:
            recs.append(score_generation(codes, spec_text, target_codes=target_codes))
    return recs

def oracle_and_best(recs_by_row, ks=DEFAULT_KS):
    """recs_by_row: list (per row) of lists (per sample) of records. Returns the curve + best-of-N."""
    out = {"rows": len(recs_by_row)}
    for k in ks:
        oracle = [max((r.get("behavioral_fidelity") or 0.0) for r in recs[:k]) for recs in recs_by_row if recs]
        out[f"oracle@{k}"] = float(np.mean(oracle)) if oracle else None
    best = []
    for recs in recs_by_row:
        if not recs:
            continue
        pick = max(recs, key=rerank_key)
        best.append(pick.get("behavioral_fidelity") or 0.0)
    out["best_of_N"] = float(np.mean(best)) if best else None
    return out

def run(model, processor, cfg, *, n=32, temperature=0.7, top_p=0.9, limit=32, chunk=16,
        input_field="attribute_spec_text"):
    rows = supported_rows(load_rows(cfg.active_rows_path), holdout=True)
    if limit:
        rows = rows[:limit]
    from sft.generate import generate_codes_for_row_batch  # add in Step 1
    recs_by_row = []
    for row in rows:
        spec = ground_truth_attribute_spec_text(row)            # canonical (SCORING)
        codes_list = generate_codes_for_row_batch(model, processor, row, input_field=input_field,
                                                  n=n, chunk=chunk,
                                                  sampling={"temperature": temperature, "top_p": top_p},
                                                  device=model.device)                     # CONDITIONING
        recs_by_row.append(score_row_samples(codes_list, spec, row.get("target_tokens")))
    return oracle_and_best(recs_by_row)
```

`main()` mirrors `sft/score_tokens.py:main`: `--config` (default `configs/candidate_two_stage.json`),
`--resized-model models/base_resized`, `--adapter`, `--limit`, `--n`, `--temperature`, `--top-p`,
`--chunk`. Load the model via **`sft.loader.load_eval_model`** (the shared helper built in this doc's
Files list — do not copy the block). Print one JSON line `{"oracle_summary": {...}}` plus a human table.
Run **both** `temperature ∈ {0.7, 1.0}` (two `run` calls) so we see whether a hotter temperature
improves coverage. Note the conditioning/scoring split already present in `run`: generation text comes
from `generate_codes_for_row_batch` (→ `input_text_for`, matches training); scoring uses the CANONICAL
`ground_truth_attribute_spec_text(row)`. Today these strings are identical, but keep them separate.

## Step 3 — tests (`tests/test_oracle_at_n.py`, no GPU)

Test the pure aggregation with synthetic records (do not load a model):
- `oracle_and_best` on hand-built `recs_by_row`: e.g. row with samples `[0.1, 0.6, 0.2]` → `oracle@4 = 0.6`,
  and `best_of_N` picks the 0.6 sample; a monotonic `oracle@k` curve (non-decreasing in k).
- `rerank_key` (from `eval.behavioral_fidelity`) prefers higher fidelity, breaks ties toward
  not-collapsed then higher entropy, then lower ΔE when present.
- `score_row_samples` maps `None`/short code lists to a `0.0`/`collapsed` record.

## Cost & runtime

- Default probe: `limit=32` rows × `n=32` samples × 2 temps = 2048 sampled 64-token trajectories.
  Batched (`num_return_sequences=32`), that is 32 rows × 2 temps = 64 `.generate` calls → tens of
  minutes on an A100. Decode+score is CPU and cheap (~0.1 s/sample).
- Scale `n`/`limit` up once the shape looks promising. Watch GPU memory at high `n` (chunk if needed).
- Reuses `models/base_resized` (build via `sft.vocab_resize`) + the P6 adapter from HF, exactly like
  `notebooks/phase1_behavioral_score.ipynb` CELLs 1–2.

## Deliverable

`{"oracle_summary": {"oracle@1":…, "oracle@8":…, "oracle@32":…, "best_of_N":…}}` at t=0.7 and t=1.0,
plus the **gate call written down** using the three-band rule in the README ("Gate thresholds"):
good (≥0.6) / capped (0.3–0.6) / RL (≤0.3). A `notebooks/phase2_oracle_at_n.ipynb` (clone of
`phase1_behavioral_score.ipynb`'s CELLs 1–2 then `python -m eval.oracle_at_n …`) is a nice-to-have.

Note: the existing `notebooks/oracle_gate_run.ipynb` is **unrelated** to this doc (it is the older
spec→codes upper-bound gate) — do not mistake it for the deliverable.
