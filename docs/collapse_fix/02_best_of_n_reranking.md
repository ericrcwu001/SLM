# Doc 02 — Best-of-N reranking (inference; no retrain)

**Prereq:** [`README.md`](README.md) + [`01_oracle_at_n.md`](01_oracle_at_n.md) (this depends on
`generate_codes_batch` from Doc 01, and is only worth shipping if Doc 01's `oracle@N` gate passed).

## Goal

A deployable inference path that samples N candidate code sequences, reranks them by behavioral
fidelity (the true objective — no target LUT needed), and returns the best. This turns the model's
"good trajectory exists but greedy misses it" behavior into a usable generator **today**, with no
training. It is also the exact primitive Doc 03 harvests from.

Success: on the holdout, best-of-N behavioral fidelity **exceeds free-running greedy 0.159** and
approaches Doc 01's `best_of_N` number (they should match — this is that measurement, deployed).

## Files

- **New:** `eval/best_of_n.py` — `best_of_n_codes(...)` (core) + `main()` demo CLI.
- **New:** `tests/test_best_of_n.py` — pure reranking-logic tests (no GPU).
- **Optional wiring:** a cell/section in `notebooks/colab_lut_slm_inference.ipynb` that calls it.

## Step 1 — core reranker (`eval/best_of_n.py`)

```python
"""Best-of-N generation reranked by behavioral fidelity (inference; docs/collapse_fix/02)."""
from data_pipeline.attribute_spec import ground_truth_attribute_spec_text
from eval.behavioral_fidelity import rerank_key, score_generation   # rerank_key added in Doc 01

def best_of_n_codes(model, processor, *, image, cond_text, spec_text=None, n=16,
                    sampling=None, device=None):
    """Return (best_codes, best_record). Generate n samples CONDITIONED on `cond_text`, score each
    against `spec_text` (the REQUESTED spec; defaults to cond_text — identical at deploy), and return
    the reranker-best VALID candidate. No target LUT needed — fidelity is agreement with the request.
    Keeping cond_text and spec_text separate honors the README rule (conditioning must match training;
    scoring uses the canonical spec) even if a future re-migration makes them diverge."""
    from sft.generate import generate_codes_batch
    spec_text = spec_text or cond_text
    sampling = sampling or {"temperature": 0.7, "top_p": 0.9}
    cand = generate_codes_batch(model, processor, image=image, text=cond_text, n=n,
                                sampling=sampling, device=device)
    scored = []
    for codes in cand:
        if codes is None or len(codes) != 64:      # refusal / malformed
            continue
        scored.append((codes, score_generation(codes, spec_text)))   # no target_codes at deploy
    if not scored:
        return None, {"behavioral_fidelity": None, "refused_all": True}
    return max(scored, key=lambda t: rerank_key(t[1]))

def best_of_n_for_row(model, processor, row, *, n=16, sampling=None,
                      input_field="attribute_spec_text", device=None):
    """Row convenience: CONDITION via input_text_for (matches training), SCORE via canonical spec."""
    from sft.example import input_text_for, resolve_image
    return best_of_n_codes(model, processor, image=resolve_image(row["image_path"]),
                           cond_text=input_text_for(row, input_field),          # conditioning
                           spec_text=ground_truth_attribute_spec_text(row),      # scoring (canonical)
                           n=n, sampling=sampling, device=device)
```

Design decisions (document these inline):
- **Reranker = the canonical `rerank_key` from `eval.behavioral_fidelity`** (imported, not re-spelled).
  It uses `behavioral_fidelity` + tie-breaks and **no `target_codes`** (there is no target LUT at
  deploy); the ΔE tie-break is skipped when the key is absent, so the deploy pick never depends on it.
- **Conditioning vs scoring are separate args** (`cond_text` vs `spec_text`). At deploy they are the
  same requested spec; for corpus rows, condition via `input_text_for` (training parity) and score via
  `ground_truth_attribute_spec_text`. They are byte-identical today but must not be conflated.
- **All-refused fallback:** if every sample is `<unsupported>`/malformed, return `(None, {...})`; the
  caller decides whether to emit a refusal or retry hotter. Do not fabricate codes.
- **Sampling default `t=0.7, top_p=0.9`** but expose it — use the temperature Doc 01 found best.

## Step 2 — evaluation harness

Add a `main()` that, for a `--limit` slice of the holdout, runs `best_of_n_for_row` and prints
`summarize_fidelity` over the picked records (reuse `eval.behavioral_fidelity.summarize_fidelity`).
Load the model via `sft.loader.load_eval_model` (Doc 01).

**To make the number comparable to greedy 0.159 and Doc 01's `best_of_N`, the harness MUST:**
- use the **same `n` and the same holdout `--limit` slice** as the Doc 01 run you compare against
  (best-of-N is monotonic in `n`, so mismatched `n` is not comparable);
- **fold an all-refused/malformed pick in as a `{"behavioral_fidelity": 0.0, "collapsed": True}`
  record before `summarize_fidelity`** — exactly what the shipped `_run_behavioral` baseline (which
  produced 0.159) does with refusals. Otherwise `summarize_fidelity` drops `None`-fidelity rows and
  you divide by survivors, inflating the mean vs the baseline. (README notes `refused=0` under greedy,
  so this rarely fires — but keep the accounting identical.)

Alternatively (cheaper to wire): add `--behavioral-sampling bestofn` and `--behavioral-n` to
`sft/score_tokens.py`, routing the behavioral pass through `best_of_n_for_row`. If you do this, keep the
existing `greedy|sample|both` modes untouched and keep the teacher-forced `METRIC=` sentinel unchanged
(it is a locked contract — see `sft/score_tokens.py` docstring). Prefer the standalone `eval/best_of_n.py`
first; fold into `score_tokens` only if convenient.

## Step 3 — tests (`tests/test_best_of_n.py`, no GPU)

- `rerank_key` (imported) ordering: higher fidelity wins; equal fidelity → not-collapsed wins; then
  higher entropy; then lower ΔE when present. (Test lives with `rerank_key` — see Doc 01/behavioral_fidelity.)
- A `best_of_n_codes` unit test with `generate_codes_batch` and `score_generation` monkeypatched to
  return fixed candidates/records: assert it returns the highest-fidelity valid candidate, skips
  `None`/short ones, and returns the all-refused fallback when every candidate is `None`.

## Step 4 — (optional) inference-notebook wiring

In `notebooks/colab_lut_slm_inference.ipynb`, after the model loads, add a cell that calls
`best_of_n_for_row`/`best_of_n_codes` and renders the chosen LUT (reuse that notebook's existing
decode+`apply_lut_trilinear` render path). This gives a visible "greedy vs best-of-16" side-by-side.

## Cost

N× the single-sample generation per request (batched into one `.generate`), plus N cheap CPU
decode+score. At deploy this is a latency/quality knob (bigger N → better, slower). Doc 03 removes the
recurring N× cost by distilling the winners into the weights.

## Deliverable

`eval/best_of_n.py` + tests; a holdout number showing best-of-N > 0.159, and — **at the same `n` and
holdout slice** — matching Doc 01's `best_of_N` (this harness IS that measurement, deployed). Shippable
on its own even if Doc 03 is deferred.
