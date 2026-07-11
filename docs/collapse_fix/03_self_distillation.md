# Doc 03 — Self-distillation (training; bake best-of-N into the weights)

**Prereq:** [`README.md`](README.md), [`01`](01_oracle_at_n.md), [`02`](02_best_of_n_reranking.md).
Only do this if Doc 01's `oracle@N` gate passed (distillation is bounded by what best-of-N can already
produce — garbage in, garbage out).

## Goal

Amortize best-of-N into the model. For each **training** row, harvest the best-of-N winning codes and
write a **distilled corpus** whose `target_tokens` are those winners; then run the normal QLoRA SFT
(fresh adapter, locked knobs, 2 epochs) on it. This is rejection-sampling fine-tuning / expert
iteration (ReST/RFT/BOND): a stable MLE objective (not RL, no biased gradient) that moves the model's
**own free-running distribution** toward high-fidelity trajectories.

Success: the retrained adapter's **free-running greedy** behavioral fidelity on the (untouched)
holdout beats the **0.159** baseline. Iterate 1–2 rounds if it helps.

## Critical invariants (get these wrong and the experiment is meaningless)

1. **Rewrite TRAINING rows only.** Holdout rows (`is_holdout_row(row) == True`, keyed on
   `split_unit_id`) MUST be copied **unchanged**, so the behavioral eval still measures generalization
   against original targets. Never run the model on, or rewrite, a holdout row.
2. **Select REACHABLE high-fidelity targets by an ABSOLUTE bar — do NOT compare to the gold codes.**
   This is the crux, and the intuitive "keep the winner only if it beats the original" rule is
   **wrong and a no-op**: the row's original `target_tokens` are the VQ encoding of the *true* LUT and
   score ~0.89 (near-tautological), while any best-of-N winner is bounded by `oracle@N` (~0.6 at best),
   so winner-beats-gold almost never fires → nothing is replaced → the retrain ≈ P6 → no fix. It is
   also backwards: the whole point is to replace *unreachable* gold targets with *reachable* good
   trajectories the model can actually produce. Correct rule (ReST/RFT/expert-iteration):
   **keep the best-of-N winner as the new `target_tokens` iff its `behavioral_fidelity ≥ τ`** (absolute
   bar). If no sample clears τ, keep the original gold row (safe, keeps the corpus full; logged).
   - **τ default ≈ 0.5, provisional — set it from Doc 01.** τ must sit well above the current
     free-running fidelity (0.16) yet low enough that a meaningful fraction of rows qualify. Read the
     per-row `oracle@N` distribution from Doc 01 and pick τ so ≈30–60% of rows clear it; report the
     **replaced fraction** (it should match that estimate) *before* the overnight harvest so the run
     is falsifiable.
   - **Alternative (more principled, +1 generation/row): beat the row's OWN greedy.** Compute the
     row's greedy free-running fidelity and keep the winner iff `winner_fid > greedy_fid` (guarantees
     each replaced target is strictly better than what the model does today). Use this if the flat τ
     bar replaces too few/too many.
   - Rows where nothing clears the bar keep gold (an unreachable target — mild exposure-bias
     reinforcement for that slice). If keeping gold dilutes the effect, an alternative is to **drop**
     those training rows instead; try that only if keep-gold underperforms (it shrinks the train set).
3. **Preserve every other field.** Only `target_tokens`, `assistant_target`, and `token_status`
   change. Keep `id`, `image_path`, `instruction`, `measured_behavior`, `split_unit_id`,
   `source_family`, `is_supported`, `tokenizer_version`, `vq_codebook_sha256`, `vq_decoder_sha256`
   (the winners are valid frozen-codebook codes, so the identity fields still hold). NOTE:
   `attribute_spec_text` is **not** a stored field — the conditioning spec is derived on the fly from
   `measured_behavior` via `ground_truth_attribute_spec_text(row)`, so there is nothing to preserve
   there; just keep `measured_behavior`.
4. **Refuse/unsupported rows copied unchanged** (`is_supported == False`; `assistant_target` stays
   `<unsupported>`).
5. **`assistant_target` MUST be rebuilt with the exact materializer format:**
   `"<lut_bos> " + " ".join("<lut_%03d>" % c for c in codes) + " <lut_eos>"`. WARNING: importing
   `scripts.materialize_target_tokens._assistant_target` transitively pulls in `torch` (via
   `tokenizer.frozen`), which would make the "no-GPU" `distill_row` test require torch. To keep
   `distill_row` torch-free, **re-spell this one-liner as a private helper in the build script** and add
   a test asserting it equals the materializer's output (guard that single test with
   `pytest.importorskip("torch")`). This is the one sanctioned exception to "don't re-spell" — it
   avoids coupling a pure transform to the torch import graph while still pinning the format via an
   equality test.
6. **64-code guard (silent-drop risk).** `codes_from_output` can return **fewer than 64** codes on a
   short generation. The trainer/scorer require exactly 64 (`is_supported_materialized` + the exact-64
   guard in `build_supervised_example`) and will silently drop / skip a non-64 row. So **before
   accepting a winner, assert `len(best_codes) == 64`**; if not, treat it as "no valid sample" for that
   row (keep gold). The distilled row must still pass `sft.example.is_supported_materialized`.
7. **`token_status="distilled"` is safe** — verified that nothing in `sft/`, `eval/`, `data_pipeline/`,
   `scripts/` branches on `token_status`'s value (`is_supported_materialized`/`supported_rows` ignore
   it), so a new enum value does not drop or misroute the row.

## Files

- **New:** `scripts/build_distillation_corpus.py` (harvest + write the distilled corpus).
- **New:** `configs/candidate_distill.json` (candidate config pointing at the distilled corpus).
- **New:** `tests/test_build_distillation_corpus.py` (pure row-transform tests, no GPU).
- **Reuse:** `sft/train.py` (unchanged), `sft/score_tokens.py` (unchanged), `eval/best_of_n.py` (Doc 02).

## Step 1 — `scripts/build_distillation_corpus.py`

Load the P6 adapter, iterate the **source** corpus, harvest winners for training supported rows, write
a distilled `active_rows.jsonl` under a separate **out** dir. Source and out are explicit CLI args,
independent of any training config:
- `--source-rows` (default `data/active_sft/active_rows.jsonl`) — the P6 corpus to read.
- `--out-dir` (default `data/active_sft_distilled/`) — where the distilled corpus is written.
- `--adapter` (the harvest model, e.g. `models/sft_adapters/p6_twostage_d0f9c744_smokefull`),
  `--resized-model models/base_resized`, `--n` (default 16), `--temperature`/`--top-p`, `--tau`
  (default 0.5), `--limit` (smoke).
**Do NOT read `active_rows_path` from a training config here** — `candidate_distill.json` points at the
(not-yet-written) distilled path and is consumed only in Step 3. Reading it here would be circular.

Factor the per-row transform into a **pure, testable, torch-free** function (no model), then the
harvest loop calls it. Re-spell the assistant-target format locally (see invariant 5) so the module
import stays torch-free:
```python
def _assistant_target(codes):   # MUST equal scripts.materialize_target_tokens._assistant_target
    return "<lut_bos> " + " ".join(f"<lut_{c:03d}>" for c in codes) + " <lut_eos>"

def distill_row(row, best_codes, best_fid, tau):
    """Pure transform. Returns a (possibly) rewritten row. Rewrites ONLY when a valid, reachable
    winner clears the absolute bar; otherwise returns row unchanged (keep gold)."""
    if best_codes is None or len(best_codes) != 64:        # 64-guard (invariant 6)
        return row
    if (best_fid or 0.0) < tau:                            # absolute bar (invariant 2); NOT vs gold
        return row
    return {**row,
            "target_tokens": [int(c) for c in best_codes],
            "assistant_target": _assistant_target(best_codes),
            "token_status": "distilled"}
```

Harvest loop:
```
cfg = _load_config(args.config)                            # default SFTConfig(); only locked pixel/quant
                                                           # knobs matter for loading. NOT the source path.
rows = load_rows(args.source_rows)                         # full corpus (train + holdout)
model, processor = load_eval_model(cfg, args.resized_model, args.adapter)   # sft.loader (Doc 01)
out_rows, counts = [], Counter()
for row in rows:
    if not row["is_supported"]:            out_rows.append(row); counts["unsupported"]+=1; continue
    if is_holdout_row(row):                out_rows.append(row); counts["holdout"]+=1; continue   # sacred
    if not is_supported_materialized(row): out_rows.append(row); counts["not_materialized"]+=1; continue
    best_codes, best_rec = best_of_n_for_row(model, processor, row, n=args.n,
                                             sampling={"temperature":args.temperature,"top_p":args.top_p})
    new_row = distill_row(row, best_codes, (best_rec or {}).get("behavioral_fidelity"), args.tau)
    counts["replaced" if new_row is not row else "kept_gold"] += 1
    out_rows.append(new_row)
write out_rows -> {out_dir}/active_rows.jsonl              # backup + .tmp + os.replace, json.dumps(sort_keys=True)
copy {out_dir}/active_manifest.json with a "distillation" block
print counts + mean best-of-N fidelity over training rows
```

Details / requirements:
- **Conditioning** `input_field="attribute_spec_text"`, no bucketize (via `best_of_n_for_row`, which
  conditions with `input_text_for` and scores with the canonical spec — see Doc 02).
- **Batched harvest** via `generate_codes_batch`/`best_of_n_for_row` (Doc 01/02). Expensive step:
  N samples × 2641 training supported rows. `--limit` for a smoke first.
- **Resumability (non-deterministic by design):** cache per-row winners to a sidecar JSONL keyed by row
  `id` so a crash resumes without re-generating. Sampling is non-deterministic; that is expected — do
  **not** claim run-to-run determinism.
- **Manifest** `active_manifest.json` gets a `distillation` block: {source adapter, N, temperature, τ,
  replaced/kept_gold/holdout/unsupported counts, source path} for provenance.
- **Write discipline** mirrors `materialize_target_tokens.py`: back up, write `.tmp`, atomic
  `os.replace`, `json.dumps(row, sort_keys=True)` per line.

## Step 2 — `configs/candidate_distill.json`

Same locked knobs as `candidate_two_stage.json`, but pointed at the distilled corpus and P6 conditioning:
```json
{
  "learning_rate_lora": 0.0002, "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
  "warmup_ratio": 0.03, "max_grad_norm": 1.0, "weight_decay": 0.0, "max_pixels": 200704,
  "input_field": "attribute_spec_text",
  "active_rows_path": "data/active_sft_distilled/active_rows.jsonl"
}
```
`active_rows_path` is a real `SFTConfig` field (`_load_config` merges JSON overrides matching dataclass
fields; `__post_init__` has no path check, so this passes). **Governance caveat:** AGENTS.md's
locked-knob list includes "paths", so this override is *nominally* a locked knob. The lock governs the
bilevel hill-climb loop; a one-off, out-of-loop distillation experiment with **every locked
hyperparameter identical to P6** is a deliberate, documented exception — state it explicitly when you
run, and never let the bilevel loop propose a path change.

## Step 3 — train + evaluate

- **Train (fresh adapter — no resume; the trainer always builds a fresh LoRA):**
  `python -m sft.train --config configs/candidate_distill.json --resized-model models/base_resized --smoke-size 0 --run-id distill_r1`
  (or via `sft.bilevel_bridge --config configs/candidate_distill.json --run-id distill_r1` to also upload
  the adapter to HF, mirroring `notebooks/generator_retrain_run.ipynb` CELL 4).
- **Evaluate on the UNTOUCHED holdout:**
  `python -m sft.score_tokens --config configs/candidate_distill.json --adapter models/sft_adapters/distill_r1_smokefull --behavioral-sampling both --behavioral-limit 64`
  Compare `behavioral` (greedy) fidelity to the 0.159 baseline. The teacher-forced `METRIC=` is a
  secondary check (it may move less; free-running greedy is the number that matters here).
- **Round 2 (optional):** repeat Step 1 with `distill_r1` as the harvest model → `data/active_sft_distilled_r2/`
  → retrain. Stop when free-running fidelity plateaus.

## Step 4 — tests (`tests/test_build_distillation_corpus.py`, no GPU)

Test the pure `distill_row(row, best_codes, best_fid, tau)` (factored in Step 1 — no model, no torch):
- Winner ≥ τ and `len==64` → row with new `target_tokens`, `assistant_target` rebuilt (assert the
  **exact** string, e.g. `"<lut_bos> <lut_000> … <lut_eos>"`), `token_status == "distilled"`, and ALL
  other fields identical (esp. `split_unit_id`, `image_path`, `measured_behavior`). Assert the result
  still passes `sft.example.is_supported_materialized`.
- A separate test (guard with `pytest.importorskip("torch")`) asserts the build script's local
  `_assistant_target` equals `scripts.materialize_target_tokens._assistant_target` on a sample — pins
  the format against drift without making the main tests depend on torch.
- Winner **below τ** → row returned unchanged (identity — keep gold).
- Winner with `len(best_codes) != 64` (e.g. 63) → row unchanged (64-guard).
- `best_codes is None` (all-refused) → row unchanged.
- Harvest-loop routing (test the small dispatcher, stubbing `is_holdout_row`/`is_supported`): a holdout
  row and an unsupported row are copied unchanged and the model is never called on them.

## Cost estimate

Corpus (verified): **2761** supported / **120** holdout / **2641** supported-train. Full harvest ≈
**2641** training rows × N samples. Batched at `chunk=16`, N=16 is **one** `.generate` call per row
(num_return_sequences=16) → ~2641 calls on an A100 → order of a few hours. Do a `--limit 200` smoke
first to validate the pipeline and see the **replaced / kept_gold** ratio (compare it to the fraction
Doc 01 predicted at your chosen τ); then the full harvest overnight. Training itself is the standard
2-epoch QLoRA run (same cost as P6). (Note: the `--behavioral-limit 64` in Step 3's eval is a 64-row
slice of the 120-row holdout, not the holdout size.)

## Why this over RL/scheduled sampling (for the implementer's confidence)

- It reuses the existing, stable SFT loop — no policy-gradient variance, no critic, no biased
  scheduled-sampling gradient, and it fits the locked 2-epoch budget (the "extra" cost is a one-time
  generation pass, not extra training).
- It directly attacks exposure bias: the new targets are trajectories the model can actually reach, so
  MLE on them improves its *own-prefix* behavior.
- It composes with Doc 02: best-of-N on the distilled model compounds (higher floor + better coverage).
- RL (GRPO/MRT) remains the reserve ceiling-raiser if this plateaus; it is intentionally NOT specced
  here (large new infra: sample log-probs, group sampling, reference-KL — see README).

## Deliverable

`scripts/build_distillation_corpus.py` + `configs/candidate_distill.json` + tests; a `distill_r1`
adapter whose free-running greedy behavioral fidelity on the untouched holdout beats 0.159, with the
replaced/kept harvest counts reported.
