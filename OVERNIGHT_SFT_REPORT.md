# Overnight LUT-SLM SFT Improvement Report

## Status

**RUNNING — 1 of at most 5 full runs completed.**

Best full-run held-out LUT-code token accuracy: **0.413696**.

## Ledger

| Ledger iter | Full run | Knob changed | Metric | Delta vs current full best | Adapter / note |
|---:|---:|---|---:|---:|---|
| 1 | — | Historical baseline smoke (`smoke600`) | 0.044271 | -0.369425 (not comparable to full run) | `models/sft_adapters/bl_a0ccbcff_smoke600`; HF pushed |
| 2 | — | Historical `learning_rate_lora`: 0.0002 → 0.0003 | — | — | Non-counting transient remote Jupyter disconnect; no training result |
| 3 | 1 | Baseline (no knob change) | **0.413696** | **0.000000** | `models/sft_adapters/bl_a0ccbcff_smokefull`; HF pushed |

## Current winner

```json
{
  "learning_rate_lora": 0.0002,
  "lora_r": 16,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "warmup_ratio": 0.03,
  "max_grad_norm": 1.0,
  "weight_decay": 0.0,
  "max_pixels": 200704
}
```

- Local adapter: `models/sft_adapters/bl_a0ccbcff_smokefull`
- Hugging Face: `ericrcwu/LUT_SLM_sft_adapters/bl_a0ccbcff_smokefull`
- Training evidence: 5,184 rows, 162 optimizer steps, 544 expected unsupported absolute-path skips, mean loss 1.9172154127758134.
- Adapter SHA-256: `dcc5f50ea32fe0d59f5dfb1d02fda4996e54a2088618eb622e71f2e0c5dfe693`

## Errors and fixes

- Historical transient: Cursor lost the remote Jupyter connection before a candidate produced output. Per the run rules this did not count as a candidate failure or full run. The notebook was reattached to the existing Colab session and the full baseline was run successfully.
- No error occurred in full run 1. The pre-flight correctly reused the staged corpus and `models/base_resized`; Gate 0 remained skipped as already complete.

## Next steps

1. Launch the required proposal subagent against the persisted ledger.
2. Validate that its strict JSON changes exactly one allowed knob from the current full-run winner.
3. Run the next whole-corpus two-epoch evaluation, then append its metric/upload result and refresh this report.
