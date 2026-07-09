"""Target fidelity (L5) — DISABLED in this build.

Real responsibility (docs/eval_harness_implementation.md "Target Fidelity"): compare the
model's decoded LUT to the (tokenizer-decoded) target/reference LUTs via image + chart
CIEDE2000, selecting the gate by ``acceptance_mode``:

    exact_target      -> image/chart mean<=3.0 and p95<=8.0 vs single decoded target
    multi_reference   -> any of K decoded reference LUTs passes that gate
    behavior_window   -> measured behavior within the frozen per-dimension window

Needs a decoded LUT + a decoded target, both unavailable without the frozen tokenizer,
so this returns ``not_evaluated: decoder_disabled``.
"""

from __future__ import annotations

from .schemas import LayerResult


def target_fidelity_check(row, decoded_lut=None, decoded_target=None) -> LayerResult:  # noqa: ANN001
    return LayerResult.disabled("L5_target_fidelity")
