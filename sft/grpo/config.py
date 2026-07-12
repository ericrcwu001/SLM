"""GRPO configuration (docs/grpo/04_training_loop.md).

**Design: compose, do not mutate.** ``GRPOConfig`` WRAPS a frozen :class:`sft.config.SFTConfig` rather
than adding fields to it, so the SFT locked identity + its ``__post_init__`` lock enforcement
(``sft/config.py:98-118``) stay byte-identical to SFT. The locked SFT identity (base, quant,
``num_new_tokens``, ``max_seq_len``, ``seed``, paths) is inherited untouched; the GRPO **methodology**
knobs live here, firewalled from the locked bilevel search (flagged exactly like the Phase-3 soft-loss
knobs, ``sft/config.py:82-91`` — the bilevel loop must never propose them).

``load_grpo_config`` reads a FLAT JSON: SFT-known keys fill the wrapped ``SFTConfig`` (same coercion as
``load_config``), the GRPO-declared keys fill ``GRPOConfig``, and any other key is a HARD ERROR (guards a
typo in a methodology knob). A name on BOTH dataclasses (e.g. ``ckpt_every``) routes to ``GRPOConfig`` —
the value the loop actually reads via ``gcfg.ckpt_every`` — and never double-populates.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from pathlib import Path

from sft.config import SFTConfig

_INT_RE = re.compile(r"^[+-]?\d+$")
_FLOAT_RE = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def _coerce_num(v):
    """Coerce a numeric-looking string to int/float (PyYAML leaves ``5e-6`` etc. as a string)."""
    if not isinstance(v, str):
        return v
    s = v.strip()
    if _INT_RE.match(s):
        return int(s)
    if _FLOAT_RE.match(s):
        return float(s)
    return v

# The P6 two-stage adapter — the GRPO policy init AND the frozen KL reference (grounding: distill_r1
# does NOT exist; use P6). Gitignored on a fresh clone (snapshot_download from the HF adapters repo).
DEFAULT_INIT_ADAPTER = "models/sft_adapters/p6_twostage_d0f9c744_smokefull"


@dataclass(frozen=True)
class GRPOConfig:
    """Immutable GRPO methodology config wrapping a locked :class:`SFTConfig`."""

    sft: SFTConfig                                    # locked identity + SFT-tunable knobs
    init_adapter: str = DEFAULT_INIT_ADAPTER          # policy init AND frozen reference

    # -- rollout (methodology) --
    group_size: int = 8                               # G samples/prompt (advantage needs G >= 2)
    rollout_temperature: float = 0.7                  # where oracle@32=0.42 coverage was measured
    rollout_top_p: float = 0.9
    rollout_chunk: int = 16                           # ceil(G/chunk) .generate calls
    prompts_per_round: int = 8                        # prompts sampled per rollout round

    # -- optimization (methodology) --
    grpo_lr: float = 5.0e-6                           # RL lr << SFT 2e-4 (Doc 03 §7)
    warmup_steps: int = 10
    grad_accum: int = 8                               # prompt-groups accumulated per optimizer step
    update_epochs: int = 1                            # μ inner passes (μ=1 => ρ≡1 => clip inactive)
    clip_eps: float = 0.2                             # ε
    kl_beta: float = 0.05                             # β
    adv_eps: float = 1.0e-4                           # eps_adv (advantage std-divide guard)
    entropy_coef: float = 0.0                         # optional rollout-entropy bonus (off by default)
    total_steps: int = 500                            # optimizer-step budget (NOT sft.epochs)

    # -- checkpoint / eval (methodology) --
    ckpt_every: int = 20                              # C: save `latest` every C steps + on SIGINT
    eval_every: int = 20                              # holdout greedy eval cadence (optimizer steps)
    eval_limit: int = 64                              # holdout slice size for the periodic eval
    keep_history: bool = False                        # also snapshot history/step_NNNNNN/ each eval

    # -- reward shaping (OWNED by Doc 01; carried in the same JSON, passed through) --
    collapse_penalty: float = 0.25                    # Doc 01 §2/§5 default
    delta_e_weight: float = 0.0                       # eval-only ΔE; never enters reward selection

    # -- anti-hacking guard bands (Doc 05; PROVISIONAL anchors, calibrate on the first run's init
    #    reading — NOT in the canonical JSON, so they take these defaults unless overridden) --
    guard_collapse_margin: float = 0.10               # veto if collapse_rate > init + this
    guard_degenerate_ceiling: float = 0.02            # veto if degenerate_rate above this
    guard_delta_e_margin: float = 1.0                 # veto if decoded_delta_e_mean > best + this
    guard_entropy_floor_frac: float = 0.5             # veto if code_entropy_norm_mean < frac * init
    guard_kl_ceiling: float = 10.0                    # veto if mean per-token KL(policy||ref) above this
    early_stop_patience: int = 6                      # P: evals without a new BEST -> stop at plateau
    bad_window: int = 4                               # K: consecutive bad evals -> divergence stop

    def __post_init__(self):
        if self.group_size < 2:
            raise ValueError("group_size must be >= 2 (group-relative std needs >= 2 samples)")
        if not (0.0 < self.clip_eps < 1.0):
            raise ValueError("clip_eps must be in (0, 1)")
        if self.kl_beta < 0:
            raise ValueError("kl_beta must be >= 0")
        if self.adv_eps <= 0:
            raise ValueError("adv_eps must be > 0")
        if self.grad_accum < 1 or self.update_epochs < 1 or self.prompts_per_round < 1:
            raise ValueError("grad_accum, update_epochs, prompts_per_round must be >= 1")
        if self.ckpt_every < 1 or self.eval_every < 1:
            raise ValueError("ckpt_every and eval_every must be >= 1")
        if self.total_steps < 1:
            raise ValueError("total_steps must be >= 1")
        if self.rollout_chunk < 1:
            raise ValueError("rollout_chunk must be >= 1")
        # NOTE: SFTConfig.__post_init__ already enforced the locked identity; do not re-check it here.

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)


def load_grpo_config(path: str) -> GRPOConfig:
    """Load a flat JSON into a :class:`GRPOConfig` (SFT keys -> SFTConfig, GRPO keys -> GRPOConfig).

    Unknown keys are a hard error. A shared name (e.g. ``ckpt_every``) routes to GRPO only, so
    ``gcfg.ckpt_every`` reflects the JSON while ``SFTConfig`` keeps its (unused-in-GRPO) default.

    Parsed as JSON first (the file is flat JSON — this handles scientific notation like ``5e-6`` that
    PyYAML's float resolver silently leaves as a string), falling back to YAML for a ``.yaml`` override.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        import yaml
        raw = yaml.safe_load(text) or {}
        raw = {k: _coerce_num(v) for k, v in raw.items()}   # PyYAML leaves 5e-6/1e-4 as strings
    grpo_names = {f.name for f in fields(GRPOConfig)} - {"sft"}     # GRPO names WIN on a collision
    sft_names = {f.name for f in fields(SFTConfig)}
    unknown = set(raw) - sft_names - grpo_names
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
    sft_kw = {k: (tuple(v) if isinstance(v, list) else v)
              for k, v in raw.items() if k in sft_names and k not in grpo_names}
    grpo_kw = {k: v for k, v in raw.items() if k in grpo_names}
    return GRPOConfig(sft=SFTConfig(**sft_kw), **grpo_kw)
