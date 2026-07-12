"""Checkpointed GRPO training loop for the prompt->LUT generator (docs/grpo/04 + 05).

Directly optimizes the FREE-RUNNING behavioral reward so free-running GREEDY behavioral fidelity climbs
from ~0.159 toward/past the oracle 0.42. Policy = the P6 QLoRA LoRA params; KL reference = the FROZEN
P6 init (a second, frozen named adapter on the SAME 4-bit base — never ``disable_adapter()``); 4-bit
NF4 base shared and frozen. No value net (the group mean is the baseline).

Each ROUND: sample ``prompts_per_round`` prompts × G grammar-constrained rollouts [02] -> shaped reward
[01] -> group-relative advantages [01] -> a per-prompt buffer. Then ``update_epochs`` passes over the
buffer, each accumulating ``grad_accum`` prompt-groups per optimizer step of the clipped surrogate + KL
[03]. One GRPO **step** = one optimizer update; checkpoint/eval cadence counts steps.

The loop is **anytime**: ``latest/`` is written atomically every ``ckpt_every`` steps AND on
SIGINT/SIGTERM (resume only); a guard-vetoed ``best/`` [05] is a separate directory (the deployable).
Interrupting at any moment leaves a usable, holdout-validated adapter. Submit ``best/``, never
``latest/``.

Machine-readable output: a single ``{"grpo_summary": {...}}`` line before ``[grpo][OK]``; a round that
produces zero gradable samples prints ``[grpo][ABORT]`` and returns non-zero (never a misleading OK on
an untrained adapter). Heavy deps (torch/transformers/peft) are imported lazily. Runs nothing on import.

Usage (Colab A100):
    python -m sft.grpo_train --config configs/candidate_grpo.json --resized-model models/base_resized \
        --run-id grpo_smoke --total-steps 4 --prompts-per-round 2
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import shutil
import signal
from dataclasses import dataclass, field
from pathlib import Path

from data_pipeline.errors import RequiresTokenizer, SFTError
from sft.grpo.config import GRPOConfig, load_grpo_config
from sft.manifest import build_adapter_manifest, write_manifest

_INTERRUPTED = False


def install_signal_handlers() -> None:
    """SIGINT (Ctrl-C) + SIGTERM (Colab preemption) -> set a flag; the loop saves at the next safe
    boundary. A SECOND signal forces an immediate KeyboardInterrupt (escape hatch if a save hangs)."""
    def _on_signal(signum, _frame):
        global _INTERRUPTED
        if _INTERRUPTED:
            raise KeyboardInterrupt
        _INTERRUPTED = True
        print(f"[grpo] signal {signum} received — will checkpoint `latest` and exit cleanly")
    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)


# ---------------------------------------------------------------------------------------------------
# Trainer state (persisted in trainer_state.pt alongside the optimizer + rng)
# ---------------------------------------------------------------------------------------------------
@dataclass
class TrainerState:
    step: int = 0
    round: int = 0
    data_cursor: int = 0
    best_summary: dict | None = None
    init_summary: dict | None = None
    evals_since_best: int = 0
    consecutive_bad: int = 0
    _order_cache: tuple = field(default=None, repr=False, compare=False)


# ---------------------------------------------------------------------------------------------------
# Seeding / RNG capture
# ---------------------------------------------------------------------------------------------------
def _seed_everything(seed: int):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _rng_state() -> dict:
    import random

    import numpy as np
    import torch

    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }


def _restore_rng(st: dict | None):
    if not st:
        return
    import random

    import numpy as np
    import torch

    torch.set_rng_state(st["torch"])
    if st.get("cuda") is not None and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(st["cuda"])
        except Exception:  # noqa: BLE001 — device-count mismatch on resume; bookkeeping only
            pass
    random.setstate(st["python"])
    np.random.set_state(st["numpy"])


# ---------------------------------------------------------------------------------------------------
# Model stack — two adapters (policy trainable + reference frozen) on ONE shared 4-bit NF4 base
# ---------------------------------------------------------------------------------------------------
def _disable_dropout(model) -> None:
    """Zero every ``nn.Dropout`` so a ``.train()``-mode forward is numerically identical to ``.eval()``.

    GRPO needs dropout OFF (LoRA dropout would give the old and current forwards different masks and
    corrupt the importance ratio — Doc 02 invariant 6), but it also needs the UPDATE forward to run in
    ``.train()`` so gradient checkpointing engages (transformers gates checkpointing on ``self.training``,
    else full activations are retained -> OOM risk on the A100). Globally zeroing dropout reconciles
    both: ``ρ≡1`` on the first inner step is preserved AND checkpointing fires. The P6 adapter ships
    ``lora_dropout=0.05`` (baked into its ``adapter_config.json``, NOT read from the GRPO JSON), so this
    must be done programmatically, not via config."""
    import torch.nn as nn

    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = 0.0


def build_stack(gcfg: GRPOConfig, resized_model: str, *, policy_init: str, reference_init: str):
    """Return ``(policy, processor, opt, trainable)``.

    Mirrors ``sft/train.py:97-113`` for the base + kbit prep, then attaches the trainable ``policy``
    adapter (from ``policy_init`` — P6 fresh, or ``latest/`` on resume) and the FROZEN ``reference``
    adapter (always P6 = ``reference_init``) on the SAME base. Dropout is globally zeroed so the update
    forward can run in ``.train()`` (gradient checkpointing on) while staying numerically identical to
    ``.eval()`` (Doc 02 invariant 6)."""
    try:
        import torch
        from peft import PeftModel, prepare_model_for_kbit_training
        from transformers import AutoProcessor, BitsAndBytesConfig
    except Exception as exc:  # noqa: BLE001
        raise SFTError(f"QLoRA stack unavailable (install the `sft` extra): {exc}") from exc
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _ModelCls
    except Exception:  # noqa: BLE001
        from transformers import AutoModelForVision2Seq as _ModelCls  # type: ignore

    from sft.example import resolve_compute_dtype
    from sft.rollout import init_code_maps

    compute_dtype = resolve_compute_dtype(gcfg.sft)
    processor = AutoProcessor.from_pretrained(resized_model, trust_remote_code=True,
                                              min_pixels=gcfg.sft.min_pixels, max_pixels=gcfg.sft.max_pixels)
    bnb = BitsAndBytesConfig(
        load_in_4bit=gcfg.sft.load_in_4bit, bnb_4bit_quant_type=gcfg.sft.bnb_4bit_quant_type,
        bnb_4bit_use_double_quant=gcfg.sft.bnb_4bit_use_double_quant,
        bnb_4bit_compute_dtype=compute_dtype)
    base = _ModelCls.from_pretrained(resized_model, quantization_config=bnb, torch_dtype=compute_dtype,
                                     device_map="auto", trust_remote_code=True)
    base = prepare_model_for_kbit_training(base, use_gradient_checkpointing=gcfg.sft.gradient_checkpointing)

    policy = PeftModel.from_pretrained(base, policy_init, adapter_name="policy", is_trainable=True)
    policy.load_adapter(reference_init, adapter_name="reference")   # frozen (is_trainable defaults False)
    policy.set_adapter("policy")
    _disable_dropout(policy)                       # so .train() (checkpointing) == .eval() numerically
    policy.eval()

    trainable = [p for p in policy.parameters() if p.requires_grad]
    if not trainable:
        raise SFTError("no trainable params on the policy adapter (is_trainable / modules_to_save?)")
    opt = torch.optim.AdamW(trainable, lr=gcfg.grpo_lr, weight_decay=gcfg.sft.weight_decay)
    init_code_maps(processor.tokenizer)
    return policy, processor, opt, trainable


def _set_mode(policy, *, generate: bool):
    """Toggle train/eval + ``use_cache`` at the rollout(generate)↔update boundary; keep ``policy`` active.

    Generation/eval: ``.eval()`` + ``use_cache=True`` (no autograd graph). Update: ``.train()`` +
    ``use_cache=False`` so gradient checkpointing engages (it is gated on ``self.training``). Dropout was
    globally zeroed in ``build_stack``, so ``.train()`` and ``.eval()`` forwards are numerically identical
    and ``ρ≡1`` on the first inner update is preserved."""
    try:
        policy.config.use_cache = bool(generate)
    except Exception:  # noqa: BLE001
        pass
    try:
        policy.set_adapter("policy")
    except Exception:  # noqa: BLE001
        pass
    policy.eval() if generate else policy.train()


# ---------------------------------------------------------------------------------------------------
# Optimizer step / trailing flush (mirror sft/train.py:154-201)
# ---------------------------------------------------------------------------------------------------
def _lr(step: int, gcfg: GRPOConfig) -> float:
    """Short linear warmup to a FLAT grpo_lr (GRPO runs open-ended total_steps — no cosine-to-zero)."""
    if gcfg.warmup_steps and step < gcfg.warmup_steps:
        return gcfg.grpo_lr * (step + 1) / max(1, gcfg.warmup_steps)
    return gcfg.grpo_lr


def optim_step(policy, opt, gcfg: GRPOConfig, state: TrainerState, trainable):
    import torch

    for g in opt.param_groups:
        g["lr"] = _lr(state.step, gcfg)
    torch.nn.utils.clip_grad_norm_(trainable, gcfg.sft.max_grad_norm)
    opt.step()
    opt.zero_grad(set_to_none=True)
    state.step += 1


# ---------------------------------------------------------------------------------------------------
# Data cursor: deterministic round-robin over the train pool, reshuffled per epoch (resumable)
# ---------------------------------------------------------------------------------------------------
def _epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    import random as _r

    order = list(range(n))
    _r.Random(seed * 1_000_003 + epoch).shuffle(order)
    return order


def next_prompts(rows, state: TrainerState, k: int, seed: int) -> list[dict]:
    """Take the next ``k`` training rows, advancing ``state.data_cursor`` (a global sample counter).

    Deterministic and resumable: the per-epoch shuffle is a pure function of ``(seed, epoch)`` where
    ``epoch = data_cursor // n``, so restoring ``data_cursor`` reproduces the exact order."""
    n = len(rows)
    out: list[dict] = []
    for _ in range(k):
        epoch, idx = divmod(state.data_cursor, n)
        if state._order_cache is None or state._order_cache[0] != epoch:
            state._order_cache = (epoch, _epoch_order(n, seed, epoch))
        out.append(rows[state._order_cache[1][idx]])
        state.data_cursor += 1
    return out


# ---------------------------------------------------------------------------------------------------
# Checkpoint IO — atomic dir swap, adapter save, trainer state
# ---------------------------------------------------------------------------------------------------
def _atomic_write_dir(write_fn, dst: Path):
    """Write via a three-step dir swap so an interrupt never leaves a half-written ``dst``.

    ``os.replace`` is atomic but can only replace an EMPTY dir (POSIX ``rename`` raises ENOTEMPTY over a
    populated one), so a single replace over a live dir crashes on the 2nd save. Instead: write to
    ``dst.tmp``; ``replace(dst -> dst.old)``; ``replace(dst.tmp -> dst)``; remove ``dst.old``. Each
    replace is itself atomic."""
    dst = Path(dst)
    tmp = dst.with_name(dst.name + ".tmp")
    old = dst.with_name(dst.name + ".old")
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(old, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    write_fn(tmp)
    if dst.exists():
        os.replace(dst, old)
    os.replace(tmp, dst)
    shutil.rmtree(old, ignore_errors=True)


def _grpo_manifest_block(gcfg: GRPOConfig, state: TrainerState) -> dict:
    best_fid = (state.best_summary or {}).get("behavioral_fidelity_mean")
    return {
        "init_adapter": gcfg.init_adapter, "group_size": gcfg.group_size, "clip_eps": gcfg.clip_eps,
        "kl_beta": gcfg.kl_beta, "rollout_temperature": gcfg.rollout_temperature,
        "rollout_top_p": gcfg.rollout_top_p, "grpo_lr": gcfg.grpo_lr, "total_steps": gcfg.total_steps,
        "step": state.step, "best_greedy_fidelity": best_fid,
    }


def _flatten_adapter_subdir(dst: Path, name: str) -> None:
    """peft's ``save_pretrained(selected_adapters=[name])`` writes a NON-'default' adapter under
    ``dst/<name>/``; move those files up to ``dst/`` so ``PeftModel.from_pretrained(base, dst)`` (resume,
    deploy, and ``sft.loader.load_eval_model``) finds a FLAT ``adapter_config.json`` /
    ``adapter_model.safetensors`` — the layout the whole eval/SFT ecosystem assumes."""
    sub = Path(dst) / name
    if not sub.is_dir():
        return
    for f in sub.iterdir():
        target = Path(dst) / f.name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else target.unlink()
        os.replace(f, target)
    sub.rmdir()


def _write_adapter(policy, processor, gcfg: GRPOConfig, state: TrainerState, resized_model: str,
                   dst: Path):
    """Write the TRAINED ``policy`` adapter (never the frozen reference) + tokenizer + manifest to dst."""
    from sft.example import artifact_root
    from tokenizer.manifest import hash_state_dict

    policy.save_pretrained(str(dst), selected_adapters=["policy"])
    _flatten_adapter_subdir(dst, "policy")        # peft nests dst/policy/* -> flatten to dst/*
    processor.tokenizer.save_pretrained(str(dst))
    trainable_sd = {n: p for n, p in policy.named_parameters() if p.requires_grad}
    adapter_sha = hash_state_dict({n: p.detach().float().cpu() for n, p in trainable_sd.items()})
    tok_manifest = _read_json(artifact_root() / "tokenizer" / "final" / "manifest.json")
    vr_manifest = _read_json(Path(resized_model) / "vocab_resize_manifest.json")
    manifest = build_adapter_manifest(
        run_id=str(dst.parent.name), adapter_step=state.step, adapter_sha256=adapter_sha,
        cfg=gcfg.sft.to_dict(), tokenizer_manifest=tok_manifest, vocab_resize_manifest=vr_manifest,
        train_report={"grpo": _grpo_manifest_block(gcfg, state)})
    manifest["grpo"] = _grpo_manifest_block(gcfg, state)
    write_manifest(dst / "adapter_manifest.json", manifest)
    (dst / "grpo_state.json").write_text(json.dumps({
        "step": state.step, "round": state.round,
        "best_greedy_fidelity": (state.best_summary or {}).get("behavioral_fidelity_mean"),
    }, indent=2), encoding="utf-8")
    return adapter_sha


def save_latest(policy, processor, opt, gcfg: GRPOConfig, state: TrainerState, resized_model: str,
                run_dir: Path):
    """Atomically write ``latest/`` (policy adapter + tokenizer + manifest + trainer_state.pt)."""
    import torch

    def _write(tmp: Path):
        _write_adapter(policy, processor, gcfg, state, resized_model, tmp)
        torch.save({
            "opt_state_dict": opt.state_dict(), "step": state.step, "round": state.round,
            "data_cursor": state.data_cursor, "best_summary": state.best_summary,
            "init_summary": state.init_summary, "evals_since_best": state.evals_since_best,
            "consecutive_bad": state.consecutive_bad, "rng": _rng_state(),
        }, tmp / "trainer_state.pt")

    _atomic_write_dir(_write, run_dir / "latest")


def save_best(policy, processor, gcfg: GRPOConfig, state: TrainerState, resized_model: str,
              run_dir: Path):
    """Atomically write the guard-vetoed ``best/`` (policy adapter only; no trainer_state — deploy)."""
    _atomic_write_dir(lambda tmp: _write_adapter(policy, processor, gcfg, state, resized_model, tmp),
                      run_dir / "best")


def recover_latest(run_dir: Path) -> None:
    """Promote an orphaned checkpoint if a crash landed in the middle of the ``latest/`` dir swap.

    The three-step swap has a 1-syscall window where ``latest/`` is absent but ``latest.old/`` (or
    ``latest.tmp/``) holds a complete checkpoint; a ``kill -9`` / second SIGINT there would otherwise make
    ``resume_or_init`` silently restart from P6, discarding all progress."""
    latest = run_dir / "latest"
    if (latest / "trainer_state.pt").exists():
        return
    for cand in (run_dir / "latest.tmp", run_dir / "latest.old"):
        if (cand / "trainer_state.pt").exists():
            if latest.exists():
                shutil.rmtree(latest, ignore_errors=True)
            os.replace(cand, latest)
            print(f"[grpo] recovered interrupted checkpoint {cand.name} -> latest/")
            return


def resume_or_init(run_dir: Path, gcfg: GRPOConfig) -> tuple[bool, str]:
    """Decide the policy init: resume from ``latest/`` if present, else fresh from P6.

    Returns ``(resuming, policy_init_path)``. The frozen reference is ALWAYS P6 (``gcfg.init_adapter``),
    regardless of resume."""
    latest = run_dir / "latest"
    if (latest / "trainer_state.pt").exists():
        return True, str(latest)
    return False, gcfg.init_adapter


# ---------------------------------------------------------------------------------------------------
# Periodic holdout greedy eval + guard-vetoed BEST (Doc 05)
# ---------------------------------------------------------------------------------------------------
def holdout_greedy_eval(policy, processor, sft_cfg, *, limit: int) -> dict:
    """Free-running GREEDY behavioral fidelity on the untouched holdout (the gate) — read-only.

    Reuses ``sft.score_tokens._run_behavioral`` (greedy: ``sampling=None``) on
    ``supported_rows(rows, holdout=True)``; returns the ``summarize_fidelity`` panel + ``scored``/
    ``refused``. Holdout is sacred (Invariant 1): only scored, never trained on."""
    from sft.example import load_rows, supported_rows
    from sft.score_tokens import _run_behavioral

    rows = supported_rows(load_rows(sft_cfg.active_rows_path), holdout=True)
    if limit:
        rows = rows[:limit]
    if not rows:
        raise SFTError("no held-out supported rows for the periodic eval (empty holdout slice)")
    _set_mode(policy, generate=True)
    return _run_behavioral(policy, processor, rows, input_field=sft_cfg.input_field,
                           bucketize=getattr(sft_cfg, "spec_bucketize", False), sampling=None,
                           device=policy.device)


def is_best(new_summary: dict, best_summary: dict | None, init_summary: dict | None,
            gcfg: GRPOConfig) -> bool:
    """Guard-vetoed argmax (Doc 05): a NEW HIGH greedy fidelity AND every reward-hacking guard healthy.

    Guards vs the init reading (collapse/entropy/KL) or the running BEST (ΔE). A fidelity gain that
    trips any veto is logged (so the hack is visible) but does NOT overwrite BEST. The FIRST eval
    (``best_summary is None``) always passes -> BEST starts at the init baseline."""
    fid = new_summary.get("behavioral_fidelity_mean")
    if fid is None:
        return False
    best_fid = (best_summary or {}).get("behavioral_fidelity_mean")
    if best_fid is not None and not (fid > best_fid):
        return False
    init = init_summary or {}
    best = best_summary or {}

    cr, icr = new_summary.get("collapse_rate"), init.get("collapse_rate")
    if cr is not None and icr is not None and cr > icr + gcfg.guard_collapse_margin:
        return False
    dg = new_summary.get("degenerate_rate")
    if dg is not None and dg > gcfg.guard_degenerate_ceiling:
        return False
    de, bde = new_summary.get("decoded_delta_e_mean"), best.get("decoded_delta_e_mean")
    if de is not None and bde is not None and de > bde + gcfg.guard_delta_e_margin:
        return False
    ent, ient = new_summary.get("code_entropy_norm_mean"), init.get("code_entropy_norm_mean")
    if ent is not None and ient is not None and ent < gcfg.guard_entropy_floor_frac * ient:
        return False
    kl = new_summary.get("kl_to_ref")
    if kl is not None and kl > gcfg.guard_kl_ceiling:
        return False
    return True


def append_eval_log(run_dir: Path, step: int, summ: dict, is_best_flag: bool):
    rec = {"step": step, "is_best": bool(is_best_flag)}
    for k in ("behavioral_fidelity_mean", "collapse_rate", "degenerate_rate", "code_entropy_norm_mean",
              "decoded_delta_e_mean", "kl_to_ref", "rollout_entropy", "reward_mean", "advantage_std",
              "scored", "refused"):
        if k in summ:
            rec[k] = summ[k]
    with open(run_dir / "eval_log.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def _read_json(p: Path) -> dict:
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


# ---------------------------------------------------------------------------------------------------
# Loop telemetry accumulator
# ---------------------------------------------------------------------------------------------------
@dataclass
class LoopStats:
    kl_sum: float = 0.0
    ent_sum: float = 0.0
    reward_sum: float = 0.0
    adv_std_sum: float = 0.0
    n: int = 0

    def add(self, *, kl: float, rollout_entropy, reward_mean, advantage_std):
        self.kl_sum += float(kl)
        if rollout_entropy is not None:
            self.ent_sum += float(rollout_entropy)
        if reward_mean is not None:
            self.reward_sum += float(reward_mean)
        if advantage_std is not None:
            self.adv_std_sum += float(advantage_std)
        self.n += 1

    def snapshot(self) -> dict:
        n = max(1, self.n)
        return {"kl_to_ref": self.kl_sum / n, "rollout_entropy": self.ent_sum / n,
                "reward_mean": self.reward_sum / n, "advantage_std": self.adv_std_sum / n}


# ---------------------------------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------------------------------
def train(gcfg: GRPOConfig, resized_model: str, out_dir: str, run_id: str) -> int:
    import torch

    from eval.grpo_reward import group_advantages, shaped_rewards
    from sft.example import artifact_root, load_rows, supported_rows
    from sft.grpo_loss import grpo_loss
    from sft.rollout import RolloutGroup, code_logprobs, rollout_row

    if gcfg.entropy_coef and gcfg.entropy_coef > 0:
        raise SFTError("entropy_coef>0 is not wired in v1 — keep it 0 (rollout entropy is a guard, "
                       "not a bonus). See docs/grpo/03 §7.")

    run_dir = Path(out_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    recover_latest(run_dir)                        # promote an orphaned mid-swap checkpoint (if any)
    resuming, policy_init = resume_or_init(run_dir, gcfg)

    if not resuming:
        _seed_everything(gcfg.sft.seed)
    policy, processor, opt, trainable = build_stack(
        gcfg, resized_model, policy_init=policy_init, reference_init=gcfg.init_adapter)

    rows = supported_rows(load_rows(gcfg.sft.active_rows_path), holdout=False)  # TRAIN pool
    if not rows:
        raise SFTError("no materialized supported (non-holdout) rows to roll out on")

    state = TrainerState()
    if resuming:
        ts = torch.load(run_dir / "latest" / "trainer_state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(ts["opt_state_dict"])
        state.step = ts["step"]; state.round = ts["round"]; state.data_cursor = ts["data_cursor"]
        state.best_summary = ts["best_summary"]; state.init_summary = ts.get("init_summary")
        state.evals_since_best = ts.get("evals_since_best", 0)
        state.consecutive_bad = ts.get("consecutive_bad", 0)
        _restore_rng(ts.get("rng"))
        print(f"[grpo] RESUME run_id={run_id} step={state.step} "
              f"best_fid={(state.best_summary or {}).get('behavioral_fidelity_mean')}")
    else:
        print(f"[grpo] RUN_BEGIN run_id={run_id} rows={len(rows)} seed={gcfg.sft.seed} "
              f"input_field={gcfg.sft.input_field} artifact_root={artifact_root()} "
              f"init_adapter={gcfg.init_adapter}")

    install_signal_handlers()

    # Establish the init reading (BEST baseline + guard bands) once, before any update.
    if state.init_summary is None:
        summ = holdout_greedy_eval(policy, processor, gcfg.sft, limit=gcfg.eval_limit)
        state.init_summary = summ
        state.best_summary = summ
        append_eval_log(run_dir, state.step, {**summ, "kl_to_ref": 0.0}, is_best_flag=True)
        save_best(policy, processor, gcfg, state, resized_model, run_dir)
        print(f"[grpo] INIT greedy fidelity={summ.get('behavioral_fidelity_mean')} "
              f"collapse_rate={summ.get('collapse_rate')} (BEST baseline)")

    total_gradable = 0
    empty_rounds = 0
    stop = False
    while state.step < gcfg.total_steps and not _INTERRUPTED and not stop:
        # ---- ROLLOUT ROUND (no grad; use_cache=True; adapter='policy') ------------------------------
        _set_mode(policy, generate=True)
        buffer: list = []
        round_valid = 0
        for row in next_prompts(rows, state, gcfg.prompts_per_round, gcfg.sft.seed):
            with torch.no_grad():
                samples = rollout_row(policy, processor, row, gcfg.sft, G=gcfg.group_size,
                                      sampling={"temperature": gcfg.rollout_temperature,
                                                "top_p": gcfg.rollout_top_p},
                                      chunk=gcfg.rollout_chunk, device=policy.device)
            round_valid += sum(1 for s in samples if s.valid64)
            spec_text = samples[0].spec_text if samples else None
            rewards = shaped_rewards([s.codes for s in samples], spec_text, device=policy.device,
                                     collapse_penalty=gcfg.collapse_penalty,
                                     delta_e_weight=gcfg.delta_e_weight)
            adv = group_advantages([r for r, _ in rewards], eps=gcfg.adv_eps)
            group = RolloutGroup(samples, rewards, adv)
            if group.has_grad():
                buffer.append(group)
        state.round += 1
        total_gradable += sum(len(g.gradable()) for g in buffer)

        # Fail-loud: a first round with zero valid-64 rollouts means generation/images are broken
        # (SLM_ARTIFACT_ROOT case trap) — never spin on a no-op.
        if round_valid == 0 and state.step == 0:
            print(json.dumps({"grpo_summary": {"run_id": run_id, "step": state.step,
                                               "rows_trained": 0, "aborted": True}}))
            print(f"[grpo][ABORT] round produced 0 valid-64 rollouts — check SLM_ARTIFACT_ROOT "
                  f"({artifact_root()}) / image paths / the adapter")
            return 1
        if not buffer:
            empty_rounds += 1
            print(f"[grpo][warn] round {state.round}: 0 gradable groups (all refused/None) "
                  f"[{empty_rounds} consecutive]")
            if empty_rounds >= 5:
                print(json.dumps({"grpo_summary": {"run_id": run_id, "step": state.step,
                                                   "rows_trained": total_gradable, "aborted": True}}))
                print(f"[grpo][ABORT] {empty_rounds} consecutive rounds produced 0 gradable rollouts — "
                      f"reward scoring or generation is broken (not a transient)")
                return 1
            continue
        empty_rounds = 0

        # ---- μ (update_epochs) UPDATE PASSES (grad; use_cache=False; adapter='policy') --------------
        _set_mode(policy, generate=False)
        lstats = LoopStats()
        import random as _r
        for _ in range(gcfg.update_epochs):
            _r.Random(gcfg.sft.seed + state.step).shuffle(buffer)
            for micro, rg in enumerate(buffer):
                batch, old_lp, ref_lp, adv_t = rg.build(policy.device)
                cur_lp, sel = code_logprobs(policy, batch)
                loss, lst = grpo_loss(cur_lp, old_lp, ref_lp, adv_t, sel,
                                      clip_eps=gcfg.clip_eps, kl_beta=gcfg.kl_beta)
                (loss / gcfg.grad_accum).backward()
                lstats.add(kl=lst["kl_mean"], rollout_entropy=rg.entropy_mean,
                           reward_mean=rg.reward_mean, advantage_std=rg.advantage_std)
                if (micro + 1) % gcfg.grad_accum == 0:
                    optim_step(policy, opt, gcfg, state, trainable)
                    stop = _post_step(policy, processor, opt, gcfg, state, resized_model, run_dir,
                                      lstats) or stop
                    if state.step >= gcfg.total_steps or _INTERRUPTED or stop:
                        stop = True
                        break
            # trailing sub-grad_accum window: apply it UNLESS stopping (never fire an extra step past
            # total_steps / a SIGINT boundary — mirrors sft/train.py:197-201)
            if not stop and (len(buffer) % gcfg.grad_accum != 0):
                optim_step(policy, opt, gcfg, state, trainable)
                stop = _post_step(policy, processor, opt, gcfg, state, resized_model, run_dir,
                                  lstats) or stop
            if stop:
                break

    # Final anytime save.
    save_latest(policy, processor, opt, gcfg, state, resized_model, run_dir)

    if state.step == 0 and total_gradable == 0 and not _INTERRUPTED:
        print(json.dumps({"grpo_summary": {"run_id": run_id, "step": 0, "rows_trained": 0,
                                           "aborted": True}}))
        print(f"[grpo][ABORT] 0 optimizer steps and 0 gradable rollouts — nothing was trained")
        return 1
    # (an interrupt during the initial baseline eval still leaves a valid `best/` — fall through to
    #  the summary + a clean exit rather than a misleading ABORT.)

    best_fid = (state.best_summary or {}).get("behavioral_fidelity_mean")
    init_fid = (state.init_summary or {}).get("behavioral_fidelity_mean")
    summary = {"run_id": run_id, "step": state.step, "rounds": state.round,
               "best_greedy_fidelity": best_fid, "init_greedy_fidelity": init_fid,
               "total_gradable_samples": total_gradable, "interrupted": bool(_INTERRUPTED),
               "best_dir": str(run_dir / "best"), "latest_dir": str(run_dir / "latest")}
    print(json.dumps({"grpo_summary": summary}))
    print(f"[grpo][OK] best greedy fidelity={best_fid} (init {init_fid}) step={state.step} "
          f"-> submit {run_dir / 'best'}")
    return 0


def _post_step(policy, processor, opt, gcfg: GRPOConfig, state: TrainerState, resized_model: str,
               run_dir: Path, lstats: LoopStats) -> bool:
    """After an optimizer step: periodic eval + guard-vetoed BEST, atomic ``latest/``, early-stop.

    Returns ``True`` when a stop condition (success plateau / divergence) fires."""
    stop = False
    if state.step % gcfg.eval_every == 0:
        summ = holdout_greedy_eval(policy, processor, gcfg.sft, limit=gcfg.eval_limit)
        summ.update(lstats.snapshot())
        best = is_best(summ, state.best_summary, state.init_summary, gcfg)
        append_eval_log(run_dir, state.step, summ, is_best_flag=best)
        fid = summ.get("behavioral_fidelity_mean")
        best_fid = (state.best_summary or {}).get("behavioral_fidelity_mean")
        if best:
            state.best_summary = summ
            state.evals_since_best = 0
            state.consecutive_bad = 0
            save_best(policy, processor, gcfg, state, resized_model, run_dir)
            print(f"[grpo][eval] step={state.step} greedy_fidelity={fid} -> NEW BEST "
                  f"(kl={summ.get('kl_to_ref'):.4f} collapse={summ.get('collapse_rate')})")
        else:
            state.evals_since_best += 1
            if best_fid is not None and fid is not None and fid < best_fid:
                state.consecutive_bad += 1
            else:
                state.consecutive_bad = 0
            print(f"[grpo][eval] step={state.step} greedy_fidelity={fid} (BEST {best_fid}; "
                  f"evals_since_best={state.evals_since_best})")
        # early-stop / divergence (Doc 05)
        if best_fid is not None and best_fid >= 0.42 and state.evals_since_best >= gcfg.early_stop_patience:
            print(f"[grpo][stop] success plateau: BEST {best_fid} >= 0.42, no new BEST for "
                  f"{state.evals_since_best} evals")
            stop = True
        if state.consecutive_bad >= gcfg.bad_window:
            print(f"[grpo][stop] divergence: {state.consecutive_bad} consecutive evals below BEST")
            stop = True
        _set_mode(policy, generate=False)   # eval flipped use_cache/adapter — restore update mode
    if state.step % gcfg.ckpt_every == 0:
        save_latest(policy, processor, opt, gcfg, state, resized_model, run_dir)
    return stop


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/candidate_grpo.json")
    ap.add_argument("--resized-model", default="models/base_resized")
    ap.add_argument("--out", default="models/sft_adapters")
    ap.add_argument("--run-id", default="grpo_run")
    ap.add_argument("--init-adapter", default=None, help="override the policy init / frozen reference")
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--prompts-per-round", type=int, default=None)
    ap.add_argument("--group-size", type=int, default=None)
    ap.add_argument("--eval-limit", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    args = ap.parse_args(argv)

    try:
        gcfg = load_grpo_config(args.config)
    except Exception as exc:  # noqa: BLE001
        print(f"[grpo][ABORT] bad config {args.config}: {exc}")
        return 1
    overrides = {}
    if args.init_adapter is not None:
        overrides["init_adapter"] = args.init_adapter
    if args.total_steps is not None:
        overrides["total_steps"] = args.total_steps
    if args.prompts_per_round is not None:
        overrides["prompts_per_round"] = args.prompts_per_round
    if args.group_size is not None:
        overrides["group_size"] = args.group_size
    if args.eval_limit is not None:
        overrides["eval_limit"] = args.eval_limit
    if args.eval_every is not None:
        overrides["eval_every"] = args.eval_every
    if args.ckpt_every is not None:
        overrides["ckpt_every"] = args.ckpt_every
    try:
        if overrides:
            gcfg = dataclasses.replace(gcfg, **overrides)   # re-runs GRPOConfig.__post_init__ validation
        return train(gcfg, args.resized_model, args.out, args.run_id)
    except (SFTError, RequiresTokenizer, ValueError) as exc:
        print(f"[grpo][ABORT] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
