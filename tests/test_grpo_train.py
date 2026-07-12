"""Anytime-harness logic tests (sft.grpo_train; docs/grpo/04 + 05 'Verification').

Pure logic, no GPU, no model load (mirrors test_build_distillation_corpus's discipline): the atomic
``latest/`` swap, ``save_latest``->``resume_or_init`` round-trip (stubbed adapter save), the
guard-vetoed ``is_best`` selector, the deterministic/resumable data cursor, and the LR schedule.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import sft.grpo_train as gt
from sft.grpo.config import load_grpo_config

_CFG = "configs/candidate_grpo.json"


def _gcfg():
    return load_grpo_config(_CFG)


# --- config round-trip / routing / locks (docs/grpo/04) -------------------------------------------
def test_config_roundtrips_candidate_grpo():
    g = _gcfg()
    assert g.group_size == 8 and g.clip_eps == 0.2 and g.kl_beta == 0.05 and g.total_steps == 500
    assert g.rollout_temperature == 0.7 and g.rollout_top_p == 0.9 and g.prompts_per_round == 8
    assert isinstance(g.grpo_lr, float) and g.grpo_lr == pytest.approx(5e-6)
    assert isinstance(g.adv_eps, float) and g.adv_eps == pytest.approx(1e-4)
    assert g.collapse_penalty == 0.25 and g.delta_e_weight == 0.0


def test_config_sft_locks_intact():
    g = _gcfg()
    assert g.sft.epochs == 2 and g.sft.num_new_tokens == 259 and g.sft.max_seq_len == 1024
    assert g.sft.seed == 0 and g.sft.base_model_id == "Qwen/Qwen2.5-VL-3B-Instruct"
    assert g.sft.bnb_4bit_quant_type == "nf4"
    assert g.sft.input_field == "attribute_spec_text" and g.sft.max_pixels == 200704


def test_config_rejects_unknown_key(tmp_path):
    import json
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"group_size": 8, "bogus_knob": 1}))
    with pytest.raises(ValueError, match="unknown keys"):
        load_grpo_config(str(p))


def test_config_shared_name_routes_to_grpo(tmp_path):
    import json
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"ckpt_every": 5}))       # a name on BOTH dataclasses
    g = load_grpo_config(str(p))
    assert g.ckpt_every == 5           # GRPO gets the JSON value (the loop reads gcfg.ckpt_every)
    assert g.sft.ckpt_every == 200     # SFTConfig keeps its default; never double-populated


# --- atomic latest/ swap --------------------------------------------------------------------------
def test_atomic_write_dir_survives_repeated_writes(tmp_path):
    dst = tmp_path / "latest"
    gt._atomic_write_dir(lambda t: (t / "f.txt").write_text("v1"), dst)
    assert (dst / "f.txt").read_text() == "v1"
    # a SECOND write over the now-populated dir must not crash (the ENOTEMPTY trap) and must swap in
    gt._atomic_write_dir(lambda t: (t / "f.txt").write_text("v2"), dst)
    assert (dst / "f.txt").read_text() == "v2"
    assert not dst.with_name("latest.tmp").exists()
    assert not dst.with_name("latest.old").exists()


# --- save_latest -> resume_or_init round-trip -----------------------------------------------------
class _StubOpt:
    def __init__(self):
        self._sd = {"state": {}, "param_groups": [{"lr": 0.1}]}

    def state_dict(self):
        return self._sd

    def load_state_dict(self, sd):
        self._sd = sd


def test_save_latest_resume_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(gt, "_write_adapter", lambda *a, **k: "stub_sha")   # skip the real adapter save
    gcfg = _gcfg()
    run_dir = tmp_path / "grpo_x"
    run_dir.mkdir()
    state = gt.TrainerState(step=7, round=2, data_cursor=13,
                            best_summary={"behavioral_fidelity_mean": 0.30},
                            init_summary={"behavioral_fidelity_mean": 0.16},
                            evals_since_best=1, consecutive_bad=0)

    gt.save_latest(None, None, _StubOpt(), gcfg, state, "models/base_resized", run_dir)

    assert (run_dir / "latest" / "trainer_state.pt").exists()
    resuming, policy_init = gt.resume_or_init(run_dir, gcfg)
    assert resuming is True
    assert policy_init.endswith("latest")

    ts = torch.load(run_dir / "latest" / "trainer_state.pt", map_location="cpu", weights_only=False)
    assert ts["step"] == 7 and ts["round"] == 2 and ts["data_cursor"] == 13
    assert ts["best_summary"]["behavioral_fidelity_mean"] == 0.30
    assert ts["init_summary"]["behavioral_fidelity_mean"] == 0.16
    assert "rng" in ts and "torch" in ts["rng"]


def test_resume_or_init_fresh_uses_p6(tmp_path):
    gcfg = _gcfg()
    resuming, policy_init = gt.resume_or_init(tmp_path / "fresh_run", gcfg)
    assert resuming is False
    assert policy_init == gcfg.init_adapter        # fresh -> P6, never latest


# --- guard-vetoed BEST selector -------------------------------------------------------------------
def test_is_best_first_eval_promotes_baseline():
    gcfg = _gcfg()
    init = {"behavioral_fidelity_mean": 0.16, "collapse_rate": 0.9, "code_entropy_norm_mean": 0.6}
    assert gt.is_best(init, best_summary=None, init_summary=None, gcfg=gcfg) is True


def test_is_best_promotes_healthy_improvement():
    gcfg = _gcfg()
    init = {"behavioral_fidelity_mean": 0.16, "collapse_rate": 0.9, "code_entropy_norm_mean": 0.6,
            "decoded_delta_e_mean": 5.0}
    best = dict(init)
    new = {"behavioral_fidelity_mean": 0.30, "collapse_rate": 0.9, "code_entropy_norm_mean": 0.6,
           "decoded_delta_e_mean": 5.0, "degenerate_rate": 0.0, "kl_to_ref": 0.1}
    assert gt.is_best(new, best, init, gcfg) is True


def test_is_best_requires_new_high():
    gcfg = _gcfg()
    best = {"behavioral_fidelity_mean": 0.30}
    new = {"behavioral_fidelity_mean": 0.16, "collapse_rate": 0.0}
    assert gt.is_best(new, best, best, gcfg) is False


@pytest.mark.parametrize("veto", [
    {"collapse_rate": 0.65},                       # init 0.50 + margin 0.10 = 0.60 -> 0.65 vetoes
    {"degenerate_rate": 0.05},                     # > 0.02 ceiling
    {"code_entropy_norm_mean": 0.25},              # < 0.5 * init(0.60) = 0.30
    {"decoded_delta_e_mean": 3.5},                 # > best(2.0) + margin(1.0) = 3.0
    {"kl_to_ref": 15.0},                           # > 10.0 ceiling
])
def test_is_best_vetoes_reward_hacking(veto):
    gcfg = _gcfg()
    init = {"behavioral_fidelity_mean": 0.16, "collapse_rate": 0.50, "code_entropy_norm_mean": 0.60,
            "decoded_delta_e_mean": 2.0}
    best = dict(init)
    new = {"behavioral_fidelity_mean": 0.30, "collapse_rate": 0.50, "code_entropy_norm_mean": 0.60,
           "decoded_delta_e_mean": 2.0, "degenerate_rate": 0.0, "kl_to_ref": 0.1}
    new.update(veto)
    assert gt.is_best(new, best, init, gcfg) is False, f"guard should veto {veto}"


# --- deterministic / resumable data cursor --------------------------------------------------------
def test_next_prompts_deterministic():
    rows = [{"id": i} for i in range(10)]
    a = gt.next_prompts(rows, gt.TrainerState(), 4, seed=0)
    b = gt.next_prompts(rows, gt.TrainerState(), 4, seed=0)
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_next_prompts_resume_from_cursor():
    rows = [{"id": i} for i in range(10)]
    s = gt.TrainerState()
    first = gt.next_prompts(rows, s, 4, seed=0)          # advances cursor to 4
    cont = gt.next_prompts(rows, s, 4, seed=0)           # 4..7 of epoch 0
    resumed = gt.next_prompts(rows, gt.TrainerState(data_cursor=4), 4, seed=0)
    assert [r["id"] for r in cont] == [r["id"] for r in resumed]
    assert first != cont


def test_next_prompts_reshuffles_across_epoch():
    rows = [{"id": i} for i in range(6)]
    s = gt.TrainerState()
    ep0 = [r["id"] for r in gt.next_prompts(rows, s, 6, seed=0)]     # full epoch 0
    ep1 = [r["id"] for r in gt.next_prompts(rows, s, 6, seed=0)]     # full epoch 1
    assert sorted(ep0) == list(range(6)) and sorted(ep1) == list(range(6))   # permutations
    assert ep0 != ep1                                                        # reshuffled per epoch


def test_epoch_order_is_permutation():
    order = gt._epoch_order(20, seed=3, epoch=2)
    assert sorted(order) == list(range(20))


# --- LR schedule (linear warmup -> flat) ----------------------------------------------------------
def test_lr_warmup_then_flat():
    gcfg = _gcfg()          # warmup_steps=10, grpo_lr=5e-6
    assert gt._lr(0, gcfg) == pytest.approx(gcfg.grpo_lr * 1 / 10)
    assert gt._lr(9, gcfg) == pytest.approx(gcfg.grpo_lr)
    assert gt._lr(50, gcfg) == pytest.approx(gcfg.grpo_lr)          # flat after warmup


# --- signal flag causes a clean stop --------------------------------------------------------------
def test_signal_handler_sets_flag(monkeypatch):
    import signal as _sig

    orig_int, orig_term = _sig.getsignal(_sig.SIGINT), _sig.getsignal(_sig.SIGTERM)
    monkeypatch.setattr(gt, "_INTERRUPTED", False)
    try:
        gt.install_signal_handlers()
        handler = _sig.getsignal(_sig.SIGINT)
        handler(_sig.SIGINT, None)                  # first signal -> flag, NOT KeyboardInterrupt
        assert gt._INTERRUPTED is True
        with pytest.raises(KeyboardInterrupt):      # second signal -> hard exit escape hatch
            handler(_sig.SIGINT, None)
    finally:
        _sig.signal(_sig.SIGINT, orig_int)
        _sig.signal(_sig.SIGTERM, orig_term)
