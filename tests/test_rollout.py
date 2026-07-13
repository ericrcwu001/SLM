"""Rollout + logprob-extraction tests (sft.rollout; docs/grpo/02_rollout.md 'Verification').

The grammar-masked teacher-forced ``code_logprobs`` and the buffer types are tested with a fake
deterministic model + the ``_FakeTokenizer`` from ``tests/test_generate.py`` — no GPU, no frozen
weights, no transformers model load (that path runs on Colab). The ``.generate``/``rollout_row`` path
itself needs the QLoRA stack and is exercised by the Colab smoke.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from sft.rollout import (
    RolloutGroup,
    RolloutSample,
    _mean_code_entropy,
    assistant_target_from_codes,
    code_logprobs,
    init_code_maps,
)
from tests.test_generate import _FakeTokenizer

_N_PROMPT = 4
_CODES = list(range(64))
# prompt tokens (small ids) + <lut_bos>(1000) + 64 code tokens (2000+c) + <lut_eos>(1001)
_SEQ = [11, 12, 13, 14][:_N_PROMPT] + [1000] + [2000 + c for c in _CODES] + [1001]


class _OracleModel(torch.nn.Module):
    """logits[:, t] one-hots input_ids[:, t+1], so greedy reproduces the sequence exactly.

    Lets us assert the ``n_prompt+j`` alignment: argmax of the grammar-masked step distribution equals
    the emitted code at every one of the 64 positions."""

    def __init__(self, vocab: int):
        super().__init__()
        self.vocab = vocab
        self.device = torch.device("cpu")

    def forward(self, *, input_ids, attention_mask=None, **kw):
        B, L = input_ids.shape
        logits = torch.full((B, L, self.vocab), -10.0)
        nxt = input_ids[:, 1:]                                  # token predicted at position t
        logits[:, :-1, :].scatter_(2, nxt.unsqueeze(-1), 10.0)
        return SimpleNamespace(logits=logits)


def _batch(seq=_SEQ, n_prompt=_N_PROMPT):
    input_ids = torch.tensor([seq])
    labels = input_ids.clone()
    labels[:, :n_prompt] = -100
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": labels}


def test_code_logprobs_alignment_and_mask():
    init_code_maps(_FakeTokenizer())
    model = _OracleModel(vocab=2300)
    batch = _batch()
    logp, sel = code_logprobs(model, batch)

    assert logp.shape == (1, len(_SEQ) - 1)
    assert int(sel.sum()) == 64                                # exactly the 64 code positions
    # off-span logprobs are masked to 0 (BOS/EOS/prompt)
    assert float(logp.masked_select(~sel).abs().sum()) == 0.0
    # alignment: the argmax over the 256 legal code columns equals the emitted code at every position
    from sft.rollout import _forward_code_logp
    logp_full, gidx, sel2 = _forward_code_logp(model, batch)
    assert torch.equal(sel, sel2)
    assert torch.equal(logp_full.argmax(-1)[sel2], gidx[sel2])


def test_code_logprobs_deterministic_and_teacher_forced():
    """Same sequence -> identical logprobs (the contract the update relies on for ρ≡1)."""
    init_code_maps(_FakeTokenizer())
    model = _OracleModel(vocab=2300)
    batch = _batch()
    a, _ = code_logprobs(model, batch)
    b, _ = code_logprobs(model, batch)
    assert torch.equal(a, b)
    # per-token logprob is a valid log-prob (<= 0) on the code span
    assert bool((a[b == b] <= 1e-6).all())


def test_batched_logprobs_equal_per_sample():
    """The batched old/ref forward (rollout_row's hot path) must equal per-sample B=1 forwards — the
    correctness contract behind batching the logprob passes for GPU efficiency."""
    init_code_maps(_FakeTokenizer())
    m = _OracleModel(vocab=2300)
    seq2 = [11, 12, 13, 14][:_N_PROMPT] + [1000] + [2000 + ((c + 7) % 256) for c in _CODES] + [1001]
    b1, b2 = _batch(), _batch(seq=seq2)
    batch = {k: torch.cat([b1[k], b2[k]], 0) for k in ("input_ids", "attention_mask", "labels")}
    lp, sel = code_logprobs(m, batch)
    lp1, _ = code_logprobs(m, b1)
    lp2, _ = code_logprobs(m, b2)
    assert torch.allclose(lp[0], lp1[0]) and torch.allclose(lp[1], lp2[0])
    assert int(sel.sum()) == 128            # 64 code positions per row, 2 rows


def test_rollout_entropy_positive_and_bounded():
    init_code_maps(_FakeTokenizer())
    model = _OracleModel(vocab=2300)
    from sft.rollout import _forward_code_logp
    logp_full, _gidx, sel = _forward_code_logp(model, _batch())
    ent = _mean_code_entropy(logp_full, sel)
    import math
    assert 0.0 <= ent <= math.log(256) + 1e-6                  # entropy over 256 codes, in nats


def test_init_code_maps_idempotent_keeps_cache():
    """Re-init with the SAME tokenizer must NOT clear the per-device tensor cache (called per row)."""
    import sft.rollout as R
    R.init_code_maps(_FakeTokenizer())
    R._TENSOR_CACHE["__sentinel__"] = ("x", "y")
    R.init_code_maps(_FakeTokenizer())                 # same code ids -> no-op
    assert "__sentinel__" in R._TENSOR_CACHE

    class _Shifted(_FakeTokenizer):
        def convert_tokens_to_ids(self, tok):
            return super().convert_tokens_to_ids(tok) + 1000   # different code ids

    R.init_code_maps(_Shifted())                       # changed -> clears cache
    assert "__sentinel__" not in R._TENSOR_CACHE
    R.init_code_maps(_FakeTokenizer())                 # restore for the other tests


def test_assistant_target_matches_materializer():
    from scripts.materialize_target_tokens import _assistant_target
    codes = [0, 1, 255, 42] + list(range(60))
    assert assistant_target_from_codes(codes) == _assistant_target(codes)


# --- buffer types ---------------------------------------------------------------------------------
def _valid_sample(row_id, code0, adv, *, L=6, P=2, D=3):
    """A gradable RolloutSample with a tiny fake teacher-forced example + old/ref logprobs."""
    ex = {
        "input_ids": torch.tensor([[1, 2, 3, 1000, 2000 + code0, 1001]]),
        "attention_mask": torch.ones(1, L, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, -100, -100, 2000 + code0, 1001]]),
        "pixel_values": torch.zeros(P, D),
        "image_grid_thw": torch.tensor([[1, 1, 1]]),
    }
    s = RolloutSample(row_id=row_id, cond_text="c", spec_text="route=grade | warmer=+2.0",
                      codes=[code0] * 64, refused=False, valid64=True, n_prompt=4, example=ex,
                      entropy=1.0)
    s.old_logprobs = torch.full((L - 1,), -0.1)
    s.ref_logprobs = torch.full((L - 1,), -0.2)
    s.reward = 0.5
    s.advantage = adv
    return s


def test_rolloutgroup_assigns_and_filters_gradable():
    valid = _valid_sample("r", 5, adv=1.0)
    refusal = RolloutSample(row_id="r", cond_text="c", spec_text="s", codes=None, refused=True,
                            valid64=False)
    none_adv = _valid_sample("r", 7, adv=None)           # valid-64 but excluded (None advantage)
    samples = [valid, refusal, none_adv]
    rewards = [(0.5, {}), (0.0, {"refused": True}), (None, {})]
    adv = [1.0, -0.7, None]
    g = RolloutGroup(samples, rewards, adv)

    assert g.row_id == "r"
    assert valid.reward == 0.5 and valid.advantage == 1.0
    assert refusal.advantage == -0.7                     # assigned, but not gradable (not valid64)
    grad = g.gradable()
    assert grad == [valid]                               # refusal + None-advantage excluded
    assert g.has_grad()
    assert g.refusal_rate == pytest.approx(1 / 3)
    assert g.entropy_mean == pytest.approx(1.0)          # only the valid sample carries entropy


def test_rolloutgroup_build_stacks_gradable_samples():
    init_code_maps(_FakeTokenizer())
    s1 = _valid_sample("r", 5, adv=1.0)
    s2 = _valid_sample("r", 9, adv=-0.5)
    g = RolloutGroup([s1, s2], [(0.6, {}), (0.4, {})], [1.0, -0.5])
    batch, old_lp, ref_lp, adv_t = g.build(torch.device("cpu"))

    assert batch["input_ids"].shape == (2, 6)            # B=2 stacked, same length
    assert batch["pixel_values"].shape == (4, 3)         # 2 x [2,3] image copies concatenated
    assert batch["image_grid_thw"].shape == (2, 3)
    assert old_lp.shape == (2, 5) and ref_lp.shape == (2, 5)
    assert adv_t.shape == (2, 1)
    assert adv_t.flatten().tolist() == pytest.approx([1.0, -0.5])
    # the stacked batch runs through code_logprobs (grammar-masked over the fake vocab)
    logp, sel = code_logprobs(_OracleModel(2300), batch)
    assert logp.shape == (2, 5)
    assert int(sel.sum()) == 2                            # one code position per row here


def test_rolloutsample_accepts_logprobs_in_constructor():
    """old_logprobs/ref_logprobs MUST be real dataclass fields (constructor kwargs), the exact call
    rollout_row makes — guards the 'unannotated field becomes a class attribute' trap."""
    s = RolloutSample(row_id="r", cond_text="c", spec_text="s", codes=list(range(64)), refused=False,
                      valid64=True, n_prompt=4, example={}, old_logprobs=torch.zeros(5),
                      ref_logprobs=torch.zeros(5), entropy=0.5)
    assert s.old_logprobs is not None and s.ref_logprobs is not None
    # distinct instances must not share the tensors (no class-attribute aliasing)
    s2 = RolloutSample(row_id="r2", cond_text="c", spec_text="s", codes=None, refused=True,
                       valid64=False)
    assert s2.old_logprobs is None and s2.ref_logprobs is None


def test_rolloutgroup_no_grad_when_all_refused():
    refusals = [RolloutSample(row_id="r", cond_text="c", spec_text="s", codes=None, refused=True,
                              valid64=False) for _ in range(3)]
    g = RolloutGroup(refusals, [(0.0, {"refused": True})] * 3, [0.0, 0.0, 0.0])
    assert not g.has_grad()
    assert g.gradable() == []
