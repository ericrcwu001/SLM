"""Dependency-free tests for the vocab-resize math + SFT config.

These exercise the invariants the (heavy) sft/vocab_resize.py preflight will assert, using
eval.vocab.SpecialVocab directly with a realistic base offset — no torch/transformers needed.
The model-side preflight (row counts, only-259-trainable, roundtrip) lives in
tests/test_sft_preflight_smoke.py behind importorskip.
"""

from eval.vocab import (
    CODEBOOK_SIZE,
    LUT_BOS,
    LUT_EOS,
    NUM_SPECIAL_TOKENS,
    UNSUPPORTED,
    SpecialVocab,
    code_index,
    code_token,
)
from sft.config import DEFAULT_CONFIG, SFTConfig

# A plausible base-tokenizer length for Qwen2.5-VL (exact value is irrelevant to the math).
BASE_LEN = 151_000


def test_special_vocab_count_and_contiguity():
    v = SpecialVocab(base_offset=BASE_LEN)
    assert v.vocab_size == NUM_SPECIAL_TOKENS == 259
    ids = [v.token_to_id[t] for t in v.all_tokens]
    # contiguous block [BASE_LEN, BASE_LEN+259), unique, no gaps
    assert sorted(ids) == list(range(BASE_LEN, BASE_LEN + NUM_SPECIAL_TOKENS))
    assert len(set(ids)) == NUM_SPECIAL_TOKENS


def test_code_token_identity_mapping():
    v = SpecialVocab(base_offset=BASE_LEN)
    # token suffix "kkk" <-> codebook index k is identity; code ids are contiguous after the 3 controls
    for k in range(CODEBOOK_SIZE):
        assert code_index(code_token(k)) == k
        assert v.code_id(k) == BASE_LEN + 3 + k
    assert v.token_suffix_to_codebook_index() == "identity"


def test_control_tokens_distinct_from_codes():
    v = SpecialVocab(base_offset=BASE_LEN)
    controls = {v.bos_id, v.eos_id, v.unsupported_id}
    assert len(controls) == 3
    assert controls.isdisjoint(set(v.code_ids))
    for t in (LUT_BOS, LUT_EOS, UNSUPPORTED):
        assert code_index(t) is None  # a control token is not a code token


def test_added_special_token_ids_shape():
    v = SpecialVocab(base_offset=BASE_LEN)
    added = v.added_special_token_ids
    assert len(added) == NUM_SPECIAL_TOKENS
    assert added[LUT_BOS] == BASE_LEN and added[code_token(255)] == BASE_LEN + 258


def test_sft_config_defaults_and_invariants():
    c = DEFAULT_CONFIG
    assert c.epochs == 2
    assert c.per_device_batch_size * c.gradient_accumulation_steps == c.effective_batch_size
    assert c.num_new_tokens == NUM_SPECIAL_TOKENS
    assert set(c.lora_target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj",
                                          "gate_proj", "up_proj", "down_proj"}


def test_sft_config_rejects_epoch_change():
    import pytest
    with pytest.raises(ValueError):
        SFTConfig(epochs=1)
    with pytest.raises(ValueError):  # pdb*accum must equal effective batch
        SFTConfig(per_device_batch_size=2, gradient_accumulation_steps=32, effective_batch_size=32)


# --- vocab-resize preflight gate (pure logic; the model path is Colab-only) ---
def _synthetic_ids(base_len):
    new_ids = list(range(base_len, base_len + NUM_SPECIAL_TOKENS))  # 3 controls then 256 codes
    code_ids = new_ids[3:]  # <lut_000..255> follow the 3 control tokens, contiguous
    return new_ids, code_ids


def test_preflight_passes_on_correct_resize():
    from sft.vocab_resize import preflight_checks
    new_ids, code_ids = _synthetic_ids(BASE_LEN)
    n_tok = BASE_LEN + NUM_SPECIAL_TOKENS
    rep = preflight_checks(base_len=BASE_LEN, n_tok=n_tok, n_in=n_tok, n_out=n_tok,
                           code_ids=code_ids, new_ids=new_ids, tied=True)
    assert rep["all_pass"] is True
    assert rep["tied_embedding_status"] == "tied"


def test_preflight_fails_on_head_mismatch():
    from sft.vocab_resize import preflight_checks
    new_ids, code_ids = _synthetic_ids(BASE_LEN)
    n_tok = BASE_LEN + NUM_SPECIAL_TOKENS
    # lm_head not resized to match (n_out stale) -> gate must fail, not silently pass
    rep = preflight_checks(base_len=BASE_LEN, n_tok=n_tok, n_in=n_tok, n_out=BASE_LEN,
                           code_ids=code_ids, new_ids=new_ids, tied=False)
    assert rep["all_pass"] is False
    assert rep["len_tok_eq_embed_eq_head"] is False


def test_preflight_fails_on_noncontiguous_codes():
    from sft.vocab_resize import preflight_checks
    new_ids, code_ids = _synthetic_ids(BASE_LEN)
    code_ids = code_ids[:]              # break contiguity
    code_ids[100] = code_ids[100] + 9999
    n_tok = BASE_LEN + NUM_SPECIAL_TOKENS
    rep = preflight_checks(base_len=BASE_LEN, n_tok=n_tok, n_in=n_tok, n_out=n_tok,
                           code_ids=code_ids, new_ids=new_ids, tied=True)
    assert rep["all_pass"] is False
    assert rep["code_tokens_contiguous"] is False
