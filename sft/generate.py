"""Free-running (autoregressive) code generation for the prompt->LUT VLM.

The single reusable implementation of "run the trained adapter free-running under the
64-code grammar and return the codes it commits to" — shared by :mod:`sft.score_tokens`
(the behavioral-fidelity scorer), the Colab notebooks, and future eval. Previously this
loop lived only inline in ``notebooks/generator_retrain_run.ipynb``.

Unlike the teacher-forced scorer (which feeds the gold prefix every step and cannot see the
exposure-bias collapse), this drives the model from ITS OWN outputs, constrained to the
grammar ``<lut_bos> + 64x<lut_NNN> + <lut_eos>`` (a grade output) OR ``<unsupported>`` (a
refusal). Returns the 64 codebook indices, or ``None`` for a refusal.

Heavy deps (torch, transformers, qwen_vl_utils) are imported lazily so the pure grammar
helper (:func:`make_prefix_fn`) unit-tests without the ``sft`` extra or a GPU.
"""

from __future__ import annotations

from eval.vocab import LUT_BOS, LUT_EOS, UNSUPPORTED, code_token

TOKEN_COUNT = 64
CODEBOOK_SIZE = 256
# BOS + 64 codes + LUT_EOS = 66; +1 model-EOS; +1 headroom.
DEFAULT_MAX_NEW_TOKENS = 68


class SpecialIds:
    """The control/code token ids for a resized tokenizer, plus the id->codebook-index map."""

    def __init__(self, tokenizer):
        self.bos = tokenizer.convert_tokens_to_ids(LUT_BOS)
        self.lut_eos = tokenizer.convert_tokens_to_ids(LUT_EOS)
        self.unsupported = tokenizer.convert_tokens_to_ids(UNSUPPORTED)
        self.model_eos = tokenizer.eos_token_id
        self.codes = [tokenizer.convert_tokens_to_ids(code_token(i)) for i in range(CODEBOOK_SIZE)]
        self.id_to_index = {cid: i for i, cid in enumerate(self.codes)}


def make_prefix_fn(prompt_len: int, ids: SpecialIds):
    """A ``prefix_allowed_tokens_fn`` enforcing the LUT grammar for a batch-size-1 generate.

    Position 0 -> {BOS, UNSUPPORTED}; after UNSUPPORTED -> model EOS; the 64 code positions ->
    the 256 code ids; then LUT_EOS; then model EOS. Identical to the notebook grammar.
    """
    def prefix_fn(_batch_id, input_ids):
        g = input_ids[prompt_len:].tolist()
        if not g:
            return [ids.bos, ids.unsupported]
        if g[0] == ids.unsupported:
            return [ids.model_eos]
        n = len(g) - 1  # code positions emitted so far (excludes the leading BOS)
        if n < TOKEN_COUNT:
            return ids.codes
        if n == TOKEN_COUNT:
            return [ids.lut_eos]
        return [ids.model_eos]
    return prefix_fn


def codes_from_output(output_row, prompt_len: int, ids: SpecialIds) -> list[int] | None:
    """Map a generated token-id row to 64 codebook indices, or ``None`` for a refusal."""
    g = output_row[prompt_len:].tolist() if hasattr(output_row, "tolist") else list(output_row[prompt_len:])
    if ids.unsupported in g:
        return None
    return [ids.id_to_index[t] for t in g if t in ids.id_to_index][:TOKEN_COUNT]


def generate_codes(model, processor, *, image, text: str, sampling: dict | None = None,
                   max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, device=None) -> list[int] | None:
    """Free-running generate for one (image, text) prompt; returns 64 codes or ``None`` (refusal).

    ``sampling`` is ``None`` for greedy (``do_sample=False, num_beams=1``) or a kwargs dict for
    sampling, e.g. ``{"temperature": 0.7, "top_p": 0.9}`` (``do_sample=True`` is added).
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

    gen_kwargs = {"max_new_tokens": max_new_tokens, "prefix_allowed_tokens_fn": make_prefix_fn(plen, ids)}
    if sampling:
        gen_kwargs.update(do_sample=True, **sampling)
    else:
        gen_kwargs.update(do_sample=False, num_beams=1)
    with torch.no_grad():
        out = model.generate(**inp, **gen_kwargs)
    return codes_from_output(out[0], plen, ids)


def generate_codes_for_row(model, processor, row: dict, *, input_field: str = "instruction",
                           bucketize: bool = False, sampling: dict | None = None,
                           max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, device=None) -> list[int] | None:
    """Free-running generate for a corpus row, conditioned exactly as in training.

    Resolves the conditioning text via :func:`sft.example.input_text_for` (honoring ``input_field``
    and ``bucketize`` — which MUST match training) and the image via
    :func:`sft.example.resolve_image`, so the free-running prompt is byte-identical to the trainer's
    prompt half.
    """
    from sft.example import input_text_for, resolve_image

    text = input_text_for(row, input_field, bucketize=bucketize)
    return generate_codes(model, processor, image=resolve_image(row["image_path"]), text=text,
                          sampling=sampling, max_new_tokens=max_new_tokens, device=device)


def generate_codes_batch(model, processor, *, image, text, n: int, sampling: dict, chunk: int = 16,
                         max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, device=None) -> list[list[int] | None]:
    """Free-running generate ``n`` samples for one (image, text) prompt, in chunks of ``chunk``.

    Returns a list of length ``n``; each element is 64 codebook indices, or ``None`` for a refusal
    (``<unsupported>``). ``sampling`` MUST enable sampling (e.g. ``{"temperature":0.7,"top_p":0.9}``);
    greedy with ``n>1`` would return ``n`` identical rows. ``chunk`` bounds peak memory:
    ``ceil(n/chunk)`` ``.generate`` calls, each with ``num_return_sequences <= chunk``. The grammar
    (:func:`make_prefix_fn`) ignores ``batch_id`` and slices on a fixed prompt length, so it stays
    correct under ``num_return_sequences`` expansion.
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

    results: list[list[int] | None] = []
    remaining = int(n)
    with torch.no_grad():
        while remaining > 0:
            k = min(chunk, remaining)
            out = model.generate(**inp, do_sample=True, num_return_sequences=k,
                                 prefix_allowed_tokens_fn=prefix_fn,
                                 max_new_tokens=max_new_tokens, **sampling)
            results.extend(codes_from_output(out[i], plen, ids) for i in range(out.shape[0]))
            remaining -= k
    return results


def generate_codes_for_row_batch(model, processor, row: dict, *, input_field: str = "instruction",
                                 bucketize: bool = False, n: int, sampling: dict, chunk: int = 16,
                                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS, device=None) -> list[list[int] | None]:
    """Batched free-running generate for a corpus row, conditioned exactly as in training.

    Conditioning text comes from :func:`sft.example.input_text_for` (``input_field``/``bucketize`` MUST
    match training); the image from :func:`sft.example.resolve_image`. Scoring against the canonical
    spec is the caller's responsibility (keep conditioning and scoring separate).
    """
    from sft.example import input_text_for, resolve_image

    text = input_text_for(row, input_field, bucketize=bucketize)
    return generate_codes_batch(model, processor, image=resolve_image(row["image_path"]), text=text,
                                n=n, sampling=sampling, chunk=chunk, max_new_tokens=max_new_tokens,
                                device=device)
