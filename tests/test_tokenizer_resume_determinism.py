"""Resume must continue the SAME trajectory as an uninterrupted run.

Guards the RNG-restore fix: train.py now restores torch+numpy(+cuda) RNG on resume and
seeds only on a fresh start, so an interrupted-then-resumed run reproduces the
uninterrupted run bit-for-bit. Without the fix (reseed on resume) the sampler restarts
and the two diverge. Uses tiny synthetic data, CPU, a handful of steps; LR is held
constant (no warmup/decay) so the only thing under test is RNG continuity.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from dataclasses import replace

from tokenizer import data as D
from tokenizer import freeze as F
from tokenizer.config import TokenizerConfig
from tokenizer.train import train


def _records(tmp_path, n=8):
    recs = []
    for i in range(n):
        arr = np.random.default_rng(i).standard_normal((17, 17, 17, 3)) * 0.05
        p = tmp_path / f"k{i}.npy"
        np.save(p, arr)
        recs.append(D.LutRecord(f"k{i}", str(p), "gmic_rawtherapee", "gold"))
    return recs


def test_resume_matches_uninterrupted_run(tmp_path):
    recs = _records(tmp_path, 8)
    base = dict(max_steps=4, batch_size=4, warmup_steps=0, lr_decay=False,
                eval_every=0, ckpt_every=0, keep_last=1)

    # (A) uninterrupted 4-step run
    cfg4 = replace(TokenizerConfig(), **base)
    out_a = str(tmp_path / "a")
    ck_a = train(cfg4, recs, out_dir=out_a, device="cpu", log_fn=lambda *_: None)

    # (B) 2 steps, checkpoint, then resume for the remaining 2 (same constant-LR schedule)
    out_b = str(tmp_path / "b")
    cfg2 = replace(TokenizerConfig(), **{**base, "max_steps": 2})
    ck_b2 = train(cfg2, recs, out_dir=out_b, device="cpu", log_fn=lambda *_: None)
    ck_b = train(cfg4, recs, out_dir=out_b, device="cpu", resume=ck_b2, log_fn=lambda *_: None)

    ma, _, _ = F.load_model_from_checkpoint(ck_a)
    mb, _, _ = F.load_model_from_checkpoint(ck_b)

    assert torch.allclose(ma.vq.codebook, mb.vq.codebook, atol=1e-6), "codebook diverged on resume"
    for (na, pa), (nb, pb) in zip(ma.decoder.state_dict().items(), mb.decoder.state_dict().items()):
        assert torch.allclose(pa, pb, atol=1e-6), f"decoder weight {na} diverged on resume"
