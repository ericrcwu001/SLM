"""Dataset + manifest + registry-reconstruction (leakage-safe) tests.

Uses tiny temp .npy files for the Dataset path; exercises the real registry
reconstruction read-only (asserting only fail-closed behavior + coverage shape, not
exact counts, since the demo pool changes).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tokenizer import data as D


def _make_record(tmp_path, key, family="gmic_rawtherapee", tier="gold"):
    arr = (np.random.default_rng(abs(hash(key)) % 2**32).standard_normal((17, 17, 17, 3)) * 0.05)
    p = tmp_path / f"{key}.npy"
    np.save(p, arr)
    return D.LutRecord(residual_key=key, path=str(p), source_family=family, representability_tier=tier)


def test_dataset_yields_conv_tensor(tmp_path):
    recs = [_make_record(tmp_path, f"k{i}") for i in range(5)]
    ds = D.ResidualDataset(recs)
    x, fam = ds[0]
    assert tuple(x.shape) == (3, 17, 17, 17)
    assert isinstance(fam, int) and 0 <= fam < len(ds.families)
    assert len(ds) == 5


def test_manifest_roundtrip(tmp_path):
    recs = [_make_record(tmp_path, f"k{i}") for i in range(4)]
    mpath = str(tmp_path / "train_manifest.jsonl")
    D.write_train_manifest(recs, mpath)
    loaded = D.load_train_manifest(mpath)
    assert len(loaded) == 4
    assert {r.residual_key for r in loaded} == {r.residual_key for r in recs}


def test_manifest_filters_non_train_and_bad_tier(tmp_path):
    good = _make_record(tmp_path, "g", tier="gold")
    rej = _make_record(tmp_path, "r", tier="rejected")
    evalrow = D.LutRecord("e", good.path, "x", "gold", split="eval")
    D.write_train_manifest([good, rej, evalrow], str(tmp_path / "m.jsonl"))
    loaded = D.load_train_manifest(str(tmp_path / "m.jsonl"))
    keys = {r.residual_key for r in loaded}
    assert keys == {"g"}                      # rejected tier + eval split dropped


def test_dev_holdout_deterministic_and_disjoint(tmp_path):
    recs = [_make_record(tmp_path, f"k{i}") for i in range(200)]
    tr1, dv1 = D.dev_holdout(recs, frac=0.1)
    tr2, dv2 = D.dev_holdout(recs, frac=0.1)
    assert {r.residual_key for r in dv1} == {r.residual_key for r in dv2}      # deterministic
    assert not ({r.residual_key for r in tr1} & {r.residual_key for r in dv1})  # disjoint
    assert len(tr1) + len(dv1) == len(recs)
    assert 0 < len(dv1) < len(recs)


def test_family_balanced_sampler_length(tmp_path):
    recs = ([_make_record(tmp_path, f"a{i}", family="A") for i in range(20)]
            + [_make_record(tmp_path, f"b{i}", family="B") for i in range(4)])
    s = D.family_balanced_sampler(recs, num_samples=32)
    assert len(list(iter(s))) == 32


def test_registry_reconstruction_is_fail_closed():
    """Reconstruction from the real registry keeps only confirmed train rows and reports
    coverage; every kept record must be train + accepted tier."""
    import os
    if not os.path.exists("data/splits/split_manifest.json"):
        pytest.skip("no split manifest in repo")
    recs, cov = D.build_records_from_registry(root=".")
    assert set(cov) >= {"residuals_on_disk", "kept", "unresolved_no_row", "wrong_split"}
    assert cov["kept"] == len(recs)
    assert all(r.split == "train" and r.representability_tier in D.ACCEPTED_TIERS for r in recs)
