"""Tests for cross-split leakage detection."""

import numpy as np

from data_pipeline.leakage import (
    LeakageItem,
    dct_phash,
    hamming,
    jaccard,
    leakage_report,
    word_3grams,
)


def test_clean_rows_pass():
    items = [
        LeakageItem(id="a", split="train", lut_hash="h1", image_hash="i1", prompt_text="warm faded"),
        LeakageItem(id="b", split="eval", lut_hash="h2", image_hash="i2", prompt_text="cool crisp night"),
    ]
    rep = leakage_report(items)
    assert rep.status == "pass"
    assert rep.leakage_report_hash


def test_exact_lut_hash_cross_split_fails():
    items = [
        LeakageItem(id="a", split="train", lut_hash="dup"),
        LeakageItem(id="b", split="eval", lut_hash="dup"),
    ]
    rep = leakage_report(items)
    assert rep.status == "fail"
    assert rep.per_axis_violations.get("lut_behavior", 0) >= 1


def test_phash_near_dup_cross_split_fails():
    # a structured image (not flat) so the pHash is stable under tiny perturbations
    yy, xx = np.mgrid[0:64, 0:64]
    base = 0.5 + 0.4 * np.sin(xx / 5.0) * np.cos(yy / 7.0)
    rng = np.random.default_rng(0)
    near = np.clip(base + rng.normal(scale=0.002, size=base.shape), 0, 1)
    items = [
        LeakageItem(id="a", split="train", phash=dct_phash(base)),
        LeakageItem(id="b", split="eval", phash=dct_phash(near)),
    ]
    assert hamming(items[0].phash, items[1].phash) <= 6
    assert leakage_report(items).status == "fail"


def test_pca_lut_near_dup_cross_split_fails():
    rng = np.random.default_rng(0)
    train_res = [rng.normal(size=48) for _ in range(6)]
    special = rng.normal(size=48) * 3.0
    items = [LeakageItem(id=f"t{i}", split="train", residual_vec=train_res[i]) for i in range(6)]
    items.append(LeakageItem(id="t_special", split="train", residual_vec=special))
    items.append(LeakageItem(id="e_special", split="eval", residual_vec=special + 1e-4))
    rep = leakage_report(items)
    assert rep.status == "fail"
    assert rep.per_axis_violations.get("lut_behavior", 0) >= 1


def test_semantic_axes_skipped_without_embeddings():
    items = [
        LeakageItem(id="a", split="train", image_hash="i1", prompt_text="x y z"),
        LeakageItem(id="b", split="eval", image_hash="i2", prompt_text="p q r"),
    ]
    rep = leakage_report(items)
    assert any("image_semantics" in s for s in rep.skipped_axes)
    assert any("prompt_semantics" in s for s in rep.skipped_axes)


def test_prompt_lexical_jaccard():
    a = word_3grams("make it warmer and more muted please")
    b = word_3grams("make it warmer and more muted now")
    assert jaccard(a, b) > 0.4
