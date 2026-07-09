"""Tests for deterministic split-unit assignment + eval reservation."""

from data_pipeline.splits import SplitCandidate, build_split_manifest


def test_same_group_co_assigned():
    cands = [
        SplitCandidate(id="r1", base_key="group_7", lut_hash="a"),
        SplitCandidate(id="r2", base_key="group_7", lut_hash="b"),
    ]
    m = build_split_manifest(cands)
    u1 = m.assignments["r1"]["split_unit_id"]
    u2 = m.assignments["r2"]["split_unit_id"]
    assert u1 == u2
    assert m.assignments["r1"]["split"] == m.assignments["r2"]["split"]


def test_procedural_forced_train():
    cands = [SplitCandidate(id=f"p{i}", base_key=f"proc_{i}", procedural=True) for i in range(20)]
    m = build_split_manifest(cands)
    assert all(a["split"] == "train" for a in m.assignments.values())


def test_eval_reservation_and_determinism():
    cands = [SplitCandidate(id=f"r{i}", base_key=f"k{i}", lut_hash=f"h{i}") for i in range(300)]
    m1 = build_split_manifest(cands, seed=1234)
    m2 = build_split_manifest(cands, seed=1234)
    splits = {a["split"] for a in m1.assignments.values()}
    assert {"train", "eval"} <= splits            # eval reserved
    assert "diagnostic" in splits or "qualitative" in splits
    assert m1.split_id == m2.split_id             # deterministic


def test_duplicate_lut_coassigned_leakage_clean():
    # two rows with the same LUT hash must be co-located -> leakage stays clean
    cands = [
        SplitCandidate(id="r1", base_key="g1", lut_hash="dup"),
        SplitCandidate(id="r2", base_key="g2", lut_hash="dup"),
    ] + [SplitCandidate(id=f"r{i}", base_key=f"k{i}", lut_hash=f"h{i}") for i in range(50)]
    m = build_split_manifest(cands)
    assert m.assignments["r1"]["split"] == m.assignments["r2"]["split"]
    assert m.leakage_status == "pass"
