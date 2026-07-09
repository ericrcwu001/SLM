"""Tests for the source inventory loader against the real configs/source_inventory.yaml."""

import pytest

from data_pipeline.source_inventory import load_source_inventory


def test_loads_real_inventory():
    inv = load_source_inventory("configs/source_inventory.yaml")
    assert inv.version == "source_inventory_v1"
    ids = {s.source_pack_id for s in inv.sources}
    assert {"ppr10k_expert_abc", "fivek_expert_abcde", "freshluts_public",
            "gmic_rawtherapee_haldclut", "public_lut_packs_misc",
            "procedural_fillers_v1"} <= ids


def test_validate_clean():
    inv = load_source_inventory("configs/source_inventory.yaml")
    assert inv.validate() == []


def test_priority_order():
    inv = load_source_inventory("configs/source_inventory.yaml")
    ordered = inv.by_priority()
    assert ordered[0].source_pack_id == "ppr10k_expert_abc"
    assert [s.priority for s in ordered] == sorted(s.priority for s in inv.sources)


def test_excluded_families_rejected():
    inv = load_source_inventory("configs/source_inventory.yaml")
    excl = inv.excluded_families()
    assert {"dped", "hdr_plus_isp", "camera_log_unknown_domain"} <= excl
    with pytest.raises(ValueError):
        inv.assert_not_excluded("dped")
    # a legitimate family passes
    inv.assert_not_excluded("fresh_luts")


def test_get_and_access_method():
    inv = load_source_inventory("configs/source_inventory.yaml")
    proc = inv.get("procedural_fillers_v1")
    assert proc.access_method == "generated"
    assert proc.family == "controlled_procedural"
