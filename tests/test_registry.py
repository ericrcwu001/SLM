"""Tests for the provenance registry: schema round-trip, validation, removal scope."""

from data_pipeline.constants import CANONICAL_DOMAIN_ID
from data_pipeline.registry import (
    ProvenanceRow,
    RegistryStore,
    removal_manifest,
    validate_row,
)


def _good_row(**kw) -> ProvenanceRow:
    base = dict(
        source_family="gmic_rawtherapee",
        source_pack_id="gmic_rawtherapee_haldclut",
        file_hash="abc123",
        canonical_domain_id=CANONICAL_DOMAIN_ID,
    )
    base.update(kw)
    return ProvenanceRow(**base)


def test_roundtrip_dict():
    row = _good_row(lut_id="rt_001", fit_deltaE00_mean=1.2, normalization_warnings=["clip"])
    d = row.to_dict()
    back = ProvenanceRow.from_dict(d)
    assert back == row
    assert back.fit_deltaE00_mean == 1.2
    assert back.normalization_warnings == ["clip"]


def test_from_dict_ignores_unknown_keys():
    row = ProvenanceRow.from_dict({"source_family": "x", "not_a_field": 1})
    assert row.source_family == "x"


def test_validate_missing_required():
    row = ProvenanceRow(source_family="x")  # missing pack_id, file_hash, domain
    errs = validate_row(row)
    assert any("missing_required:source_pack_id" in e for e in errs)
    assert any("missing_required:file_hash" in e for e in errs)
    assert any("missing_required:canonical_domain_id" in e for e in errs)


def test_validate_good_row_passes():
    assert validate_row(_good_row()) == []


def test_validate_bad_domain_and_headline_tier():
    row = _good_row(canonical_domain_id="wrong_domain")
    assert any("bad_canonical_domain_id" in e for e in validate_row(row))
    row2 = _good_row(headline_eligible=True, representability_tier="diagnostic_only")
    assert any("headline_row_tier" in e for e in validate_row(row2))


def test_registry_store_roundtrip(tmp_path):
    store = RegistryStore(tmp_path / "reg")
    rows = [_good_row(lut_id=f"l{i}") for i in range(3)]
    store.write_all(rows)
    loaded = store.load()
    assert len(loaded) == 3
    assert {r.lut_id for r in loaded} == {"l0", "l1", "l2"}
    store.add(_good_row(lut_id="l3"))
    assert len(store.load()) == 4


def test_removal_manifest_invalidation_scope():
    rows = [
        _good_row(source_family="fivek_derived", used_for_tokenizer=True, file_hash="h1"),
        _good_row(source_family="fivek_derived", used_for_eval=True, file_hash="h2"),
        _good_row(source_family="fresh_luts", used_for_sft=True, file_hash="h3"),
    ]
    man = removal_manifest(rows, "fivek_derived")
    assert man["affected_row_count"] == 2
    assert man["invalidates"]["tokenizer"] is True
    assert man["invalidates"]["eval"] is True
    assert man["invalidates"]["sft"] is False
    assert any("retrain_tokenizer" in a for a in man["required_actions"])
    assert set(man["affected_file_hashes"]) == {"h1", "h2"}
