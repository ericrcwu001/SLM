"""Unified tag vocabulary sync (ADR 0022; docs/attribute_spec.md §10).

Asserts the ONE source of truth (:mod:`eval.tag_vocabulary`) is what
``data_pipeline.instruction_gen`` and ``eval.frontier_scoring`` actually use — the divergent
tag sets are reconciled and the aliases retired everywhere in code.
"""

from __future__ import annotations

from data_pipeline import instruction_gen as ig
from eval import frontier_scoring as fs
from eval import tag_vocabulary as tv


def test_instruction_gen_sources_the_unified_vocabulary():
    assert ig._TAG_BEHAVIOR == dict(tv.DIRECTIONAL_TAG_AXIS)
    assert ig._STYLE_TAGS == set(tv.STYLE_TAGS)


def test_frontier_directions_match_unified_vocabulary():
    # frontier's TAG_DIRECTIONS is derived from the same table (canonical tags only, no aliases).
    assert set(fs.TAG_DIRECTIONS) == set(tv.DIRECTIONAL_TAG_AXIS)
    for tag, (axis, sign) in tv.DIRECTIONAL_TAG_AXIS.items():
        checks = fs.TAG_DIRECTIONS[tag]
        assert len(checks) == 1
        f_axis, f_sign, f_mag = checks[0]
        assert (f_axis, f_sign) == (axis, sign)
        assert f_mag == tv.min_magnitude_for_axis(axis)


def test_retired_aliases_absent_from_canonical_tables_but_still_ingest():
    for alias, canon in tv.RETIRED_ALIASES.items():
        assert alias not in tv.DIRECTIONAL_TAG_AXIS      # retired from the canonical table
        assert alias not in fs.TAG_DIRECTIONS
        assert canon in tv.DIRECTIONAL_TAG_AXIS          # canonical target exists
        assert tv.canonicalize_tag(alias) == canon       # but ingest still maps it


def test_alias_still_scores_via_canonicalization():
    # A row/fixture using a retired alias must still be direction-checkable (canonicalized on ingest).
    behavior = {"tint_delta_a": 3.0}
    res_alias = fs.evaluate_direction(behavior, ["more_magenta"])
    res_canon = fs.evaluate_direction(behavior, ["tint_magenta"])
    assert res_alias.status == res_canon.status == "pass"


def test_validate_tags_canonicalizes_aliases():
    # Only the contrast axis moves, so the reverse unmentioned-behavior check is satisfied and the
    # alias must be backed by contrast_l_spread_delta exactly like the canonical tag.
    contrast = {"contrast_l_spread_delta": 4.0}
    ok_alias, _ = ig.validate_tags_against_behavior(["higher_contrast"], contrast)
    ok_canon, _ = ig.validate_tags_against_behavior(["more_contrast"], contrast)
    assert ok_alias == ok_canon is True


def test_new_behavior_v2_tag_families_declared():
    assert len(tv.HUE_SECTORS) == 7
    assert tv.HUE_CAST_TAGS == tuple(f"hue_cast_{s}" for s in tv.HUE_SECTORS)
    assert len(tv.SAT_SECTOR_TAGS) == 14   # up + down per sector
    # the new families are known vocabulary but NOT in the sign-checkable direction table
    for t in tv.HUE_CAST_TAGS + tv.SAT_SECTOR_TAGS:
        assert t not in tv.DIRECTIONAL_TAG_AXIS


def test_known_tags_have_no_alias_leakage():
    for alias in tv.RETIRED_ALIASES:
        assert alias not in ig.KNOWN_TAGS
