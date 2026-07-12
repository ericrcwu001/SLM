"""Unsupported/refusal generation: validator, plan balance/leakage, and row schema."""

from __future__ import annotations

import importlib

import pytest

from data_pipeline.unsupported_gen import (
    MIXED_FAMILIES,
    PURE_CATEGORIES,
    SUPPORTED_ATTRS,
    validate_unsupported_prompt,
)
from eval.refuse_taxonomy import (
    OUT_OF_GAMUT_CATEGORIES,
    REFUSE_OUT_OF_GAMUT,
    REFUSE_OUT_OF_SCOPE,
    ROUTE_REFUSE,
)

gu = importlib.import_module("scripts.generate_unsupported")


def _pure(cat):
    return {"id": "x", "mixed": False, "category": cat, "unsupported_components": [cat],
            "supported_components": []}


def _mixed(fam, attr_pair):
    return {"id": "x", "mixed": True, "category": fam["category"],
            "component_category": fam["component_category"],
            "supported_attr": attr_pair[0], "_attr_pair": attr_pair,
            "unsupported_components": [fam["unsupported_component"]],
            "supported_components": [attr_pair[0]]}


def test_validator_accepts_category_appropriate():
    ok, issues = validate_unsupported_prompt("Make only the sky bluer.", _pure("local_region_edit"))
    assert ok, issues
    ok, _ = validate_unsupported_prompt("Change the shirt to red.", _pure("semantic_object_recolor"))
    assert ok
    ok, _ = validate_unsupported_prompt("Remove the person in the background.", _pure("content_removal"))
    assert ok


def test_validator_rejects_globally_supported_phrasing():
    # A purely global request assigned to a local category has no category cue -> rejected.
    ok, issues = validate_unsupported_prompt("Make the whole image warmer and brighter.",
                                             _pure("local_region_edit"))
    assert not ok and any("no_category_cue" in i for i in issues)


def test_validator_rejects_empty():
    ok, issues = validate_unsupported_prompt("   ", _pure("relighting"))
    assert not ok and "empty_or_too_short" in issues


def test_mixed_requires_both_cues():
    fam = MIXED_FAMILIES[1]  # semantic recolor family
    attr = ("warmer", "warm")
    ok, _ = validate_unsupported_prompt("Make it warmer and change the shirt to red.",
                                        _mixed(fam, attr))
    assert ok
    # missing the supported cue
    ok, issues = validate_unsupported_prompt("Change the shirt to red.", _mixed(fam, attr))
    assert not ok and "no_supported_cue" in issues
    # missing the unsupported component cue
    ok, issues = validate_unsupported_prompt("Just make it warmer overall.", _mixed(fam, attr))
    assert not ok and any("no_unsupported_cue" in i for i in issues)
    # stem cue matches an inflection ("mut" -> "muted") ...
    ok, _ = validate_unsupported_prompt("Give it muted colors and change the shirt to red.",
                                        _mixed(fam, ("muted colors", "mut")))
    assert ok
    # ... but must NOT fire on a mid-word substring ("swarmed" does not satisfy "warm")
    ok, issues = validate_unsupported_prompt("The birds swarmed; change the shirt to red.",
                                             _mixed(fam, attr))
    assert not ok and "no_supported_cue" in issues


def test_build_plan_deterministic_balanced_leakage_safe(tmp_path, monkeypatch):
    # Fake a source pool + supported set so the test does not depend on the real corpus.
    pool = [f"/img/src_{i:04d}.jpg" for i in range(400)]
    supported = {"/img/src_0000.jpg", "/img/src_0001.jpg"}
    monkeypatch.setattr(gu, "_source_image_pool", lambda: list(pool))
    monkeypatch.setattr(gu, "_supported_images", lambda: set(supported))

    p1 = gu.build_plan(n_train=34, n_eval=34)
    p2 = gu.build_plan(n_train=34, n_eval=34)
    assert [r["id"] for r in p1] == [r["id"] for r in p2]  # deterministic

    eval_imgs = {r["image_path"] for r in p1 if r["split"] == "eval"}
    train_imgs = {r["image_path"] for r in p1 if r["split"] == "train"}
    assert eval_imgs.isdisjoint(train_imgs)                 # no train/eval image leakage
    assert eval_imgs.isdisjoint(supported)                  # no eval/supported leakage
    assert train_imgs.isdisjoint(supported)
    # every bucket (11 out_of_scope + 3 out_of_gamut + 6 mixed) covered at least once in a
    # 40-item slice (2 x 20). Count families directly (not a deduped set of labels) so a
    # duplicated mixed-family category label drops the total below 20 and fails here.
    cats = {r["category"] for r in p1}
    assert len(cats) == len(PURE_CATEGORIES) + len(OUT_OF_GAMUT_CATEGORIES) + len(MIXED_FAMILIES) == 20


def test_build_plan_tags_route_and_refuse_kind(monkeypatch):
    # Every planned row is a refuse row; refuse_kind matches the category kind (ADR 0023).
    pool = [f"/repo/luts/raw/pack/src_{i:04d}.jpg" for i in range(400)]
    monkeypatch.setattr(gu, "_source_image_pool", lambda: list(pool))
    monkeypatch.setattr(gu, "_supported_images", lambda: set())
    plan = gu.build_plan(n_train=40, n_eval=40)
    assert plan and all(it["route"] == ROUTE_REFUSE for it in plan)
    gamut = [it for it in plan if it["category"] in OUT_OF_GAMUT_CATEGORIES]
    assert gamut and all(it["refuse_kind"] == REFUSE_OUT_OF_GAMUT for it in gamut)
    non_gamut = [it for it in plan if it["category"] not in OUT_OF_GAMUT_CATEGORIES]
    assert all(it["refuse_kind"] == REFUSE_OUT_OF_SCOPE for it in non_gamut)


def test_row_stores_portable_relative_path(monkeypatch):
    # The row must store a corpus-relative image_path (AUDIT F2 fix), not the absolute plan path.
    plan_item = {"id": "unsup_train_000001", "image_path": "/Users/x/Developer/SLM/luts/raw/f/y.jpg",
                 "split": "train", "headline_eligible": False, "split_unit_id": "unsup:y",
                 "mixed": False, "category": "hue_rotation", "unsupported_components": ["hue_rotation"],
                 "route": ROUTE_REFUSE, "refuse_kind": REFUSE_OUT_OF_GAMUT}
    row = gu._row_from(plan_item, "Rotate every hue by 180 degrees.").to_dict()
    assert row["image_path"] == "luts/raw/f/y.jpg"        # portable, resolves against SLM_ARTIFACT_ROOT
    assert not row["image_path"].startswith("/")
    assert row["route"] == ROUTE_REFUSE and row["refuse_kind"] == REFUSE_OUT_OF_GAMUT


def test_to_portable_image_path():
    assert gu.to_portable_image_path("/Users/a/SLM/luts/raw/x/y.jpg") == "luts/raw/x/y.jpg"
    assert gu.to_portable_image_path("luts/raw/x/y.jpg") == "luts/raw/x/y.jpg"   # already relative
    assert gu.to_portable_image_path("/abs/luts/raw/a.png") == "luts/raw/a.png"


def test_out_of_gamut_prompts_validate():
    for cat, prompt in [
        ("infrared_false_color", "Give the whole image an infrared false-color look."),
        ("pure_primary_cast", "Turn the entire photo pure red."),
        ("hue_rotation", "Rotate every hue by 180 degrees."),
    ]:
        ok, issues = validate_unsupported_prompt(prompt, {"category": cat, "mixed": False})
        assert ok, (cat, issues)
    # a globally-supported phrasing assigned to an out_of_gamut category has no cue -> rejected
    ok, issues = validate_unsupported_prompt("Make it a little warmer.",
                                             {"category": "hue_rotation", "mixed": False})
    assert not ok and any("no_category_cue" in i for i in issues)


def test_row_from_is_schema_valid_unsupported():
    plan = {"id": "unsup_train_000001", "image_path": "/img/x.jpg", "split": "train",
            "headline_eligible": False, "split_unit_id": "unsup:x",
            "mixed": False, "category": "relighting", "unsupported_components": ["relighting"]}
    row = gu._row_from(plan, "Make it look like sunset light from the left.").to_dict()
    assert row["is_supported"] is False
    assert row["support_label"] == "unsupported"
    assert row["assistant_target"] == "<unsupported>"
    assert row["target_tokens"] == []
    assert row["unsupported_category"] == "relighting"
    assert row["instruction"]


def test_supported_attrs_and_families_wellformed():
    assert all(len(a) == 2 for a in SUPPORTED_ATTRS)
    for fam in MIXED_FAMILIES:
        assert fam["component_category"] in PURE_CATEGORIES
        assert fam["category"].startswith("mixed_partial_supported_plus_")
    # Each mixed family must carry a UNIQUE category label and target a distinct component:
    # a duplicated label silently drops a family from the balanced plan's label space.
    assert len({f["category"] for f in MIXED_FAMILIES}) == len(MIXED_FAMILIES)
    assert len({f["component_category"] for f in MIXED_FAMILIES}) == len(MIXED_FAMILIES)


def test_clarify_validated_by_absence_of_direction():
    """Clarify prompts pass on open-ended vagueness and fail only when they name a real direction."""
    from eval.refuse_taxonomy import CLARIFY_CATEGORIES
    item = {"category": CLARIFY_CATEGORIES[0], "mixed": False}
    # Open-ended vague requests (none in the old fixed phrase list) must now PASS.
    for p in ("Can you make this photo look more appealing overall?",
              "Give it a nicer, more polished vibe.",
              "I want this to look more professional.",
              "Just make the colours feel right."):
        ok, issues = validate_unsupported_prompt(p, item)
        assert ok, (p, issues)
    # A prompt that NAMES a concrete supported direction is gradeable, not clarify -> REJECT.
    for p in ("Make it warmer and a bit brighter.",
              "Boost the saturation.",
              "Give it more contrast.",
              "Add a matte, faded look."):
        ok, issues = validate_unsupported_prompt(p, item)
        assert not ok and any(i.startswith("names_supported_direction") for i in issues), (p, issues)
