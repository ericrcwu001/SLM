"""Cross-file sync test for the route / refuse taxonomy (ADR 0023).

The single source of truth is :mod:`eval.refuse_taxonomy`. This test asserts the FIVE files that
reference the taxonomy never drift from it:

  * :mod:`data_pipeline.unsupported_gen`   — every category has a teacher brief + validator cue;
  * :mod:`scripts.generate_unsupported`    — the balanced plan covers every refuse category + kind;
  * :mod:`eval.fixtures.make_smoke_rows`   — every fixture uses a taxonomy category + correct route;
  * :mod:`eval.unsupported_metrics`        — the route-aware metrics exist and are exercised;
  * :mod:`tests.test_unsupported_gen`      — (the validator/plan tests, run separately).

If any of these adds/renames a category without updating the taxonomy (or vice-versa), one of the
assertions below fails — the guard the ADR calls for.
"""

from __future__ import annotations

import importlib

from data_pipeline import unsupported_gen as ug
from eval import refuse_taxonomy as tax
from eval import unsupported_metrics as um
from eval.fixtures import make_smoke_rows as smoke

ALL_PURE = tax.OUT_OF_SCOPE_CATEGORIES + tax.OUT_OF_GAMUT_CATEGORIES + tax.CLARIFY_CATEGORIES


def test_taxonomy_groups_disjoint_and_unique():
    for group in (tax.OUT_OF_SCOPE_CATEGORIES, tax.OUT_OF_GAMUT_CATEGORIES,
                  tax.CLARIFY_CATEGORIES, tax.REFUSE_KINDS, tax.ROUTES):
        assert len(group) == len(set(group)), f"duplicate in {group}"
    scope, gamut, clar = map(set, (tax.OUT_OF_SCOPE_CATEGORIES,
                                   tax.OUT_OF_GAMUT_CATEGORIES, tax.CLARIFY_CATEGORIES))
    assert scope.isdisjoint(gamut) and scope.isdisjoint(clar) and gamut.isdisjoint(clar)
    assert set(tax.ROUTES) == {tax.ROUTE_GRADE, tax.ROUTE_CLARIFY, tax.ROUTE_REFUSE}
    assert set(tax.REFUSE_KINDS) == {tax.REFUSE_OUT_OF_SCOPE, tax.REFUSE_OUT_OF_GAMUT}


def test_route_and_refuse_kind_mapping():
    for c in tax.OUT_OF_SCOPE_CATEGORIES:
        assert tax.route_for_category(c) == tax.ROUTE_REFUSE
        assert tax.refuse_kind_for_category(c) == tax.REFUSE_OUT_OF_SCOPE
    for c in tax.OUT_OF_GAMUT_CATEGORIES:
        assert tax.route_for_category(c) == tax.ROUTE_REFUSE
        assert tax.refuse_kind_for_category(c) == tax.REFUSE_OUT_OF_GAMUT
    for c in tax.CLARIFY_CATEGORIES:
        assert tax.route_for_category(c) == tax.ROUTE_CLARIFY
        assert tax.refuse_kind_for_category(c) is None
    # mixed families are out_of_scope refusals
    mixed = tax.MIXED_PREFIX + "content_removal"
    assert tax.refuse_kind_for_category(mixed) == tax.REFUSE_OUT_OF_SCOPE
    assert tax.route_for_category("not_a_category") is None


def test_unsupported_gen_briefs_and_cues_cover_taxonomy():
    # unsupported_gen re-exports the out_of_scope tuple and defines a brief + cue for EVERY category.
    assert ug.PURE_CATEGORIES == tax.OUT_OF_SCOPE_CATEGORIES
    for c in ALL_PURE:
        assert c in ug._CATEGORY_BRIEF, f"missing brief: {c}"
        assert c in ug._CATEGORY_CUES, f"missing cue: {c}"
        assert ug._CATEGORY_CUES[c], f"empty cue tuple: {c}"
    # no stale brief/cue keys that are not in the taxonomy (drift the other way)
    extra_brief = set(ug._CATEGORY_BRIEF) - set(ALL_PURE)
    extra_cue = set(ug._CATEGORY_CUES) - set(ALL_PURE)
    assert not extra_brief, f"stale brief keys: {extra_brief}"
    assert not extra_cue, f"stale cue keys: {extra_cue}"


def test_mixed_families_reference_out_of_scope_components():
    for fam in ug.MIXED_FAMILIES:
        assert fam["category"].startswith(tax.MIXED_PREFIX)
        assert fam["component_category"] in tax.OUT_OF_SCOPE_CATEGORIES


def test_smoke_fixtures_use_taxonomy_categories():
    scope_and_mixed = set(tax.OUT_OF_SCOPE_CATEGORIES) | {
        f["category"] for f in ug.MIXED_FAMILIES}
    for _instr, cat, *_rest in smoke._UNSUPPORTED:
        assert cat in scope_and_mixed, f"smoke _UNSUPPORTED uses unknown category {cat}"
    for _instr, cat in smoke._OUT_OF_GAMUT:
        assert cat in tax.OUT_OF_GAMUT_CATEGORIES, f"smoke _OUT_OF_GAMUT unknown {cat}"
    for _instr, cat in smoke._CLARIFY:
        assert cat in tax.CLARIFY_CATEGORIES, f"smoke _CLARIFY unknown {cat}"


def test_smoke_out_of_gamut_and_clarify_prompts_carry_a_cue():
    # Each hand-written fixture must contain its category cue, so it would survive the same
    # deterministic validator the teacher rows pass (keeps fixtures honest to the taxonomy).
    for instr, cat in smoke._OUT_OF_GAMUT:
        ok, issues = ug.validate_unsupported_prompt(instr, {"category": cat, "mixed": False})
        assert ok, (cat, instr, issues)
    for instr, cat in smoke._CLARIFY:
        ok, issues = ug.validate_unsupported_prompt(instr, {"category": cat, "mixed": False})
        assert ok, (cat, instr, issues)


def test_metrics_are_route_aware():
    # The refuse-kind + clarify metrics exist and split correctly (ADR 0023).
    recs = [
        um.DecisionRecord("a", is_supported=False, kind="unsupported",
                          route=tax.ROUTE_REFUSE, refuse_kind=tax.REFUSE_OUT_OF_GAMUT),
        um.DecisionRecord("b", is_supported=False, kind="lut_tokens",
                          route=tax.ROUTE_REFUSE, refuse_kind=tax.REFUSE_OUT_OF_GAMUT),
        um.DecisionRecord("c", is_supported=False, kind="unsupported",
                          route=tax.ROUTE_REFUSE, refuse_kind=tax.REFUSE_OUT_OF_SCOPE),
        um.DecisionRecord("d", is_supported=False, kind="unsupported", route=tax.ROUTE_CLARIFY),
        um.DecisionRecord("e", is_supported=True, kind="lut_tokens"),
    ]
    out = um.compute_unsupported_metrics(recs)
    m, s = out["metrics"], out["scalars"]
    assert m["out_of_gamut_recall"].n == 2 and m["out_of_gamut_recall"].rate == 0.5
    assert m["out_of_scope_recall"].n == 1 and m["out_of_scope_recall"].rate == 1.0
    # clarify row 'd' refused -> over-refusal 1/1; and clarify is excluded from the binary counts
    assert m["clarify_over_refusal_rate"].n == 1 and m["clarify_over_refusal_rate"].rate == 1.0
    assert s["n_gold_supported"] == 1 and s["n_gold_unsupported"] == 3 and s["n_clarify"] == 1


def test_generate_unsupported_imports_taxonomy():
    # The plan-builder script pulls the categories from the taxonomy (not a local copy).
    gu = importlib.import_module("scripts.generate_unsupported")
    assert gu.OUT_OF_GAMUT_CATEGORIES == tax.OUT_OF_GAMUT_CATEGORIES
    assert gu.refuse_kind_for_category("hue_rotation") == tax.REFUSE_OUT_OF_GAMUT
