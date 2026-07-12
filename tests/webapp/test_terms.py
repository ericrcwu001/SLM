from data_pipeline.attribute_spec import parse
from eval.tag_vocabulary import RETIRED_ALIASES
from webapp import terms


def test_glossary_is_exactly_grounded_plus_recognized_styles():
    glossary = terms.all_terms()
    assert len(glossary) == 54
    assert sum(item["grounded"] for item in glossary) == 47
    assert not set(RETIRED_ALIASES) & {item["term"] for item in glossary}
    assert all(item["term"] and item["axis"] and item["definition"] and item["example_usage"] for item in glossary)


def test_every_suggestion_is_grounded_for_representative_inputs():
    grounded = {item["term"] for item in terms.all_terms() if item["grounded"]}
    cases = [
        ("make it pop", None, "clarify"),
        ("make it cinematic", None, "grade"),
        ("warm and faded", parse("route=grade | warmer=+2.0"), "grade"),
        ("remove the person", None, "refuse"),
    ]
    for prompt, spec, route in cases:
        feedback = terms.suggest_terms(prompt, spec, route)
        assert all(item["grounded"] and item["term"] in grounded for item in feedback["suggested_terms"])


def test_vague_grade_gets_magnitude_without_redundant_axis():
    result = terms.suggest_terms("make it warmer", parse("route=grade | warmer=+1.0"), "grade")
    picks = {item["term"] for item in result["suggested_terms"]}
    assert {"slight", "moderate", "strong", "extreme"} <= picks
    assert "warmer" not in picks
    assert "HOW MUCH" in result["assessment"]


def test_specific_grade_avoids_magnitude_and_redundancy():
    result = terms.suggest_terms("make it much warmer", parse("route=grade | warmer=+4.0"), "grade")
    picks = {item["term"] for item in result["suggested_terms"]}
    assert not picks & {"slight", "moderate", "strong", "extreme", "warmer", "cooler"}


def test_style_composite_maps_to_grounded_axes_not_itself():
    result = terms.suggest_terms("make it cinematic", None, "grade")
    picks = [item["term"] for item in result["suggested_terms"]]
    assert picks[:3] == ["less_contrast", "muted", "split_strength"]
    assert "cinematic" not in picks


def test_clarify_leads_with_direction_and_refuse_has_no_terms():
    clarify = terms.suggest_terms("make it look nicer", None, "clarify")
    assert clarify["suggested_terms"][0]["term"] == "warmer"
    assert "under-specified" in clarify["assessment"]
    assert terms.suggest_terms("remove the person", None, "refuse")["suggested_terms"] == []
