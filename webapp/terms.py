"""Grounded prompt vocabulary and deterministic prompt feedback.

The glossary structure is derived from the repository's canonical vocabulary tables.  Only the
human-readable prose lives here; adding or removing a pipeline term therefore fails at import time
until the glossary is deliberately updated.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

from data_pipeline.attribute_spec import (
    AttributeSpec,
    _BIPOLAR,
    _MAG_BUCKETS,
    _UNIPOLAR,
    parse,
)
from eval.tag_vocabulary import (
    DIRECTIONAL_TAG_AXIS,
    HUE_CAST_TAGS,
    HUE_SECTORS,
    RETIRED_ALIASES,
    SAT_SECTOR_TAGS,
    STYLE_TAGS,
    canonicalize_tag,
)


def _definition(plain: str, tech: str, example: str) -> dict[str, str]:
    return {"plain": plain, "tech": tech, "example": example}


_DEFS: dict[str, dict[str, str]] = {
    "warmer": _definition("shifts the image toward warm amber and yellow", "raises mean Lab b*", "make it warmer"),
    "cooler": _definition("shifts the image toward cool blue", "lowers mean Lab b*", "a cooler look"),
    "tint_magenta": _definition("adds a magenta-pink tint", "raises mean Lab a*", "slight magenta tint"),
    "tint_green": _definition("adds a green tint", "lowers mean Lab a*", "pull the tint green"),
    "brighter": _definition("raises overall exposure", "raises mean Lab L*", "make it brighter"),
    "darker": _definition("lowers overall exposure", "lowers mean Lab L*", "darker overall"),
    "more_contrast": _definition("widens the gap between darks and lights", "increases the Lab L* p95-p5 spread", "more contrast"),
    "less_contrast": _definition("flattens the tonal range", "narrows the Lab L* p95-p5 spread", "less contrast"),
    "more_saturated": _definition("makes color more intense", "raises mean Lab chroma", "more saturated"),
    "muted": _definition("reduces color intensity", "lowers mean Lab chroma", "a muted palette"),
    "lifted_blacks": _definition("turns the darkest tones dark grey for a faded low end", "raises the output black point and shadow toe", "lifted blacks"),
    "crushed_blacks": _definition("pushes shadows to deep black with less detail", "lowers or clamps the output black point", "crush the blacks"),
    "lifted_shadows": _definition("opens the shadows to reveal detail", "raises Lab L* in the shadow region", "lift the shadows"),
    "crushed_shadows": _definition("darkens the shadow region", "lowers Lab L* for input L* at or below 25", "deepen the shadows"),
    "brighter_highlights": _definition("brightens only the highlights", "raises Lab L* in the highlight region", "brighter highlights"),
    "softer_highlights": _definition("rolls off and protects highlights", "compresses the highlight shoulder", "soften the highlights"),
    "warmer_shadows": _definition("tints the shadows warmer", "raises the shadow-region split-tone b* shift", "warmer shadows"),
    "cooler_shadows": _definition("tints the shadows cooler", "lowers the shadow-region split-tone b* shift", "cool the shadows"),
    "warmer_highlights": _definition("tints the highlights warmer", "raises the highlight-region split-tone b* shift", "warm the highlights"),
    "cooler_highlights": _definition("tints the highlights cooler", "lowers the highlight-region split-tone b* shift", "cooler highlights"),
    "matte": _definition("creates a faded-film, low-contrast finish", "combines a lifted black point, reduced contrast, and slight desaturation", "a matte finish"),
    "split_strength": _definition("controls how strongly shadows and highlights receive different hues", "sets the split-toning separation magnitude", "stronger split-tone"),
    "slight": _definition("a barely-there nudge", "an emitted magnitude below 1.5 Lab units", "slightly warmer"),
    "moderate": _definition("a clear but restrained change", "a magnitude from 1.5 to below 3.0 Lab units", "moderately more contrast"),
    "strong": _definition("a bold, obvious change", "a magnitude from 3.0 to below 6.0 Lab units", "strongly muted"),
    "extreme": _definition("pushes the adjustment as far as it goes", "a magnitude of at least 6.0 Lab units", "extreme blue cast"),
    "teal-orange": _definition("pairs cool teal shadows with warm orange highlights", "a measured multi-axis style composite, not one serializable axis", "teal-orange grade"),
    "cinematic": _definition("suggests filmic, restrained color and split toning", "a measured multi-axis style composite", "cinematic color"),
    "filmic": _definition("suggests analog-film tonality", "a composite of lifted blacks, muted color, and soft highlights", "a filmic look"),
    "faded": _definition("creates a washed-out, low-contrast impression", "a composite of lifted blacks, reduced contrast, and muted color", "a faded look"),
    "sepia": _definition("creates a warm monochrome-brown impression", "a composite of warmth, muted color, and red-magenta tint", "a sepia look"),
    "bleach bypass": _definition("creates a silvery, high-contrast, desaturated look", "a measured contrast-plus-desaturation composite", "bleach bypass look"),
    "natural": _definition("keeps the grade close to true-to-life", "a composite calibration window with small movements and no strong cast", "keep it natural"),
}

_HUE_PLAIN = {
    "red": "red", "orange": "orange", "yellow": "yellow", "green": "green",
    "cyan": "cyan", "blue": "blue", "magenta": "magenta",
}
for _sector, _label in _HUE_PLAIN.items():
    _DEFS[f"hue_cast_{_sector}"] = _definition(
        f"pushes the overall color cast toward {_label}",
        f"sets global_hue_deg to the {_label} sector with a measurable magnitude",
        f"a {_label} cast",
    )
    _subject = {"orange": "oranges and skin-tone hues", "green": "greens and foliage", "blue": "blues and skies"}.get(_sector, f"{_label} hues")
    _DEFS[f"sat_{_sector}_up"] = _definition(
        f"boosts saturation in {_subject} only",
        f"raises chroma for input pixels in the {_label} hue sector",
        {"orange": "richer oranges", "green": "greener greens", "blue": "deeper blue skies"}.get(_sector, f"boost the {_label}s"),
    )
    _DEFS[f"sat_{_sector}_down"] = _definition(
        f"mutes saturation in {_subject} only",
        f"lowers chroma for input pixels in the {_label} hue sector",
        f"mute the {_label}s",
    )


_BUCKET_TERMS = tuple(label for _hi, label in _MAG_BUCKETS) + ("extreme",)


def _category(term: str, axis_field: str | None) -> str:
    if term in _BUCKET_TERMS:
        return "Magnitude"
    if term == "matte":
        return "Tone-shape"
    if term in HUE_CAST_TAGS or term == "split_strength":
        return "Hue"
    if axis_field in {"split_tone_shadow_b", "split_tone_highlight_b"}:
        return "Hue"
    if term in SAT_SECTOR_TAGS:
        return "Saturation"
    if term in STYLE_TAGS:
        return "Style"
    return "Direction"


def build_glossary() -> list[dict[str, Any]]:
    """Build the glossary from canonical code tables and assert exact prose coverage."""
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(term: str, axis: str, grounded: bool, *, sign: int | None = None) -> None:
        if term in seen:
            return
        prose = _DEFS[term]
        item: dict[str, Any] = {
            "term": term,
            "axis": axis,
            "category": _category(term, axis.split()[0]),
            "definition": f'{prose["plain"]}; {prose["tech"]}',
            "example_usage": prose["example"],
            "grounded": grounded,
        }
        if sign is not None:
            item["sign"] = sign
        entries.append(item)
        seen.add(term)

    for tag, (field, sign) in DIRECTIONAL_TAG_AXIS.items():
        add(tag, f"{field} ({'+' if sign > 0 else '-'})", True, sign=sign)
    for field, (_positive, negative) in _BIPOLAR.items():
        if negative not in DIRECTIONAL_TAG_AXIS:
            add(negative, f"{field} (-)", True, sign=-1)
    for key, field in _UNIPOLAR.items():
        add(key, f"{field} (>=0)", True)
    for tag in HUE_CAST_TAGS:
        add(tag, "global_hue_deg + global_hue_magnitude", True)
    for tag in SAT_SECTOR_TAGS:
        sector = tag[len("sat_"):].rsplit("_", 1)[0]
        sign = 1 if tag.endswith("_up") else -1
        add(tag, f"per_hue_saturation[{sector}] ({'+' if sign > 0 else '-'})", True, sign=sign)
    for label in _BUCKET_TERMS:
        add(label, "(magnitude bucket)", True)
    for tag in STYLE_TAGS:
        if tag != "matte":
            add(tag, "composite (calibration window; eval/configs/calibration_manifest.json)", False)

    retired = set(RETIRED_ALIASES)
    assert not retired & seen, "retired aliases leaked into the glossary"
    missing_prose = seen - set(_DEFS)
    extra_prose = set(_DEFS) - seen
    assert not missing_prose, f"missing term definitions: {sorted(missing_prose)}"
    assert not extra_prose, f"definitions are not backed by vocabulary: {sorted(extra_prose)}"
    return entries


_GLOSSARY = build_glossary()
_GROUNDED = frozenset(entry["term"] for entry in _GLOSSARY if entry["grounded"])
_BY_TERM = {entry["term"]: entry for entry in _GLOSSARY}


def all_terms() -> list[dict[str, Any]]:
    """Return the complete hover glossary without exposing mutable module state."""
    return deepcopy(_GLOSSARY)


api_terms = all_terms


_MAG_WORDS = (
    "slight", "slightly", "moderate", "moderately", "strong", "strongly", "extreme",
    "extremely", "subtle", "subtly", "barely", "a touch", "a bit", "a little", "hint",
    "gentle", "faint", "very", "super", "really", "heavily", "way", "much", "tons",
    "a lot", "punch", "intense", "dramatically", "aggressively",
)

_VAGUE_TO_GROUNDED: dict[str, list[str]] = {
    "pop": ["more_saturated", "more_contrast"],
    "punchy": ["more_contrast", "more_saturated"],
    "vibrant": ["more_saturated"], "vivid": ["more_saturated"], "colorful": ["more_saturated"],
    "washed": ["lifted_blacks", "muted", "matte"],
    "faded": ["lifted_blacks", "less_contrast", "muted"],
    "matte": ["matte", "lifted_blacks"],
    "moody": ["darker", "muted", "crushed_blacks"],
    "dramatic": ["more_contrast", "crushed_blacks"],
    "soft": ["less_contrast", "softer_highlights", "lifted_blacks"],
    "dreamy": ["less_contrast", "lifted_blacks", "muted"],
    "cinematic": ["less_contrast", "muted", "split_strength"],
    "filmic": ["lifted_blacks", "muted", "softer_highlights"],
    "film": ["lifted_blacks", "muted"],
    "vintage": ["warmer", "muted", "lifted_blacks", "matte"],
    "retro": ["warmer", "muted", "lifted_blacks"],
    "warm": ["warmer"], "cold": ["cooler"], "cool": ["cooler"],
    "teal-orange": ["cooler_shadows", "warmer_highlights", "split_strength"],
    "teal and orange": ["cooler_shadows", "warmer_highlights"],
    "sepia": ["warmer", "muted", "tint_magenta"],
    "bleach": ["more_contrast", "muted"],
    "bright": ["brighter"], "dark": ["darker"], "contrasty": ["more_contrast"],
    "flat": ["less_contrast"], "clean": ["muted", "less_contrast"],
    "rich": ["more_saturated", "more_contrast"],
}
_STARTER_DIRECTION = ("warmer", "cooler", "brighter", "darker", "more_contrast", "more_saturated")
_STARTER_MAGNITUDE = ("slight", "moderate", "strong", "extreme")


def _contains_phrase(text: str, phrase: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(phrase.lower())}(?![a-z])", text.lower()) is not None


def _has_magnitude(prompt: str) -> bool:
    return any(_contains_phrase(prompt or "", word) for word in _MAG_WORDS)


def _asserted_axes(spec: AttributeSpec | None) -> set[str]:
    if spec is None:
        return set()
    return set(spec.axes) | {f"sat:{sector}" for sector in spec.sat}


def _term_axis_field(term: str) -> str | None:
    if term in DIRECTIONAL_TAG_AXIS:
        return DIRECTIONAL_TAG_AXIS[term][0]
    if term in _UNIPOLAR:
        return _UNIPOLAR[term]
    if term in HUE_CAST_TAGS:
        return "global_hue_deg"
    if term.startswith("sat_"):
        return "sat:" + term[len("sat_"):].rsplit("_", 1)[0]
    for field, (_positive, negative) in _BIPOLAR.items():
        if term == negative:
            return field
    return None


def _is_redundant(term: str, prompt: str, asserted: set[str]) -> bool:
    natural = term.replace("_", " ")
    return _contains_phrase(prompt, term) or _contains_phrase(prompt, natural) or _term_axis_field(term) in asserted


def suggest_terms(prompt: str, parsed_spec: AttributeSpec | None, route: str) -> dict[str, Any]:
    """Suggest at most six deterministic, single, provably grounded vocabulary terms."""
    normalized = (prompt or "").lower()
    asserted = _asserted_axes(parsed_spec)
    picks: list[str] = []
    notes: list[str] = []

    def want(*terms: str) -> None:
        for raw_term in terms:
            term = canonicalize_tag(raw_term)
            if term in _GROUNDED and term not in picks and not _is_redundant(term, normalized, asserted):
                picks.append(term)

    if route == "refuse":
        return {
            "assessment": "This request cannot be expressed as one global color LUT, so there are no color terms to suggest.",
            "suggested_terms": [],
        }
    if route == "clarify":
        notes.append("Your request is under-specified. Pick a direction and say how much.")
        # Reserve half of the six-chip UI budget for magnitude.  The research finding this panel
        # mitigates is specifically that a direction without strength remains under-specified.
        want(*_STARTER_DIRECTION[:3])
        want(*_STARTER_MAGNITUDE)

    hit_vague = [word for word in _VAGUE_TO_GROUNDED if _contains_phrase(normalized, word)]
    for word in hit_vague:
        want(*_VAGUE_TO_GROUNDED[word])
    if hit_vague:
        notes.append("Vague style words map to specific grounded terms. Use one to name the actual adjustment.")

    if not _has_magnitude(prompt):
        want(*_STARTER_MAGNITUDE)
        notes.append("Your request does not say HOW MUCH. Add an intensity word (slight, moderate, strong, or extreme).")

    if parsed_spec is not None and len(asserted) == 1 and not hit_vague:
        only = next(iter(asserted))
        if only in {"temperature_delta_b", "tint_delta_a"}:
            want("more_contrast", "brighter")
            notes.append("Direction is clear. A grounded tonal term can sharpen it further.")

    if not picks:
        want(*_STARTER_DIRECTION)
        want(*_STARTER_MAGNITUDE)
        if picks:
            notes.append("Start by naming a direction and an intensity.")

    suggested = []
    for term in picks[:6]:
        entry = _BY_TERM[term]
        suggested.append({key: entry[key] for key in ("term", "axis", "definition", "example_usage", "grounded")})
    return {
        "assessment": " ".join(dict.fromkeys(notes)) if notes else "Your request already names a clear direction and intensity.",
        "suggested_terms": suggested,
    }


def prompt_feedback(prompt: str, route_result: Any) -> dict[str, Any]:
    """Adapt a pipeline route result to :func:`suggest_terms`."""
    route = "grade"
    spec: AttributeSpec | None = None
    if isinstance(route_result, dict):
        route = str(route_result.get("route") or route)
        candidate = route_result.get("parsed_spec")
        if isinstance(candidate, AttributeSpec):
            spec = candidate
        elif route_result.get("attribute_spec_text"):
            try:
                spec = parse(str(route_result["attribute_spec_text"]))
            except (TypeError, ValueError):
                spec = None
    elif isinstance(route_result, AttributeSpec):
        route, spec = route_result.route, route_result
    elif isinstance(route_result, str):
        route = route_result
    return suggest_terms(prompt, spec, route)


class TermsModule:
    """Small injectable facade used by :class:`PromptToLutPipeline`."""

    @staticmethod
    def all_terms() -> list[dict[str, Any]]:
        return all_terms()

    @staticmethod
    def prompt_feedback(prompt: str, route_result: Any) -> dict[str, Any]:
        return prompt_feedback(prompt, route_result)
