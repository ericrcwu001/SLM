"""Unified tag vocabulary — single source of truth (ADR 0022; ``docs/attribute_spec.md`` §10).

Reconciles the three previously-divergent tag sets — ``data_pipeline.instruction_gen._TAG_BEHAVIOR``,
``eval.frontier_scoring.TAG_DIRECTIONS``, and the ``docs/eval_harness_implementation.md`` direction
table — into ONE canonical ``tag -> (behavior axis, sign)`` mapping, and RETIRES the divergent
aliases (`more_magenta`→`tint_magenta`, `higher_contrast`→`more_contrast`, `desaturated`→`muted`, …).

Lives in :mod:`eval` (the lower layer) so both ``eval`` and ``data_pipeline`` import it without a
cycle (``data_pipeline`` depends one-way on ``eval``). Pure / stdlib-only.

Consumers canonicalize incoming tags via :func:`canonicalize_tag` before lookup, so any legacy row
or fixture still using a retired alias continues to score correctly while the alias is removed from
the code vocabulary. ``tests/test_tag_vocabulary.py`` guards the cross-file invariants.
"""

from __future__ import annotations

# --- canonical directional tags: tag -> (behavior_v2 axis key, required sign) -----------------
# The sign-checkable subset (a tag passes iff its measured axis has the right sign + magnitude).
# Region-tone hue tags map to the split-tone b* component (warmer=+b toward yellow, cooler=-b
# toward blue), the backward-compatible sign proxy for the behavior_v2 region-hue axes.
DIRECTIONAL_TAG_AXIS: dict[str, tuple[str, int]] = {
    "warmer": ("temperature_delta_b", +1),
    "cooler": ("temperature_delta_b", -1),
    "tint_magenta": ("tint_delta_a", +1),
    "tint_green": ("tint_delta_a", -1),
    "brighter": ("mean_l_delta", +1),
    "darker": ("mean_l_delta", -1),
    "more_contrast": ("contrast_l_spread_delta", +1),
    "less_contrast": ("contrast_l_spread_delta", -1),
    "more_saturated": ("chroma_delta", +1),
    "muted": ("chroma_delta", -1),
    "lifted_blacks": ("black_point_l_delta", +1),
    "crushed_blacks": ("black_point_l_delta", -1),
    "lifted_shadows": ("shadow_l_delta", +1),
    "brighter_highlights": ("highlight_l_delta", +1),
    "softer_highlights": ("highlight_l_delta", -1),
    "warmer_shadows": ("split_tone_shadow_b", +1),
    "cooler_shadows": ("split_tone_shadow_b", -1),
    "warmer_highlights": ("split_tone_highlight_b", +1),
    "cooler_highlights": ("split_tone_highlight_b", -1),
}

# Retired aliases -> canonical tag. Removed from the code vocabulary; canonicalized on ingest.
RETIRED_ALIASES: dict[str, str] = {
    "more_magenta": "tint_magenta",
    "more_green": "tint_green",
    "higher_contrast": "more_contrast",
    "softer_contrast": "less_contrast",
    "desaturated": "muted",
}

# Style-bundle tags: measured composites (not sign-directional). Kept verbatim across the pipeline.
STYLE_TAGS: tuple[str, ...] = (
    "matte", "faded", "filmic", "cinematic", "teal-orange", "sepia", "bleach bypass", "natural",
)

# --- NEW behavior_v2 hue-resolution tag families (ADR 0022 §10) --------------------------------
# Declared as KNOWN vocabulary. They are validated by hue-angle / sector proximity (not a simple
# sign on one axis), so they are intentionally NOT in DIRECTIONAL_TAG_AXIS; the interpreter and the
# eval honesty slices (P5+) match them against global_hue_deg / per_hue_saturation.
HUE_SECTORS: tuple[str, ...] = ("red", "orange", "yellow", "green", "cyan", "blue", "magenta")
HUE_CAST_TAGS: tuple[str, ...] = tuple(f"hue_cast_{s}" for s in HUE_SECTORS)
SAT_SECTOR_TAGS: tuple[str, ...] = (
    tuple(f"sat_{s}_up" for s in HUE_SECTORS) + tuple(f"sat_{s}_down" for s in HUE_SECTORS)
)

# Per-axis minimum measurable magnitude (Lab units): chroma / split-tone-b run smaller than L/temp.
_MAG_LAB = 1.5
_MAG_CHROMA = 1.0
_CHROMA_AXES = {"chroma_delta", "highlight_chroma_delta", "shadow_chroma_delta",
                "split_tone_shadow_b", "split_tone_highlight_b",
                "split_tone_shadow_a", "split_tone_highlight_a"}


def min_magnitude_for_axis(axis: str) -> float:
    """The perceptible-movement bar for a behavior axis (used by the direction check)."""
    return _MAG_CHROMA if axis in _CHROMA_AXES else _MAG_LAB


def canonicalize_tag(tag: str) -> str:
    """Map a retired alias to its canonical tag (idempotent for canonical/unknown tags)."""
    return RETIRED_ALIASES.get(tag, tag)


def canonicalize_tags(tags) -> list[str]:
    return [canonicalize_tag(t) for t in (tags or [])]


DIRECTIONAL_TAGS: tuple[str, ...] = tuple(DIRECTIONAL_TAG_AXIS)
# The teacher-emittable vocabulary (directional + style), sorted for a stable prompt.
KNOWN_TAGS: list[str] = sorted(DIRECTIONAL_TAG_AXIS) + sorted(STYLE_TAGS)
# The full behavior_v2 vocabulary including the new (non-sign-checked) hue families.
ALL_TAGS: tuple[str, ...] = tuple(KNOWN_TAGS) + HUE_CAST_TAGS + SAT_SECTOR_TAGS
