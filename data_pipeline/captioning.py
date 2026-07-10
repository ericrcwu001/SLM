"""Captioning-for-diversity (ADR 0026): many diverse user-style captions per real corpus LUT.

Linguistic diversity comes from CAPTIONING existing corpus LUTs, not scraping. For each accepted
LUT we already have its measured ``behavior_v2`` vector; serialized to ``attribute_spec_text`` (ADR
0021) it is the **grounded target**. The teacher is repointed from writing one instruction to
writing MANY stylistically diverse requests that all map to that same target — literal, metaphor,
mood, concept, and slang phrasings — so the *input language* is unbounded while the *target* stays
backable (every caption is grounded in a renderable LUT). Recovered LUT titles seed concept-style
captions (AUDIT F7).

This module holds the PURE, teacher-independent pieces (caption styles, the system/user prompts, the
target derivation, and validation) so they are unit-testable without the gateway. The resumable
orchestration + gateway call live in :mod:`scripts.generate_captions`.
"""

from __future__ import annotations

from data_pipeline.attribute_spec import (
    AttributeSpec,
    from_measured_behavior,
    is_backed,
    serialize,
)
from eval.refuse_taxonomy import ROUTE_GRADE

# The diversity axes ADR 0026 names: the teacher writes one caption per style per LUT.
CAPTION_STYLES: tuple[str, ...] = ("literal", "metaphor", "mood", "concept", "slang")

_STYLE_BRIEF: dict[str, str] = {
    "literal": "a plain, literal editing instruction naming the color/tone changes",
    "metaphor": "a metaphor or simile for the look (e.g. 'like a faded old postcard')",
    "mood": "the emotional mood/vibe the look evokes (e.g. 'make it feel nostalgic and calm')",
    "concept": "a named concept or reference the look resembles (e.g. 'a vintage film stock', a "
               "place, a time of day) — use the LUT title as a hint when given, but only if it "
               "matches the measured look",
    "slang": "casual, everyday slang a non-expert might say (e.g. 'make it pop', 'give it that "
             "moody film vibe')",
}


def caption_target(measured_behavior: dict) -> AttributeSpec:
    """The grounded target for every caption of a LUT: its measured behavior as a grade AttributeSpec."""
    return from_measured_behavior(measured_behavior, route=ROUTE_GRADE)


def caption_target_text(measured_behavior: dict) -> str:
    return serialize(caption_target(measured_behavior))


def build_caption_system_prompt(n_styles: int) -> str:
    return (
        "You write DIVERSE natural-language requests a real user might type to ask a global "
        "color-grading model for a specific look. You are given the MEASURED color effect of one "
        "3D color LUT (Lab-domain deltas + hue/saturation summary) and, when available, its title "
        "and source image. A 3D LUT is GLOBAL: it remaps every pixel the same way; it cannot "
        "address regions, objects, content, lighting, geometry, or texture.\n\n"
        f"Write EXACTLY {n_styles} requests for the SAME look, one per requested STYLE, each a "
        "single sentence. Every request must describe the SAME measured global look (do not invent "
        "changes the measurements do not show, and never ask for local/content/relighting edits).\n\n"
        "OUTPUT CONTRACT — output EXACTLY ONE JSON object and nothing else (no prose, no fences):\n"
        '{ "captions": { "<style>": "the request", ... } }'
    )


def build_caption_user_text(measured_behavior: dict, *, title: str | None,
                            styles: tuple[str, ...] = CAPTION_STYLES) -> str:
    """The user message: the measured summary + the requested styles (+ title hint if recovered)."""
    spec_text = caption_target_text(measured_behavior)
    lines = [f"MEASURED look (attribute_spec): {spec_text}"]
    if title:
        lines.append(f"LUT title hint (use only if consistent with the measured look): {title!r}")
    lines.append("")
    lines.append("Write one request for EACH of these styles:")
    for s in styles:
        lines.append(f"  - {s}: {_STYLE_BRIEF[s]}")
    lines.append("")
    lines.append("Return ONLY the JSON object described in the system prompt.")
    return "\n".join(lines)


def validate_caption(caption: str) -> tuple[bool, list[str]]:
    """Deterministic guard: a usable caption is non-empty and a single realistic request."""
    issues: list[str] = []
    if not caption or len(caption.strip()) < 6:
        issues.append("empty_or_too_short")
    if len(caption) > 400:
        issues.append("too_long")
    return (len(issues) == 0), issues


def caption_is_grounded(measured_behavior: dict) -> tuple[bool, list[str]]:
    """The caption TARGET must be backed by the measured axes (ADR 0021 backing rule) — it is by
    construction (target == measured), so this is a defensive self-check for the pipeline."""
    spec = caption_target(measured_behavior)
    return is_backed(spec, measured_behavior)
