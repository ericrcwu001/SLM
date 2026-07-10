"""Single source of truth for the route / refuse taxonomy (ADR 0023).

The interpreter's 3-way **route** ``{grade, clarify, refuse}`` and the two **refuse kinds**
``{out_of_scope, out_of_gamut}`` plus the category strings that MUST stay byte-identical across
the five files that reference them:

  * :mod:`data_pipeline.unsupported_gen`   — teacher briefs / cues / validator;
  * :mod:`scripts.generate_unsupported`    — the balanced plan + row assembly;
  * :mod:`eval.fixtures.make_smoke_rows`   — the smoke eval fixtures;
  * :mod:`eval.unsupported_metrics`        — the boundary / refuse metrics;
  * :mod:`tests.test_unsupported_gen`      — the validator / plan tests.

``tests/test_taxonomy_sync.py`` asserts the cross-file invariants (the sync test ADR 0023 requires).

This module lives in :mod:`eval` — the lower layer — so BOTH ``eval`` and ``data_pipeline`` can
import it without a cycle (``data_pipeline`` depends one-way on ``eval``; ``eval`` must never import
``data_pipeline``). Pure / stdlib-only: no color, torch, or teacher deps, so it is import- and
unit-test-safe everywhere.
"""

from __future__ import annotations

# --- route enum (ADR 0021 / 0023): the interpreter's 3-way decision --------------------------
ROUTE_GRADE = "grade"        # a global color transform the generator can emit as 64 VQ codes
ROUTE_CLARIFY = "clarify"    # valid but under-specified color intent -> offer supported directions
ROUTE_REFUSE = "refuse"      # cannot / must-not be graded -> emit <unsupported>
ROUTES: tuple[str, ...] = (ROUTE_GRADE, ROUTE_CLARIFY, ROUTE_REFUSE)

# --- refuse kinds (ADR 0023) -----------------------------------------------------------------
REFUSE_OUT_OF_SCOPE = "out_of_scope"    # not a single global 3D-LUT transform at all
REFUSE_OUT_OF_GAMUT = "out_of_gamut"    # a global look the frozen tokenizer cannot represent
REFUSE_KINDS: tuple[str, ...] = (REFUSE_OUT_OF_SCOPE, REFUSE_OUT_OF_GAMUT)

# --- out_of_scope categories (existing 11) — a single global LUT cannot perform these ---------
OUT_OF_SCOPE_CATEGORIES: tuple[str, ...] = (
    "local_region_edit",
    "semantic_object_recolor",
    "content_removal",
    "content_replacement",
    "content_generation",
    "selective_preservation",
    "reference_style_transfer",
    "relighting",
    "texture_detail",
    "geometry",
    "inpainting",
)

# --- out_of_gamut categories (NEW, ADR 0023) — global, well-specified looks the frozen ---------
# tokenizer cannot represent (nearest materializable LUT exceeds mean dE00<=3.0 / p95<=6.0).
OUT_OF_GAMUT_CATEGORIES: tuple[str, ...] = (
    "infrared_false_color",   # infrared / false-color remap (foliage glows, skies go black)
    "pure_primary_cast",      # flood the whole frame with one pure primary (all red/green/blue)
    "hue_rotation",           # rotate every hue around the wheel (e.g. all hues by 180 degrees)
)

# --- clarify categories (NEW, ADR 0023) — valid but under-specified GLOBAL color intent -------
# These name no measurable direction ("make it better"); the interpreter asks for a supported
# direction instead of guessing a grade or refusing a legitimate request. NOTE: clarify is an
# INTERPRETER route — it is NOT a generator target and never enters the generator's SFT pool.
CLARIFY_CATEGORIES: tuple[str, ...] = (
    "underspecified_intent",
)

# All refuse categories, both kinds (pure only; mixed families are derived below).
REFUSE_CATEGORIES: tuple[str, ...] = OUT_OF_SCOPE_CATEGORIES + OUT_OF_GAMUT_CATEGORIES

# Mixed families: a supported global change + one out_of_scope component (still a refusal, because
# the whole request cannot be satisfied). The category string is this prefix + the component.
MIXED_PREFIX = "mixed_partial_supported_plus_"


def is_mixed_category(category: str) -> bool:
    return bool(category) and category.startswith(MIXED_PREFIX)


def refuse_kind_for_category(category: str | None) -> str | None:
    """The refuse kind (``out_of_scope`` / ``out_of_gamut``) for a category, or ``None``.

    Mixed families are ``out_of_scope`` (a supported half + an out-of-scope component). Returns
    ``None`` for clarify categories and for anything not in the refuse taxonomy.
    """
    if not category:
        return None
    if category in OUT_OF_GAMUT_CATEGORIES:
        return REFUSE_OUT_OF_GAMUT
    if category in OUT_OF_SCOPE_CATEGORIES or is_mixed_category(category):
        return REFUSE_OUT_OF_SCOPE
    return None


def route_for_category(category: str | None) -> str | None:
    """The route (``clarify`` / ``refuse``) for a non-grade category, or ``None`` if unknown.

    (``grade`` is not category-keyed — it is any supported row.)
    """
    if not category:
        return None
    if category in CLARIFY_CATEGORIES:
        return ROUTE_CLARIFY
    if refuse_kind_for_category(category) is not None:
        return ROUTE_REFUSE
    return None


def is_refuse_category(category: str | None) -> bool:
    return refuse_kind_for_category(category) is not None
