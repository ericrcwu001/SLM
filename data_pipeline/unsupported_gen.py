"""Unsupported / refusal prompt generation (teacher) + deterministic category validation.

The supported instruction path (:mod:`data_pipeline.instruction_gen`) turns an accepted
image+LUT pair into a natural editing instruction whose target is a LUT. This module builds
the *other* half of the behavior contract: (image, natural prompt, ``<unsupported>``) rows for
edits a single global 3D LUT cannot perform — local/semantic/generative/relighting/reference/
geometry/texture edits, selective preservation, and *mixed* prompts (a supported global change
plus an unsupported component). See ``docs/detailed_behavior_spec.md`` "Unsupported Prompt
Space" and ADR 0014.

Design (matches the master plan "deterministic category assignment + teacher labeling"):
  * the **category is assigned deterministically** by the caller's balanced plan;
  * the **teacher phrases** one realistic, image-grounded request in that category;
  * a **deterministic validator** (:func:`validate_unsupported_prompt`) rejects a phrasing that
    drifted into globally-supported territory (no category cue) — the guard that keeps a refusal
    row from silently becoming a supportable one.

Teacher-generated (not template) on purpose: the supported instructions are teacher-written, so
template-y refusal prompts would let the model learn the label from surface style instead of
meaning. Same gateway/profile as the supported teacher; gated identically (see
:class:`UnsupportedTeacherClient`).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Optional

import yaml

from eval import openai_compat

from .errors import RequiresTeacher, TeacherGenerationError

_ALIASES = {"latest", "stable", "current", "default", "auto"}
_REQUIRED_PROFILE_KEYS = ("provider", "model_id", "endpoint_env", "api_key_env",
                          "prompt_version", "batch_id")

# Canonical unsupported categories — MUST match the strings the eval harness scores
# (eval/fixtures/make_smoke_rows.py ``_UNSUPPORTED``; eval/unsupported_metrics.py).
PURE_CATEGORIES: tuple[str, ...] = (
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

# Human description handed to the teacher per category (what the request must require).
_CATEGORY_BRIEF: dict[str, str] = {
    "local_region_edit": "an edit applied to only ONE region/area of the image (e.g. only the "
                         "sky, only the background, only the left side), not the whole frame",
    "semantic_object_recolor": "recoloring ONE specific object/subject (e.g. a shirt, a car, "
                               "the hair) to a named color, leaving the rest unchanged",
    "content_removal": "removing/erasing an object or person from the image",
    "content_replacement": "replacing part of the image with different content (e.g. swap the sky)",
    "content_generation": "adding/generating NEW content that is not in the photo (e.g. add rain, "
                          "add birds, add a moon)",
    "selective_preservation": "changing part of the image while keeping another part exactly the "
                             "same (e.g. brighten the face but leave everything else dark)",
    "reference_style_transfer": "copying the look/colors FROM a separate reference image",
    "relighting": "changing the lighting, light direction, or adding cast shadows (e.g. make it "
                 "look like sunset light from the left)",
    "texture_detail": "a texture/detail edit such as sharpen, blur, denoise, deblur, skin "
                     "smoothing, or hair cleanup",
    "geometry": "a geometry/camera change such as crop, straighten, rotate, or perspective",
    "inpainting": "filling in a missing/damaged region of the photo",
}

# Deterministic cue substrings per category. The generated prompt (lowercased) must contain at
# least one, or it is rejected as "no category cue" — the guard against a globally-supported
# phrasing sneaking in. Kept broad (recall over precision): the teacher is already category-
# conditioned, so this catches drift, it does not do the teacher's job.
_CATEGORY_CUES: dict[str, tuple[str, ...]] = {
    "local_region_edit": ("only", "just the", "background", "foreground", "the sky", "left",
                          "right", "top ", "bottom", "corner", "region", "area", "behind",
                          "in the back", "blur the back"),
    "semantic_object_recolor": ("recolor", "shirt", "car", "dress", "hair", "eyes", "wall",
                               "jacket", "sky blue", "make the", "turn the", "change the",
                               "to red", "to blue", "to green", "to purple", "to pink"),
    "content_removal": ("remove", "erase", "delete", "get rid of", "take out", "without the"),
    "content_replacement": ("replace", "swap", "instead of", "turn ... into", " into a", "change the sky to"),
    "content_generation": ("add", "insert", "put a", "generate", "create", "rain", "clouds",
                          "birds", "moon", "stars", "snow", "fog"),
    "selective_preservation": ("leave", "keep", "but leave", "except", "everything else",
                              "rest of", "only the", "while keeping", "unchanged", "the same"),
    "reference_style_transfer": ("reference", "this image", "copy the colors", "match the style",
                                "like the attached", "from the other", "same look as"),
    "relighting": ("relight", "light", "lighting", "sunset", "shadow", "sunlight", "from the left",
                  "from the right", "golden hour", "backlit"),
    "texture_detail": ("sharpen", "blur", "denoise", "deblur", "smooth", "clean up", "detail",
                      "grain", "soften the", "skin"),
    "geometry": ("crop", "straighten", "rotate", "perspective", "warp", "stretch", "flip",
                "resize", "tilt", "align"),
    "inpainting": ("fill in", "inpaint", "missing", "patch", "reconstruct", "restore the",
                  "damaged", "torn"),
}

# Mixed families: a supported global attribute + one unsupported component. The category string
# matches the smoke-row convention ``mixed_partial_supported_plus_<family>``.
MIXED_FAMILIES: tuple[dict, ...] = (
    {"category": "mixed_partial_supported_plus_content_removal",
     "unsupported_component": "content_removal", "component_category": "content_removal"},
    {"category": "mixed_partial_supported_plus_semantic_recolor",
     "unsupported_component": "semantic_object_recolor", "component_category": "semantic_object_recolor"},
    {"category": "mixed_partial_supported_plus_local_edit",
     "unsupported_component": "local_region_edit", "component_category": "local_region_edit"},
    {"category": "mixed_partial_supported_plus_texture",
     "unsupported_component": "texture_detail", "component_category": "texture_detail"},
    {"category": "mixed_partial_supported_plus_relighting",
     "unsupported_component": "relighting", "component_category": "relighting"},
    {"category": "mixed_partial_supported_plus_content_generation",
     "unsupported_component": "content_generation", "component_category": "content_generation"},
)

# Supported global attributes the mixed prompt's supported half may use, with a detection cue.
SUPPORTED_ATTRS: tuple[tuple[str, str], ...] = (
    ("warmer", "warm"), ("cooler", "cool"), ("brighter", "bright"), ("darker", "dark"),
    ("more contrast", "contrast"), ("muted colors", "mut"), ("a faded look", "fad"),
    ("a matte look", "matte"), ("a cinematic look", "cinema"),
)


def build_unsupported_system_prompt() -> str:
    return (
        "You write ONE realistic, natural photo-editing request that a GLOBAL color-grading model "
        "MUST REFUSE. The model can only apply a single global 3D color LUT: it remaps every "
        "pixel's color the same way and CANNOT address regions, objects, or subjects, add or "
        "remove content, relight, change geometry, edit texture/detail, or copy a reference image.\n\n"
        "You are given a source image and a target UNSUPPORTED capability. Write one request that:\n"
        "  - clearly REQUIRES that unsupported capability (so a global LUT cannot satisfy it);\n"
        "  - is natural, as a real user would phrase it;\n"
        "  - is grounded in the actual image when the capability references content (only ask to "
        "recolor/remove an object that is actually visible);\n"
        "  - is a single sentence, no preamble.\n\n"
        "OUTPUT CONTRACT — output EXACTLY ONE JSON object and nothing else (no prose, no markdown "
        "fences):\n"
        '{ "prompt": "the user request", "grounded_in_image": true }'
    )


def build_mixed_system_prompt() -> str:
    return (
        "You write ONE realistic photo-editing request that a GLOBAL color-grading model MUST "
        "REFUSE because it MIXES a supported global color change with an UNSUPPORTED component. "
        "The model can apply a single global color LUT (whole-image tone/color) but CANNOT do "
        "region/object edits, add/remove content, relight, change geometry, or edit texture.\n\n"
        "You are given a source image, a SUPPORTED global change, and an UNSUPPORTED component. "
        "Write one natural request that asks for BOTH in the same breath (e.g. \"make it warmer "
        "and remove the person\"). The whole request must be refused because of the unsupported "
        "part. One sentence, no preamble.\n\n"
        "OUTPUT CONTRACT — output EXACTLY ONE JSON object and nothing else:\n"
        '{ "prompt": "the user request", "supported_part": "...", "unsupported_part": "..." }'
    )


def build_messages(plan_item: dict, *, attach_image: bool = True, min_image_edge: int = 256) -> list:
    """OpenAI-style chat messages for one plan item (system + user[+image])."""
    mixed = bool(plan_item.get("mixed"))
    if mixed:
        system = build_mixed_system_prompt()
        user_text = (
            f"SUPPORTED global change: {plan_item['supported_attr']}\n"
            f"UNSUPPORTED component (category {plan_item['component_category']}): "
            f"{_CATEGORY_BRIEF[plan_item['component_category']]}\n\n"
            "Return ONLY the JSON object described in the system prompt."
        )
    else:
        system = build_unsupported_system_prompt()
        cat = plan_item["category"]
        user_text = (
            f"UNSUPPORTED capability required (category {cat}): {_CATEGORY_BRIEF[cat]}\n\n"
            "Return ONLY the JSON object described in the system prompt."
        )
    parts: list = []
    image_path = plan_item.get("image_path")
    if attach_image and image_path and os.path.exists(str(image_path)):
        from eval.frontier_client import encode_image

        b64, media = encode_image(str(image_path), min_image_edge)
        parts.append(openai_compat.image_part(f"data:{media};base64,{b64}"))
    parts.append(openai_compat.text_part(user_text))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": parts},
    ]


def _has_cue(text: str, cues: tuple[str, ...]) -> bool:
    # Word-boundary match, NOT substring: a bare "in" cue like "right" must not fire inside
    # "brighter" (that false-accept would let a globally-supported phrasing pass as a refusal).
    t = text.lower()
    return any(re.search(r"\b" + re.escape(c.strip()) + r"\b", t) for c in cues)


def _has_attr_cue(text: str, cue: str) -> bool:
    # Supported-attribute cues are word STEMS ("warm" for "warmer", "mut" for "muted", "fad" for
    # "faded", "cinema" for "cinematic"), so anchor at a LEADING word boundary and allow a suffix.
    # This matches the intended inflections while rejecting the mid-word substring hits that plain
    # ``cue in text`` accepts (e.g. "swarmed"/"lukewarm" satisfying the "warm" cue), which would
    # otherwise let a row with no real supported half pass as a mixed boundary case.
    return re.search(r"\b" + re.escape(cue.strip()), text.lower()) is not None


def validate_unsupported_prompt(prompt: str, plan_item: dict) -> tuple[bool, list[str]]:
    """Deterministic guard: the phrasing must actually require the assigned unsupported edit.

    Returns (ok, issues). For a pure category the prompt must contain a category cue. For a mixed
    row it must contain BOTH a supported-attribute cue and the unsupported component's cue (so it
    is genuinely a *mixed* boundary case, not a pure refusal or a pure supported request).
    """
    issues: list[str] = []
    if not prompt or len(prompt.strip()) < 8:
        return False, ["empty_or_too_short"]
    if bool(plan_item.get("mixed")):
        comp_cat = plan_item["component_category"]
        if not _has_cue(prompt, _CATEGORY_CUES[comp_cat]):
            issues.append(f"no_unsupported_cue:{comp_cat}")
        _, attr_cue = plan_item["_attr_pair"]
        if not _has_attr_cue(prompt, attr_cue):
            issues.append("no_supported_cue")
    else:
        cat = plan_item["category"]
        if not _has_cue(prompt, _CATEGORY_CUES[cat]):
            issues.append(f"no_category_cue:{cat}")
    return (len(issues) == 0), issues


class UnsupportedTeacherClient:
    """Same gateway/profile + gating as the supported teacher (``teacher_primary``)."""

    def __init__(self, model_clients_path: str | Path = "configs/model_clients.yaml",
                 *, attach_image: bool = True, min_image_edge: int = 256,
                 timeout: float = openai_compat.DEFAULT_TIMEOUT_S):
        self.path = Path(model_clients_path)
        self.attach_image = attach_image
        self.min_image_edge = min_image_edge
        self.timeout = timeout

    def _profile(self) -> Optional[dict]:
        if not self.path.exists():
            return None
        try:
            data = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001
            return None
        return data.get("teacher_primary") or None

    def is_available(self) -> bool:
        prof = self._profile()
        if not prof or any(not prof.get(k) for k in _REQUIRED_PROFILE_KEYS):
            return False
        return str(prof.get("model_id", "")).lower() not in _ALIASES

    def generate(self, plan_item: dict) -> dict:
        """Return ``{prompt, extra, provenance}`` for one plan item (raises on unusable output)."""
        if not self.is_available():
            raise RequiresTeacher("teacher_primary missing/aliased; cannot generate unsupported prompts.")
        prof = self._profile() or {}
        if str(prof.get("provider", "openai_compatible")) not in ("openai_compatible", "openai"):
            raise RequiresTeacher(f"unsupported teacher provider {prof.get('provider')!r}")
        try:
            base_url, api_key = openai_compat.resolve_endpoint(prof)
        except openai_compat.OpenAICompatError as exc:
            raise RequiresTeacher(f"teacher credentials not available: {exc}") from exc

        messages = build_messages(plan_item, attach_image=self.attach_image,
                                  min_image_edge=self.min_image_edge)
        rd = dict(prof.get("request_defaults") or {})
        try:
            client = openai_compat.build_client(base_url, api_key, timeout=self.timeout)
            res = openai_compat.chat_completion(
                client, str(prof["model_id"]), messages,
                max_tokens=int(rd.get("max_tokens", 1024)),
                temperature=rd.get("temperature"),
                reasoning_effort=openai_compat.effort_from_profile(prof))
        except openai_compat.OpenAICompatError as exc:
            raise TeacherGenerationError(f"teacher API call failed: {exc}") from exc
        if res.api_refusal or not res.text:
            raise TeacherGenerationError(f"teacher returned no usable text (finish={res.finish_reason})")
        try:
            obj = openai_compat.parse_json_object(res.text)
        except openai_compat.OpenAICompatError as exc:
            raise TeacherGenerationError(str(exc)) from exc
        prompt = str(obj.get("prompt") or "").strip()
        if not prompt:
            raise TeacherGenerationError(f"teacher JSON missing 'prompt' (keys={sorted(obj)})")
        provenance = {
            "teacher_provider": prof.get("provider"),
            "teacher_model_id": prof.get("model_id"),
            "teacher_prompt_version": prof.get("prompt_version"),
            "prompt_generation_batch_id": prof.get("batch_id"),
            "finish_reason": res.finish_reason, "usage": res.usage,
        }
        extra = {k: obj.get(k) for k in ("supported_part", "unsupported_part", "grounded_in_image")
                 if k in obj}
        return {"prompt": prompt, "extra": extra, "provenance": provenance}
