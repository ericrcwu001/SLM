"""Generate the Stage-1 smoke eval rows (50 supported / 20 unsupported).

These are **eval fixtures**, not the data-gen pipeline. Rows are written in the Eval
Unit schema with real instructions + gold_tags + support labels; ``target_tokens`` is
empty (there is no frozen tokenizer to produce 64 ids yet, and L2-L7 are disabled).
Tiny synthetic PNGs are generated so ``image_path`` / ``image_sha256`` resolve.

A companion ``mock_outputs.jsonl`` supplies a realistic mix of model outputs (valid
64-token, over-refusal, invalid, correct/incorrect refusal) so the mock-replay baseline
exercises the boundary/valid-token metrics meaningfully. Output is deterministic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

from ..output_parsers import format_tokens
from ..schemas import CANONICAL_DOMAIN_ID, EvalRow, write_rows
from ..vocab import UNSUPPORTED

# --- supported prompt templates: (instruction, gold_tags, attribute, style_bundle) ---
_SUPPORTED_ATOMS = [
    ("Make the image warmer.", ["warmer"], "temperature", None),
    ("Cool it down a little.", ["cooler"], "temperature", None),
    ("Add a touch more magenta.", ["more_magenta"], "tint", None),
    ("Make it a bit greener.", ["more_green"], "tint", None),
    ("Brighten the photo.", ["brighter"], "exposure", None),
    ("Make it darker overall.", ["darker"], "exposure", None),
    ("Give it more punch.", ["higher_contrast"], "contrast", None),
    ("Soften the contrast.", ["softer_contrast"], "contrast", None),
    ("Lift the blacks.", ["lifted_blacks"], "black_point", None),
    ("Crush the blacks.", ["crushed_blacks"], "black_point", None),
    ("Soften the highlights.", ["softer_highlights"], "highlights", None),
    ("Make the highlights brighter.", ["brighter_highlights"], "highlights", None),
    ("Lift the shadows.", ["lifted_shadows"], "shadows", None),
    ("Cool the shadows overall.", ["cooler_shadows"], "shadows", None),
    ("Make it more saturated.", ["more_saturated"], "saturation", None),
    ("Mute the colors a bit.", ["muted"], "saturation", None),
]

_SUPPORTED_STYLES = [
    ("Give it a soft matte look.", ["matte", "lifted_blacks", "muted"], "matte"),
    ("Make it look faded.", ["faded", "muted", "softer_contrast"], "faded"),
    ("Give it a filmic look.", ["filmic", "softer_highlights"], "filmic"),
    ("Add a cinematic teal-and-orange grade.", ["cinematic"], "cinematic"),
    ("Push a strong teal-orange look.", ["teal-orange"], "teal-orange"),
    ("Give it a warm sepia tone.", ["sepia", "warmer", "muted"], "sepia"),
    ("Apply a bleach-bypass look.", ["bleach_bypass", "higher_contrast", "muted"], "bleach bypass"),
    ("Keep it natural and clean.", ["natural"], "natural"),
]

_SUPPORTED_COMPOUND = [
    ("Make it warmer with softer contrast.", ["warmer", "softer_contrast"], "temperature"),
    ("Brighten it and lift the shadows.", ["brighter", "lifted_shadows"], "exposure"),
    ("Cool it down and mute the colors.", ["cooler", "muted"], "temperature"),
    ("More punch and a touch warmer.", ["higher_contrast", "warmer"], "contrast"),
]

# --- unsupported prompts: (instruction, category, unsup_components, supported_components, mixed) ---
_UNSUPPORTED = [
    ("Make only the sky bluer.", "local_region_edit", ["local_region_edit"], [], False),
    ("Change the shirt to red.", "semantic_object_recolor", ["semantic_object_recolor"], [], False),
    ("Remove the person in the background.", "content_removal", ["content_removal"], [], False),
    ("Make the face brighter but leave everything else dark.", "selective_preservation",
     ["selective_preservation"], [], False),
    ("Copy the colors from this reference image.", "reference_style_transfer",
     ["reference_style_transfer"], [], False),
    ("Make it look like sunset light is coming from the left.", "relighting",
     ["relighting"], [], False),
    ("Sharpen the details in the hair.", "texture_detail", ["texture_detail"], [], False),
    ("Blur the background.", "local_region_edit", ["local_region_edit"], [], False),
    ("Crop and straighten the photo.", "geometry", ["geometry"], [], False),
    ("Fill in the missing corner of the photo.", "inpainting", ["inpainting"], [], False),
    ("Recolor just the car to blue.", "semantic_object_recolor", ["semantic_object_recolor"], [], False),
    ("Replace the sky with a starry night.", "content_replacement", ["content_replacement"], [], False),
    # mixed (supported global request + unsupported component)
    ("Make the whole photo warmer and remove the background.",
     "mixed_partial_supported_plus_content_removal", ["content_removal"], ["warmer"], True),
    ("Give it a cinematic look and make the shirt red.",
     "mixed_partial_supported_plus_semantic_recolor", ["semantic_object_recolor"], ["cinematic"], True),
    ("Brighten it overall and blur the background.",
     "mixed_partial_supported_plus_local_edit", ["local_region_edit"], ["brighter"], True),
    ("Add a faded look and sharpen the eyes.",
     "mixed_partial_supported_plus_texture", ["texture_detail"], ["faded"], True),
    ("Cool the shadows and relight the scene from the right.",
     "mixed_partial_supported_plus_relighting", ["relighting"], ["cooler_shadows"], True),
    ("Make it moody and add falling rain.",
     "mixed_partial_supported_plus_content_generation", ["content_generation"], ["darker", "muted"], True),
]


def _synth_png(path: str, seed: int, size: int = 32) -> None:
    from PIL import Image

    # deterministic gradient with a per-row hue offset (no randomness)
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            r = (x * 8 + seed * 5) % 256
            g = (y * 8 + seed * 3) % 256
            b = (x * y + seed * 7) % 256
            px[x, y] = (r, g, b)
    img.save(path, format="PNG")


def _sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _build_supported(images_dir: str) -> list[EvalRow]:
    templates: list[tuple[str, list[str], str, str | None]] = []
    templates += _SUPPORTED_ATOMS
    templates += [(t[0], t[1], "style", t[2]) for t in _SUPPORTED_STYLES]
    templates += [(t[0], t[1], t[2], None) for t in _SUPPORTED_COMPOUND]
    # pad by cycling atoms until 50
    i = 0
    while len(templates) < 50:
        atom = _SUPPORTED_ATOMS[i % len(_SUPPORTED_ATOMS)]
        templates.append(atom)
        i += 1
    templates = templates[:50]

    rows: list[EvalRow] = []
    for idx, (instr, tags, attribute, style) in enumerate(templates, start=1):
        rid = f"eval_sup_{idx:06d}"
        img_path = os.path.join(images_dir, f"{rid}.png")
        _synth_png(img_path, seed=idx)
        rows.append(EvalRow(
            id=rid, instruction=instr, is_supported=True, support_label="supported",
            image_path=img_path, image_sha256=_sha256_file(img_path),
            gold_tags=list(tags), style_bundle=style, style_primary=style,
            attribute=attribute if attribute != "style" else None,
            style_bucket=(style or ("compound" if len(tags) > 1 else attribute)),
            canonical_domain_id=CANONICAL_DOMAIN_ID, split="smoke",
            target_tokens=[], acceptance_mode="exact_target",
            usage_weight=1.0, headline_eligible=False, procedural_filler=False,
        ))
    return rows


def _build_unsupported(images_dir: str) -> list[EvalRow]:
    rows: list[EvalRow] = []
    items = list(_UNSUPPORTED)
    i = 0
    while len(items) < 20:
        items.append(_UNSUPPORTED[i % len(_UNSUPPORTED)])
        i += 1
    items = items[:20]
    for idx, (instr, cat, unsup_c, sup_c, mixed) in enumerate(items, start=1):
        rid = f"eval_unsup_{idx:06d}"
        img_path = os.path.join(images_dir, f"{rid}.png")
        _synth_png(img_path, seed=1000 + idx)
        rows.append(EvalRow(
            id=rid, instruction=instr, is_supported=False, support_label="unsupported",
            image_path=img_path, image_sha256=_sha256_file(img_path),
            unsupported_category=cat, unsupported_components=list(unsup_c),
            supported_components=list(sup_c), mixed_prompt=bool(mixed),
            canonical_domain_id=CANONICAL_DOMAIN_ID, split="smoke",
            target_tokens=[], usage_weight=1.0, headline_eligible=False,
        ))
    return rows


def _link_boundary_pairs(supported: list[EvalRow], unsupported: list[EvalRow]) -> None:
    """Pair a few supported+unsupported rows into near-boundary contrastive pairs."""
    n_pairs = 5
    for k in range(n_pairs):
        pid = f"bp_{k + 1:03d}"
        supported[k].boundary_pair_id = pid
        supported[k].boundary_pair_role = "supported_global"
        supported[k].boundary_type = "global_vs_local"
        unsupported[k].boundary_pair_id = pid
        unsupported[k].boundary_pair_role = "unsupported_boundary"
        unsupported[k].boundary_type = "global_vs_local"


def _valid_line(offset: int) -> str:
    return format_tokens([(offset + i) % 256 for i in range(64)])


def _mock_output_for(row: EvalRow, idx: int) -> str:
    """Deterministic model-output mix for the mock-replay baseline."""
    if row.is_supported:
        if idx % 10 == 0:
            return UNSUPPORTED             # over-refusal
        if idx % 17 == 0:
            return "<lut_bos> oops " + _valid_line(idx)  # invalid (prose)
        return _valid_line(idx)            # correct non-refusal
    else:
        if idx % 9 == 0:
            return _valid_line(idx)        # false support
        if idx % 13 == 0:
            return "<lut_bos> <lut_042>"    # invalid (too few)
        return UNSUPPORTED                 # correct refusal


def generate(out_dir: str) -> tuple[str, str]:
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    supported = _build_supported(images_dir)
    unsupported = _build_unsupported(images_dir)
    _link_boundary_pairs(supported, unsupported)
    rows = supported + unsupported

    rows_path = os.path.join(out_dir, "smoke_rows.jsonl")
    write_rows(rows_path, rows)

    mock_path = os.path.join(out_dir, "mock_outputs.jsonl")
    with open(mock_path, "w", encoding="utf-8") as fh:
        for i, row in enumerate(rows):
            fh.write(json.dumps({"row_id": row.id, "text": _mock_output_for(row, i)}) + "\n")

    print(f"[make_smoke_rows] wrote {len(supported)} supported + {len(unsupported)} "
          f"unsupported rows -> {rows_path}")
    print(f"[make_smoke_rows] wrote mock outputs -> {mock_path}")
    print(f"[make_smoke_rows] wrote {len(rows)} synthetic images -> {images_dir}")
    return rows_path, mock_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate 50/20 smoke eval rows + fixtures.")
    ap.add_argument("--out", default="data/eval")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    generate(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
