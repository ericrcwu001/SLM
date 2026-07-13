"""Bake the 3 default "example" LUTs shown in the webapp gallery.

The gallery on ``webapp/static/index.html`` always renders three pinned example grades so a
first-time visitor immediately sees varied before/after effects *and* the prompt behind each.
Those examples must exist without a backend or model weights, so this script bakes them once into
static assets that ship with the site (the Modal image adds all of ``webapp/static/**``).

Each seed is a hand-authored, visibly distinct LUT built directly in absolute LUT space (the same
technique as ``webapp/pipeline.py::_stub_lut``), applied to a fitting bundled reference photo. We
reuse the repository's canonical LUT ops (``eval.cube_io.identity_grid``, ``webapp.lut.apply_lut``
/ ``export_cube`` / ``save_image``) so the exported ``.cube`` files are real and downloadable.

Run once locally; commit the outputs::

    python -m scripts.build_gallery_seeds
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from eval.cube_io import identity_grid
from webapp.lut import apply_lut, export_cube, save_image

_REPO = Path(__file__).resolve().parent.parent
_REFS = _REPO / "webapp/assets/references"
_OUT_DIR = _REPO / "webapp/static/gallery_seeds"
_OUT_JSON = _OUT_DIR / "seeds.json"

_GRID = 17


def _luma(lut: np.ndarray) -> np.ndarray:
    """Rec.709 luma of an absolute LUT, shape ``[N, N, N, 1]`` for broadcasting."""
    return (lut[..., 0] * 0.2126 + lut[..., 1] * 0.7152 + lut[..., 2] * 0.0722)[..., None]


def _shadows(luma: np.ndarray, knee: float = 0.5) -> np.ndarray:
    return np.clip((knee - luma) / knee, 0.0, 1.0)


def _highlights(luma: np.ndarray, knee: float = 0.5) -> np.ndarray:
    return np.clip((luma - knee) / (1.0 - knee), 0.0, 1.0)


def _warm_golden_hour() -> np.ndarray:
    """Amber highlights, gently lifted matte shadows, natural mids."""
    lut = identity_grid(_GRID).astype(np.float64)
    luma = _luma(lut)
    lut = (lut - 0.5) * 1.06 + 0.5                       # gentle global contrast
    lut += _shadows(luma) * np.array([0.03, 0.02, 0.05])  # lift the floor -> soft matte
    lut += np.array([0.045, 0.015, -0.045])               # overall warm balance
    lut += _highlights(luma) * np.array([0.04, 0.02, -0.03])  # amber the highlights
    return np.clip(lut, 0.0, 1.0)


def _teal_orange_cinematic() -> np.ndarray:
    """Deep teal shadows, warm highlights, crushed blacks for punchy contrast."""
    lut = identity_grid(_GRID).astype(np.float64)
    luma = _luma(lut)
    lut = (lut - 0.5) * 1.2 + 0.5                          # strong contrast, crush the blacks
    lut += _shadows(luma) * np.array([-0.04, 0.015, 0.07])  # push shadows toward teal
    lut += _highlights(luma) * np.array([0.06, 0.02, -0.06])  # warm the highlights (orange)
    return np.clip(lut, 0.0, 1.0)


def _faded_vintage_film() -> np.ndarray:
    """Muted color, milky faded blacks, a warm-green cast through the midtones."""
    lut = identity_grid(_GRID).astype(np.float64)
    luma = _luma(lut)
    lut = lut * 0.66 + luma * 0.34                         # desaturate toward luma
    lut = (lut - 0.5) * 0.88 + 0.5                         # lower contrast (flat film)
    lut += _shadows(luma) * 0.07                           # fade blacks to milky charcoal
    midtones = 1.0 - np.abs(luma - 0.5) * 2.0              # peaks in the mids
    lut += midtones * np.array([0.015, 0.03, -0.02])       # subtle warm-green cast
    return np.clip(lut, 0.0, 1.0)


# id / reference photo / prompt / spec flavor / LUT builder.
SEEDS = [
    {
        "id": "warm-golden-hour",
        "reference": "portrait.jpg",
        "prompt": (
            "Warm golden-hour glow: push the whites toward amber, lift the shadows into a soft "
            "matte, and keep skin tones natural."
        ),
        "spec_text": "route=grade | warmer=+2.4 lifted_blacks=+1.6 matte_strength=+1.2 highlight_hue=48",
        "lut": _warm_golden_hour,
    },
    {
        "id": "teal-orange-cinematic",
        "reference": "landscape.jpg",
        "prompt": (
            "Cinematic teal-and-orange: cool the shadows toward deep teal, warm the highlights, and "
            "crush the blacks for punchy contrast."
        ),
        "spec_text": "route=grade | more_contrast=+3.2 split_tone_strength=+3.0 shadow_hue=195 highlight_hue=42",
        "lut": _teal_orange_cinematic,
    },
    {
        "id": "faded-vintage-film",
        "reference": "food.jpg",
        "prompt": (
            "Faded vintage film: mute the color, fade the blacks to a milky charcoal, and add a "
            "subtle warm-green cast in the midtones."
        ),
        "spec_text": "route=grade | chroma_delta=-2.6 lifted_blacks=+2.8 matte_strength=+2.4 midtone_hue=95",
        "lut": _faded_vintage_film,
    },
]


def _render_pair(reference: Path, lut: np.ndarray, dest_dir: Path, max_edge: int) -> None:
    with Image.open(reference) as opened:
        image = opened.convert("RGB")
    image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    arr = np.asarray(image, dtype=np.float64) / 255.0
    graded = apply_lut(arr, lut)
    save_image(image, dest_dir / "before.jpg")
    save_image(np.rint(graded * 255.0).astype(np.uint8), dest_dir / "after.jpg")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-edge", type=int, default=1000, help="max edge of the before/after JPEGs")
    args = parser.parse_args(argv)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for seed in SEEDS:
        reference = _REFS / seed["reference"]
        if not reference.is_file():
            raise FileNotFoundError(f"missing reference photo: {reference}")
        dest_dir = _OUT_DIR / seed["id"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        lut = seed["lut"]()
        _render_pair(reference, lut, dest_dir, args.max_edge)
        export_cube(lut, dest_dir / "lut.cube")
        manifest.append(
            {
                "id": seed["id"],
                "prompt": seed["prompt"],
                "spec_text": seed["spec_text"],
                "before_url": f"/gallery_seeds/{seed['id']}/before.jpg",
                "after_url": f"/gallery_seeds/{seed['id']}/after.jpg",
                "cube_url": f"/gallery_seeds/{seed['id']}/lut.cube",
            }
        )
        print(f"[seeds] built {seed['id']} from {seed['reference']}")

    _OUT_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[seeds] wrote {_OUT_JSON} ({len(manifest)} seeds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
