"""Procedural filler LUT generator (`procedural_fillers_v1`, access_method: generated).

The one locally-generatable source. Produces parametric canonical 17^3 absolute LUTs with
*predictable* measured behavior by applying deltas in CIE Lab (D65) and mapping back to
encoded sRGB, so a ``+warmth`` LUT really does raise b* etc. Covers the supported attribute
taxonomy (detailed_behavior_spec.md "Supported Prompt Space") and the 8 style bundles.

Train-only / headline-ineligible by default (ADR 0015). Output: one canonical ``.cube`` per
LUT under ``luts/raw/procedural/{generator_version}/``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from eval import color_pipeline as cp
from eval.cube_io import GRID_SIZE, identity_grid, write_cube

from ..constants import PROCEDURAL_GENERATOR_VERSION

LabTransform = Callable[[np.ndarray], np.ndarray]


# --- Lab-space primitives (operate on Lab array [...,3]) --------------------------
def _t_add_b(amount: float) -> LabTransform:  # temperature
    def f(lab):
        out = lab.copy(); out[..., 2] += amount; return out
    return f


def _t_add_a(amount: float) -> LabTransform:  # tint
    def f(lab):
        out = lab.copy(); out[..., 1] += amount; return out
    return f


def _t_exposure(amount: float) -> LabTransform:
    def f(lab):
        out = lab.copy(); out[..., 0] = np.clip(out[..., 0] + amount, 0.0, 100.0); return out
    return f


def _t_contrast(amount: float) -> LabTransform:
    def f(lab):
        out = lab.copy(); out[..., 0] = np.clip(50.0 + (out[..., 0] - 50.0) * (1.0 + amount), 0.0, 100.0); return out
    return f


def _t_black_point(lift: float) -> LabTransform:
    # lift shadows: strongest at L*=0, fading out by L*=40
    def f(lab):
        out = lab.copy()
        w = np.clip((40.0 - out[..., 0]) / 40.0, 0.0, 1.0)
        out[..., 0] = np.clip(out[..., 0] + lift * w, 0.0, 100.0)
        return out
    return f


def _t_highlights(amount: float) -> LabTransform:
    def f(lab):
        out = lab.copy()
        w = np.clip((out[..., 0] - 60.0) / 40.0, 0.0, 1.0)
        out[..., 0] = np.clip(out[..., 0] + amount * w, 0.0, 100.0)
        return out
    return f


def _t_shadows(amount: float) -> LabTransform:
    def f(lab):
        out = lab.copy()
        w = np.clip((40.0 - out[..., 0]) / 40.0, 0.0, 1.0)
        out[..., 0] = np.clip(out[..., 0] + amount * w, 0.0, 100.0)
        return out
    return f


def _t_saturation(scale: float) -> LabTransform:
    def f(lab):
        out = lab.copy(); out[..., 1] *= (1.0 + scale); out[..., 2] *= (1.0 + scale); return out
    return f


def _t_split_tone(shadow_ab: tuple[float, float], hi_ab: tuple[float, float], strength: float) -> LabTransform:
    """Push shadows toward shadow_ab and highlights toward hi_ab (luminance-weighted)."""
    def f(lab):
        out = lab.copy()
        L = out[..., 0]
        hi_w = np.clip((L - 50.0) / 50.0, 0.0, 1.0)
        lo_w = np.clip((50.0 - L) / 50.0, 0.0, 1.0)
        out[..., 1] += strength * (lo_w * shadow_ab[0] + hi_w * hi_ab[0])
        out[..., 2] += strength * (lo_w * shadow_ab[1] + hi_w * hi_ab[1])
        return out
    return f


def _compose(*transforms: LabTransform) -> LabTransform:
    def f(lab):
        for t in transforms:
            lab = t(lab)
        return lab
    return f


# --- catalog ----------------------------------------------------------------------
@dataclass
class LutSpec:
    lut_id: str
    kind: str            # "attribute" | "style"
    attribute: str | None
    style: str | None
    gold_tags: list[str]
    transform: LabTransform
    usage_prior_bucket: str = "common_head"


def _attribute_specs() -> list[LutSpec]:
    # Magnitudes chosen so a base (x1.0) LUT keeps skin within the skin_locus_v1 gate
    # (gold-eligible) while still clearing the direction-magnitude floors; the x1.4
    # generate() variants intentionally push into diagnostic territory for coverage.
    specs = [
        ("warmer", "temperature", _t_add_b(3.5), ["warmer"], "common_head"),
        ("cooler", "temperature", _t_add_b(-3.5), ["cooler"], "common_head"),
        ("tint_magenta", "tint", _t_add_a(3.5), ["tint_magenta"], "common_head"),
        ("tint_green", "tint", _t_add_a(-3.5), ["tint_green"], "common_head"),
        ("brighter", "exposure", _t_exposure(5.0), ["brighter"], "common_head"),
        ("darker", "exposure", _t_exposure(-5.0), ["darker"], "common_head"),
        ("more_contrast", "contrast", _t_contrast(0.10), ["more_contrast"], "common_head"),
        ("less_contrast", "contrast", _t_contrast(-0.10), ["less_contrast"], "subtle_control"),
        ("lifted_blacks", "black_point", _t_black_point(5.0), ["lifted_blacks"], "common_style"),
        ("crushed_blacks", "black_point", _t_black_point(-4.0), ["crushed_blacks"], "subtle_control"),
        ("brighter_highlights", "highlights", _t_highlights(5.0), ["brighter_highlights"], "subtle_control"),
        ("lifted_shadows", "shadows", _t_shadows(5.0), ["lifted_shadows"], "subtle_control"),
        ("more_saturated", "saturation", _t_saturation(0.12), ["more_saturated"], "common_head"),
        ("muted", "saturation", _t_saturation(-0.12), ["muted"], "common_style"),
    ]
    return [
        LutSpec(lut_id=f"proc_attr_{name}", kind="attribute", attribute=attr, style=None,
                gold_tags=tags, transform=tf, usage_prior_bucket=bucket)
        for name, attr, tf, tags, bucket in specs
    ]


def _style_specs() -> list[LutSpec]:
    styles = {
        "matte": _compose(_t_black_point(5.0), _t_contrast(-0.12), _t_saturation(-0.10)),
        "faded": _compose(_t_black_point(6.0), _t_saturation(-0.22), _t_highlights(-4.0), _t_contrast(-0.14)),
        "filmic": _compose(_t_highlights(-5.0), _t_saturation(-0.10), _t_contrast(0.05)),
        "cinematic": _compose(_t_split_tone((-4.0, -6.0), (6.0, 8.0), 0.5), _t_saturation(-0.10)),
        "teal-orange": _compose(_t_split_tone((-8.0, -10.0), (9.0, 12.0), 0.8), _t_saturation(-0.08)),
        "sepia": _compose(_t_add_b(9.0), _t_add_a(3.0), _t_saturation(-0.18)),
        "bleach bypass": _compose(_t_saturation(-0.40), _t_contrast(0.30), _t_black_point(-4.0)),
        "natural": _compose(_t_add_b(1.2), _t_contrast(0.03)),
    }
    return [
        LutSpec(lut_id=f"proc_style_{s.replace(' ', '_')}", kind="style", attribute=None, style=s,
                gold_tags=[s], transform=tf, usage_prior_bucket="common_style")
        for s, tf in styles.items()
    ]


def catalog() -> list[LutSpec]:
    return _attribute_specs() + _style_specs()


def generate_lut_tensor(spec: LutSpec, size: int = GRID_SIZE) -> np.ndarray:
    """Apply the spec's Lab transform to the identity grid -> canonical absolute LUT."""
    grid = identity_grid(size)
    lab = cp.srgb_to_lab_d65(grid)
    rgb = cp.lab_d65_to_srgb(spec.transform(lab))
    return np.clip(rgb, 0.0, 1.0)


@dataclass
class GeneratedLut:
    lut_id: str
    path: Path
    file_hash: str
    kind: str
    attribute: str | None
    style: str | None
    gold_tags: list[str]
    usage_prior_bucket: str
    generator_version: str = PROCEDURAL_GENERATOR_VERSION
    family: str = "controlled_procedural"
    source_pack_id: str = "procedural_fillers_v1"
    normalization_warnings: list = field(default_factory=list)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate(out_dir: str | Path, magnitudes: tuple[float, ...] = (0.6, 1.0, 1.4),
             size: int = GRID_SIZE) -> list[GeneratedLut]:
    """Generate the procedural catalog (each attribute at several magnitudes) as ``.cube``.

    ``magnitudes`` scales each attribute transform to add coverage variety; styles use their
    single recipe. Deterministic — no randomness.
    """
    out = Path(out_dir) / PROCEDURAL_GENERATOR_VERSION
    out.mkdir(parents=True, exist_ok=True)
    results: list[GeneratedLut] = []

    for spec in _attribute_specs():
        for mi, mult in enumerate(magnitudes):
            scaled = _scaled_spec(spec, mult, suffix=f"m{mi}")
            lut = generate_lut_tensor(scaled, size)
            path = out / f"{scaled.lut_id}.cube"
            write_cube(str(path), lut)
            results.append(_descriptor(scaled, path))

    for spec in _style_specs():
        lut = generate_lut_tensor(spec, size)
        path = out / f"{spec.lut_id}.cube"
        write_cube(str(path), lut)
        results.append(_descriptor(spec, path))

    return results


def _scaled_spec(spec: LutSpec, mult: float, suffix: str) -> LutSpec:
    base = spec.transform
    return LutSpec(
        lut_id=f"{spec.lut_id}_{suffix}",
        kind=spec.kind, attribute=spec.attribute, style=spec.style,
        gold_tags=list(spec.gold_tags),
        transform=(lambda lab, b=base, m=mult: _scale_delta(lab, b, m)),
        usage_prior_bucket=spec.usage_prior_bucket,
    )


def _scale_delta(lab: np.ndarray, transform: LabTransform, mult: float) -> np.ndarray:
    """Scale a transform's effect: identity + mult*(transform(x) - identity)."""
    return lab + mult * (transform(lab) - lab)


def _descriptor(spec: LutSpec, path: Path) -> GeneratedLut:
    return GeneratedLut(
        lut_id=spec.lut_id, path=path, file_hash=_sha256_file(path), kind=spec.kind,
        attribute=spec.attribute, style=spec.style, gold_tags=list(spec.gold_tags),
        usage_prior_bucket=spec.usage_prior_bucket,
    )
