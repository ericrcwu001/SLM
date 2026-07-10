"""AttributeSpec — the interpreter↔generator interface (ADR 0021; ``docs/attribute_spec.md``).

The **structured, high-resolution color-attribute representation** the Interpreter produces from any
user text and the Generator is conditioned on, serialized to a deterministic, round-trippable
``attribute_spec_text``. It shares the ``behavior_v2`` axis schema with the pipeline's
``measured_behavior`` (ADR 0022), so a *requested* spec (from text) and a *measured* one (from a LUT)
are directly comparable — the symmetry the oracle upper-bound gate exploits (``docs/attribute_spec.md``
§8): serialize a LUT's ground-truth ``measured_behavior`` as ``attribute_spec_text`` and feed it to
the Generator.

Pure / stdlib + the shared tag vocabulary (:mod:`eval.tag_vocabulary`). No torch / color deps, so it
is import- and unit-test-safe and usable on the Colab GPU box and in the interpreter alike.

Serialization grammar (``docs/attribute_spec.md`` §7), deterministic and round-trippable
(``parse(serialize(spec)) == spec`` for a canonical spec):

    route=grade | warmer=+2.3 muted=+2.0 matte=+2.5 shadow_hue=210 highlight_hue=45 \
      split_strength=6.0 sat_green=-1.5 | conf=0.82

Key ordering is fixed (the ``_ORDER`` below); magnitudes are Lab units at 1-decimal precision, hue
angles are integer degrees, confidence 2 decimals; axes at/below their emit threshold are omitted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from eval.refuse_taxonomy import (
    REFUSE_KINDS,
    ROUTE_CLARIFY,
    ROUTE_GRADE,
    ROUTE_REFUSE,
    ROUTES,
)
from eval.tag_vocabulary import HUE_SECTORS

ATTRIBUTE_SPEC_VERSION = "attribute_spec_v1"

# Emit thresholds: below these an axis is considered "not asserted" and omitted from the text.
_MAG_EPS = 0.5        # Lab-unit magnitude axes (temperature, tint, L, contrast, chroma, ...)
_HUE_GATE = 1.0       # a region/global hue is emitted only if its cast magnitude clears this
_SAT_EPS = 1.0        # per-hue-sector chroma delta

# --- bipolar directional axes: behavior_v2 field -> (positive tag, negative tag) --------------
# The serialized key is the tag matching the sign; the value is the SIGNED magnitude (Lab units).
_BIPOLAR: dict[str, tuple[str, str]] = {
    "temperature_delta_b": ("warmer", "cooler"),
    "tint_delta_a": ("tint_magenta", "tint_green"),
    "mean_l_delta": ("brighter", "darker"),
    "contrast_l_spread_delta": ("more_contrast", "less_contrast"),
    "chroma_delta": ("more_saturated", "muted"),
    "black_point_l_delta": ("lifted_blacks", "crushed_blacks"),
    "highlight_l_delta": ("brighter_highlights", "softer_highlights"),
    "shadow_l_delta": ("lifted_shadows", "crushed_shadows"),
}
# tag -> (behavior field, sign) for parsing.
_TAG_TO_AXIS: dict[str, tuple[str, int]] = {}
for _fld, (_pos, _neg) in _BIPOLAR.items():
    _TAG_TO_AXIS[_pos] = (_fld, +1)
    _TAG_TO_AXIS[_neg] = (_fld, -1)

# --- unipolar magnitude axes (value >= 0): serialized key -> behavior field -------------------
_UNIPOLAR: dict[str, str] = {
    "matte": "matte_strength",
    "split_strength": "split_tone_strength",
}

# --- hue axes: serialized key -> (hue field, gate field) --------------------------------------
# A hue is only meaningful when its cast has magnitude; gate on the matching strength field.
_HUE: dict[str, tuple[str, str]] = {
    "global_hue": ("global_hue_deg", "global_hue_magnitude"),
    "shadow_hue": ("shadow_hue_deg", "split_tone_strength"),
    "highlight_hue": ("highlight_hue_deg", "split_tone_strength"),
}

# Fixed serialization order for the axis block (deterministic output).
_ORDER: tuple[str, ...] = (
    "warmer", "cooler", "tint_magenta", "tint_green", "brighter", "darker",
    "more_contrast", "less_contrast", "more_saturated", "muted",
    "lifted_blacks", "crushed_blacks", "lifted_shadows", "crushed_shadows",
    "brighter_highlights", "softer_highlights",
    "matte", "split_strength", "global_hue", "shadow_hue", "highlight_hue",
) + tuple(f"sat_{s}" for s in HUE_SECTORS)


@dataclass
class AttributeSpec:
    """A canonical, serializable color-attribute request/measurement (``behavior_v2`` axes).

    ``axes`` maps a behavior_v2 field name -> value AT CANONICAL PRECISION (magnitudes rounded to
    1 decimal, hues to int, per-hue-sat to 1 decimal) so ``parse(serialize(spec)) == spec`` exactly.
    ``sat`` maps a hue sector -> chroma delta. Build canonical instances via
    :func:`from_measured_behavior` or :func:`parse`; the raw dataclass is not auto-canonicalized.
    """

    route: str = ROUTE_GRADE
    axes: dict[str, float] = field(default_factory=dict)         # behavior_v2 field -> value
    sat: dict[str, float] = field(default_factory=dict)          # hue sector -> chroma delta
    confidence: float | None = None
    out_of_gamut: bool = False
    refuse_reason: str | None = None
    source_text: str | None = None
    attribute_spec_version: str = ATTRIBUTE_SPEC_VERSION

    def __post_init__(self):
        if self.route not in ROUTES:
            raise ValueError(f"bad route {self.route!r} (allowed: {ROUTES})")
        if self.refuse_reason is not None and self.refuse_reason not in REFUSE_KINDS:
            raise ValueError(f"bad refuse_reason {self.refuse_reason!r}")


def _round_mag(v: float) -> float:
    return round(float(v), 1)


def _fmt_mag(v: float) -> str:
    return f"{v:+.1f}"


def from_measured_behavior(mb: dict, *, route: str = ROUTE_GRADE, confidence: float | None = None,
                           source_text: str | None = None) -> AttributeSpec:
    """Build a canonical :class:`AttributeSpec` from a ``behavior_v2`` ``measured_behavior`` dict.

    This is the GROUND-TRUTH path used by the oracle gate (``docs/attribute_spec.md`` §8) and as the
    captioning target: it selects the salient axes (above their emit thresholds), rounded to the
    serialized precision, so the resulting spec serializes and round-trips exactly.
    """
    axes: dict[str, float] = {}
    for fld in _BIPOLAR:
        v = float(mb.get(fld, 0.0) or 0.0)
        if abs(v) >= _MAG_EPS:
            axes[fld] = _round_mag(v)
    for _key, fld in _UNIPOLAR.items():
        v = float(mb.get(fld, 0.0) or 0.0)
        if v >= _MAG_EPS:
            axes[fld] = _round_mag(v)
    for _key, (hue_fld, gate_fld) in _HUE.items():
        if float(mb.get(gate_fld, 0.0) or 0.0) >= _HUE_GATE:
            axes[hue_fld] = float(round(float(mb.get(hue_fld, 0.0) or 0.0)))
    sat: dict[str, float] = {}
    phs = mb.get("per_hue_saturation") or {}
    for sector in HUE_SECTORS:
        v = float(phs.get(sector, 0.0) or 0.0)
        if abs(v) >= _SAT_EPS:
            sat[sector] = _round_mag(v)
    return AttributeSpec(route=route, axes=axes, sat=sat, confidence=confidence,
                         source_text=source_text)


def serialize(spec: AttributeSpec) -> str:
    """Serialize an :class:`AttributeSpec` to canonical ``attribute_spec_text`` (deterministic)."""
    # bipolar: the tag encodes the direction, the value is a POSITIVE magnitude (Lab units).
    emitted: dict[str, str] = {}
    for fld, (pos, neg) in _BIPOLAR.items():
        if fld in spec.axes:
            v = spec.axes[fld]
            tag = pos if v >= 0 else neg
            emitted[tag] = f"{tag}={_fmt_mag(abs(v))}"
    for key, fld in _UNIPOLAR.items():
        if fld in spec.axes:
            emitted[key] = f"{key}={_fmt_mag(spec.axes[fld])}"
    for key, (hue_fld, _gate) in _HUE.items():
        if hue_fld in spec.axes:
            emitted[key] = f"{key}={int(spec.axes[hue_fld])}"
    for sector in HUE_SECTORS:
        if sector in spec.sat:
            emitted[f"sat_{sector}"] = f"sat_{sector}={_fmt_mag(spec.sat[sector])}"
    axis_str = " ".join(emitted[k] for k in _ORDER if k in emitted)

    parts = [f"route={spec.route}"]
    if spec.route == ROUTE_REFUSE:
        parts.append(f"refuse={spec.refuse_reason}" if spec.refuse_reason else "refuse=out_of_scope")
    else:
        parts.append(axis_str)
    if spec.confidence is not None:
        parts.append(f"conf={spec.confidence:.2f}")
    # a grade/clarify spec always keeps the axis field (possibly empty) so the grammar is stable
    return " | ".join(parts)


_KV_RE = re.compile(r"^([a-z_]+)=(.+)$")


def parse(text: str) -> AttributeSpec:
    """Parse ``attribute_spec_text`` back to a canonical :class:`AttributeSpec` (inverse of serialize)."""
    segments = [s.strip() for s in (text or "").split("|")]
    route = ROUTE_GRADE
    axes: dict[str, float] = {}
    sat: dict[str, float] = {}
    confidence: float | None = None
    refuse_reason: str | None = None
    for seg in segments:
        if not seg:
            continue
        if seg.startswith("route="):
            route = seg.split("=", 1)[1].strip()
            continue
        if seg.startswith("conf="):
            confidence = float(seg.split("=", 1)[1])
            continue
        if seg.startswith("refuse="):
            refuse_reason = seg.split("=", 1)[1].strip()
            continue
        # otherwise this is the axis block: space-separated key=value tokens
        for tok in seg.split():
            m = _KV_RE.match(tok)
            if not m:
                continue
            key, val = m.group(1), m.group(2)
            if key in _TAG_TO_AXIS:
                fld, sign = _TAG_TO_AXIS[key]
                axes[fld] = _round_mag(sign * abs(float(val)))   # tag encodes the direction
            elif key in _UNIPOLAR:
                axes[_UNIPOLAR[key]] = _round_mag(float(val))
            elif key in _HUE:
                axes[_HUE[key][0]] = float(int(val))
            elif key.startswith("sat_"):
                sector = key[4:]
                if sector in HUE_SECTORS:
                    sat[sector] = _round_mag(float(val))
    return AttributeSpec(route=route, axes=axes, sat=sat, confidence=confidence,
                         refuse_reason=refuse_reason)


def measured_behavior_to_text(mb: dict, *, route: str = ROUTE_GRADE,
                              confidence: float | None = None) -> str:
    """Convenience: ground-truth ``measured_behavior`` -> ``attribute_spec_text`` (oracle-gate path)."""
    return serialize(from_measured_behavior(mb, route=route, confidence=confidence))


def is_backed(spec: AttributeSpec, mb: dict, *, tol: float = 1.0) -> tuple[bool, list[str]]:
    """Backing rule (``docs/attribute_spec.md`` §6): every asserted axis must be backed by a
    measurable ``behavior_v2`` axis with the SAME sign and within ``tol`` of the measured value.

    Generalizes ``validate_tags_against_behavior``: the *language* is unbounded but the *asserted
    axes* must be the bounded, measurable set. Returns ``(ok, issues)``.
    """
    issues: list[str] = []
    for fld, v in spec.axes.items():
        m = float(mb.get(fld, 0.0) or 0.0)
        if fld.endswith("_hue_deg"):
            continue  # hue angle backing is checked by proximity elsewhere (interpreter eval)
        if v * m <= 0 and abs(v) >= _MAG_EPS:      # sign disagreement (or measured ~0)
            issues.append(f"unbacked_sign:{fld}")
        elif abs(v - m) > max(tol, 0.25 * abs(v)):  # magnitude far from measured
            issues.append(f"unbacked_magnitude:{fld}")
    phs = mb.get("per_hue_saturation") or {}
    for sector, v in spec.sat.items():
        m = float(phs.get(sector, 0.0) or 0.0)
        if v * m <= 0 and abs(v) >= _SAT_EPS:
            issues.append(f"unbacked_sat_sign:{sector}")
    return (len(issues) == 0), issues


def canonicalize(spec: AttributeSpec) -> AttributeSpec:
    """Round a raw spec to canonical precision (idempotent); used to canonicalize interpreter output."""
    axes = {}
    for fld, v in spec.axes.items():
        axes[fld] = float(int(round(v))) if fld.endswith("_hue_deg") else _round_mag(v)
    sat = {k: _round_mag(v) for k, v in spec.sat.items()}
    return replace(spec, axes=axes, sat=sat)
