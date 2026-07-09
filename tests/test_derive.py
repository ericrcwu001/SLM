"""Tests for Stage-4 derivation: XMP parse + global-LUT pair fit + decode."""

import numpy as np

from data_pipeline.lut_ops import apply_lut_trilinear
from data_pipeline.sources import procedural as proc
from data_pipeline.sources.derive import (
    cube_bytes_to_lut,
    fit_global_lut,
    parse_xmp,
)
from eval.cube_io import identity_grid, serialize_cube

_XMP_GLOBAL = """<x:xmpmeta xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">
  <rdf:Description crs:Temperature="5500" crs:Tint="+8" crs:Exposure2012="+0.35"
    crs:Contrast2012="+12" crs:Saturation="-5"/>
</x:xmpmeta>"""

_XMP_LOCAL = """<x:xmpmeta xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">
  <rdf:Description crs:Exposure2012="+0.20">
    <crs:MaskGroupBasedCorrections><rdf:Seq><rdf:li>mask</rdf:li></rdf:Seq></crs:MaskGroupBasedCorrections>
    <crs:PaintBasedCorrections><rdf:Seq><rdf:li>brush</rdf:li></rdf:Seq></crs:PaintBasedCorrections>
  </rdf:Description>
</x:xmpmeta>"""


def test_xmp_global_only_accepted():
    r = parse_xmp(_XMP_GLOBAL)
    assert r.parse_status == "parsed"
    assert r.local_tool_count == 0
    assert r.accepted is True
    assert "Temperature" in r.global_fields_present
    assert r.values["Exposure2012"] == 0.35


def test_xmp_local_tools_rejected():
    r = parse_xmp(_XMP_LOCAL)
    assert r.parse_status == "parsed"
    assert r.local_tool_count >= 2
    assert r.accepted is False
    assert any("Mask" in f or "Paint" in f for f in r.rejected_fields)


def test_xmp_unknown_schema():
    assert parse_xmp("not xmp at all").parse_status == "unknown_schema"


def _node_pixels(repeat=40):
    nodes = identity_grid(17).reshape(-1, 3)
    return np.repeat(nodes, repeat, axis=0)


def test_pair_fit_recovers_known_lut_exact():
    # raw estimator (smooth=False): every node observed -> exact per-node recovery
    lut0 = proc.generate_lut_tensor(next(s for s in proc.catalog() if s.lut_id == "proc_attr_warmer"))
    source = _node_pixels(40)
    target = apply_lut_trilinear(lut0, source)
    fit = fit_global_lut(source, target, min_support=32, smooth=False)
    assert fit.supported_mask.all()  # every node well-supported
    assert np.max(np.abs(fit.lut_abs - lut0)) < 1e-9


def test_pair_fit_empty_nodes_fall_back_to_identity():
    # raw estimator (smooth=False): unobserved nodes snap to identity
    src = np.tile(np.array([0.0, 0.0, 0.0]), (100, 1))
    tgt = np.tile(np.array([0.1, 0.05, 0.0]), (100, 1))
    fit = fit_global_lut(src, tgt, min_support=32, smooth=False)
    assert fit.empty_mask.sum() > 0
    ident = identity_grid(17)
    # a far, empty node equals identity
    assert np.allclose(fit.lut_abs[16, 16, 16], ident[16, 16, 16])


def test_smooth_fill_recovers_known_lut_when_fully_observed():
    # smooth fill (default) still reproduces a fully-observed smooth LUT to a tight tolerance:
    # the neighbour term barely moves nodes that already sit on a smooth surface.
    lut0 = proc.generate_lut_tensor(next(s for s in proc.catalog() if s.lut_id == "proc_attr_warmer"))
    source = _node_pixels(40)
    target = apply_lut_trilinear(lut0, source)
    fit = fit_global_lut(source, target, min_support=32)  # smooth=True
    assert np.max(np.abs(fit.lut_abs - lut0)) < 0.02


def test_smooth_fill_is_smoother_than_identity_fallback_on_sparse_coverage():
    # a smooth global LUT observed on only a sparse slice of the cube: the smooth fill must be
    # markedly smoother + more monotonic than the identity-fallback estimator on the same data.
    from data_pipeline.quality_filters import assess_quality

    lut0 = proc.generate_lut_tensor(next(s for s in proc.catalog() if s.lut_id == "proc_attr_warmer"))
    rng = np.random.default_rng(0)
    # a curved 2D-ish colour manifold (like a photo) -> most of the 17^3 nodes unobserved
    t = rng.random(20000)
    src = np.clip(np.stack([0.15 + 0.7 * t, 0.2 + 0.6 * t ** 2, 0.25 + 0.5 * t ** 0.5], axis=1), 0, 1)
    tgt = apply_lut_trilinear(lut0, src)
    raw = fit_global_lut(src, tgt, smooth=False)
    smooth = fit_global_lut(src, tgt, smooth=True)
    assert (raw.empty_mask.sum() / raw.empty_mask.size) > 0.5  # genuinely sparse
    q_raw = assess_quality(raw.lut_abs)
    q_smooth = assess_quality(smooth.lut_abs)
    assert q_smooth.quality_scores["smoothness"] < q_raw.quality_scores["smoothness"]
    assert q_smooth.quality_scores["foldover_rate"] <= q_raw.quality_scores["foldover_rate"]


def test_cube_decode_roundtrip():
    lut = identity_grid(17)
    back = cube_bytes_to_lut(serialize_cube(lut))
    assert np.allclose(back, lut, atol=1e-9)
