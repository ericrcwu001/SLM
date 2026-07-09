"""Tests for the procedural LUT generator."""

import numpy as np

from data_pipeline.sources import procedural as proc
from eval import color_pipeline as cp
from eval.cube_io import absolute_to_residual, parse_cube


def _spec(name):
    return next(s for s in proc.catalog() if s.lut_id == name)


def test_catalog_covers_attributes_and_styles():
    kinds = {s.kind for s in proc.catalog()}
    assert kinds == {"attribute", "style"}
    styles = {s.style for s in proc.catalog() if s.kind == "style"}
    assert {"matte", "faded", "filmic", "cinematic", "teal-orange", "sepia",
            "bleach bypass", "natural"} <= styles


def test_warmer_raises_b_star():
    lut = proc.generate_lut_tensor(_spec("proc_attr_warmer"))
    # mid-gray node [8,8,8] output should be warmer (higher b*) than identity mid-gray
    mid_out = cp.srgb_to_lab_d65(lut[8, 8, 8])
    mid_id = cp.srgb_to_lab_d65(np.array([0.5, 0.5, 0.5]))
    assert mid_out[2] > mid_id[2] + 3.0


def test_cooler_lowers_b_star():
    lut = proc.generate_lut_tensor(_spec("proc_attr_cooler"))
    mid_out = cp.srgb_to_lab_d65(lut[8, 8, 8])
    mid_id = cp.srgb_to_lab_d65(np.array([0.5, 0.5, 0.5]))
    assert mid_out[2] < mid_id[2] - 3.0


def test_brighter_raises_l_star():
    lut = proc.generate_lut_tensor(_spec("proc_attr_brighter"))
    assert cp.srgb_to_lab_d65(lut[8, 8, 8])[0] > cp.srgb_to_lab_d65(np.array([0.5, 0.5, 0.5]))[0] + 3.0


def test_natural_is_small_residual():
    lut = proc.generate_lut_tensor(_spec("proc_style_natural"))
    res = absolute_to_residual(lut)
    assert float(np.sqrt(np.mean(res ** 2))) < 0.05


def test_generate_writes_valid_cubes(tmp_path):
    gens = proc.generate(tmp_path, magnitudes=(1.0,))
    assert len(gens) == 14 + 8  # attributes x1 magnitude + 8 styles
    for g in gens:
        assert g.path.exists()
        lut, header = parse_cube(g.path.read_bytes())
        assert header["size"] == 17
        assert lut.shape == (17, 17, 17, 3)
        assert len(g.file_hash) == 64
