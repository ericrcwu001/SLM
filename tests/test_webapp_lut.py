"""Focused acceptance tests for the webapp LUT adapter layer."""

from __future__ import annotations

import numpy as np
from PIL import Image

from eval.cube_io import identity_grid, parse_cube
from webapp import lut as lut_module
from webapp.lut import apply_lut, export_cube, load_image, save_image


def test_decode_delegates_to_frozen_decoder(monkeypatch) -> None:
    expected = identity_grid(17)
    seen = {}

    def fake_decode_codes(codes, *, final_dir=None):
        seen["args"] = (codes, final_dir)
        return expected

    monkeypatch.setattr(lut_module, "decode_codes", fake_decode_codes)

    actual = lut_module.decode(list(range(64)), final_dir="tokenizer/final")

    assert actual is expected
    assert seen["args"] == (list(range(64)), "tokenizer/final")


def test_identity_lut_is_a_trilinear_roundtrip() -> None:
    rng = np.random.default_rng(20260712)
    image = rng.random((13, 19, 3))

    graded = apply_lut(image, identity_grid(17))

    assert graded.shape == image.shape
    assert np.allclose(graded, image, atol=1e-12)


def test_apply_lut_clips_inputs_and_outputs() -> None:
    lut = np.clip(identity_grid(17) + 0.25, 0.0, 1.0)
    image = np.array([[[-0.5, 0.5, 1.5]]], dtype=np.float64)

    graded = apply_lut(image, lut)

    assert np.allclose(graded, [[[0.25, 0.75, 1.0]]], atol=1e-12)
    assert np.all((0.0 <= graded) & (graded <= 1.0))


def test_export_cube_is_valid_17_cube_with_r_fastest_order(tmp_path) -> None:
    path = tmp_path / "identity.cube"
    identity = identity_grid(17)

    export_cube(identity, path)

    raw = path.read_bytes()
    assert b"\r" not in raw
    lines = raw.decode("utf-8").splitlines()
    assert lines[:3] == [
        "LUT_3D_SIZE 17",
        "DOMAIN_MIN 0 0 0",
        "DOMAIN_MAX 1 1 1",
    ]
    assert len(lines[3:]) == 17**3 == 4913
    assert lines[3] == "0.0000000000 0.0000000000 0.0000000000"
    assert lines[4] == "0.0625000000 0.0000000000 0.0000000000"
    assert lines[3 + 17] == "0.0000000000 0.0625000000 0.0000000000"

    parsed, header = parse_cube(raw)
    assert header["size"] == 17
    assert header["domain_min"] == [0.0, 0.0, 0.0]
    assert header["domain_max"] == [1.0, 1.0, 1.0]
    assert np.allclose(parsed, identity, atol=1e-10)


def test_export_cube_clips_to_canonical_domain(tmp_path) -> None:
    path = tmp_path / "clipped.cube"
    lut = identity_grid(17) * 2.0 - 0.5

    export_cube(lut, path)

    parsed, _ = parse_cube(path.read_bytes())
    assert np.allclose(parsed, np.clip(lut, 0.0, 1.0), atol=1e-10)


def test_image_io_normalizes_to_rgb_and_preserves_uint8(tmp_path) -> None:
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    Image.new("L", (3, 2), color=127).save(source)

    loaded = load_image(source)
    pixels = np.full((2, 3, 3), [12, 34, 56], dtype=np.uint8)
    save_image(pixels, output)

    assert loaded.mode == "RGB"
    assert loaded.size == (3, 2)
    assert np.array_equal(np.asarray(load_image(output)), pixels)
