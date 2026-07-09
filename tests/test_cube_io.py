"""Known-answer tests for canonical .cube serialization + identity grid."""

import numpy as np

from eval.cube_io import (
    absolute_to_residual,
    cube_bytes_hash,
    identity_grid,
    parse_cube,
    residual_to_absolute,
    serialize_cube,
)


def test_identity_grid_shape_and_nodes():
    g = identity_grid(17)
    assert g.shape == (17, 17, 17, 3)
    # node i -> i/16 along its axis
    assert np.allclose(g[0, 0, 0], [0.0, 0.0, 0.0])
    assert np.allclose(g[16, 16, 16], [1.0, 1.0, 1.0])
    assert np.allclose(g[8, 0, 0], [8 / 16, 0.0, 0.0])


def test_identity_residual_is_zero():
    res = absolute_to_residual(identity_grid(17))
    assert np.abs(res).max() == 0.0


def test_serialize_deterministic_and_hash_stable():
    g = identity_grid(17)
    b1 = serialize_cube(g)
    b2 = serialize_cube(g)
    assert b1 == b2
    assert cube_bytes_hash(g) == cube_bytes_hash(g)


def test_header_and_format():
    g = identity_grid(17)
    text = serialize_cube(g).decode("utf-8")
    lines = text.split("\n")
    assert lines[0] == "LUT_3D_SIZE 17"
    assert lines[1] == "DOMAIN_MIN 0 0 0"
    assert lines[2] == "DOMAIN_MAX 1 1 1"
    # LF only, no CR
    assert "\r" not in text
    # 3 header + 17^3 data + trailing newline
    data_lines = [ln for ln in lines[3:] if ln]
    assert len(data_lines) == 17 ** 3
    # 10-decimal formatting
    assert data_lines[0] == "0.0000000000 0.0000000000 0.0000000000"


def test_negative_zero_normalized():
    g = identity_grid(17).copy()
    g[0, 0, 0, 0] = -0.0
    first_data = serialize_cube(g).decode("utf-8").split("\n")[3]
    assert first_data.split()[0] == "0.0000000000"
    assert "-0.0000000000" not in first_data


def test_roundtrip_identity_and_nonidentity():
    g = identity_grid(17)
    back, hdr = parse_cube(serialize_cube(g))
    assert hdr["size"] == 17
    assert np.allclose(back, g, atol=1e-9)

    # non-identity: identity + a constant residual
    res = np.full((17, 17, 17, 3), 0.05)
    absol = residual_to_absolute(res, identity_grid(17))
    back2, _ = parse_cube(serialize_cube(absol))
    assert np.allclose(back2, absol, atol=1e-9)


def test_r_fastest_table_order():
    # craft a LUT where output encodes (r,g,b) so we can check ordering
    g = identity_grid(3)
    lines = [ln for ln in serialize_cube(g).decode().split("\n")[3:] if ln]
    # first line is (r=0,g=0,b=0); second line must advance r (fastest)
    assert lines[0].startswith("0.0000000000 0.0000000000 0.0000000000")
    assert lines[1].startswith("0.5000000000 0.0000000000 0.0000000000")  # r=1/2, g=0, b=0
