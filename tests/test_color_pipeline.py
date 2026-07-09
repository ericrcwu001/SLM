"""Known-answer tests for the color pipeline (sRGB<->linear, sRGB->Lab D65, CIEDE2000).

CIEDE2000 is validated against the canonical Sharma et al. (2005) reference test data.
"""

import numpy as np

from eval import color_pipeline as cp

# Sharma et al. CIEDE2000 test pairs: (L1,a1,b1, L2,a2,b2, expected_dE00).
# Subset of the published 34-pair reference table (the discontinuity/edge cases).
_SHARMA = [
    (50.0000, 2.6772, -79.7751, 50.0000, 0.0000, -82.7485, 2.0425),
    (50.0000, 3.1571, -77.2803, 50.0000, 0.0000, -82.7485, 2.8615),
    (50.0000, 2.8361, -74.0200, 50.0000, 0.0000, -82.7485, 3.4412),
    (50.0000, -1.3802, -84.2814, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -1.1848, -84.8006, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, -0.9009, -85.5211, 50.0000, 0.0000, -82.7485, 1.0000),
    (50.0000, 0.0000, 0.0000, 50.0000, -1.0000, 2.0000, 2.3669),
    (50.0000, -1.0000, 2.0000, 50.0000, 0.0000, 0.0000, 2.3669),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0009, 7.1792),
    (50.0000, 2.4900, -0.0010, 50.0000, -2.4900, 0.0011, 7.2195),
    (50.0000, 2.5000, 0.0000, 50.0000, 0.0000, -2.5000, 4.3065),
    (50.0000, 2.5000, 0.0000, 73.0000, 25.0000, -18.0000, 27.1492),
    (50.0000, 2.5000, 0.0000, 61.0000, -5.0000, 29.0000, 22.8977),
    (50.0000, 2.5000, 0.0000, 56.0000, -27.0000, -3.0000, 31.9030),
    (50.0000, 2.5000, 0.0000, 58.0000, 24.0000, 15.0000, 19.4535),
    (60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644),
    (63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.2630),
    (35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645),
    (22.7233, 20.0904, -46.6940, 23.0331, 14.9730, -42.5619, 2.0373),
    (36.4612, 47.8580, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146),
    (90.8027, -2.0831, 1.4410, 91.1528, -1.6435, 0.0447, 1.4441),
    (90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381),
    (6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377),
]


def test_ciede2000_sharma_reference():
    for L1, a1, b1, L2, a2, b2, expected in _SHARMA:
        got = float(cp.ciede2000(np.array([L1, a1, b1]), np.array([L2, a2, b2])))
        assert abs(got - expected) < 1e-4, f"pair {(L1,a1,b1,L2,a2,b2)}: {got} != {expected}"


def test_ciede2000_vectorized_matches_scalar():
    a = np.array([[50.0, 2.6772, -79.7751], [60.2574, -34.0099, 36.2677]])
    b = np.array([[50.0, 0.0, -82.7485], [60.4626, -34.1751, 39.4387]])
    out = cp.ciede2000(a, b)
    assert out.shape == (2,)
    assert abs(out[0] - 2.0425) < 1e-4
    assert abs(out[1] - 1.2644) < 1e-4


def test_ciede2000_zero_for_identical():
    lab = np.array([42.0, -5.0, 12.0])
    assert float(cp.ciede2000(lab, lab)) == 0.0


def test_srgb_linear_roundtrip():
    x = np.linspace(0.0, 1.0, 257)
    back = cp.linear_to_srgb(cp.srgb_to_linear(x))
    assert np.allclose(back, x, atol=1e-9)


def test_srgb_to_lab_known_points():
    # white -> L*=100, a*=b*=0 (tolerance covers rounding in the published sRGB->XYZ matrix)
    white = cp.srgb_to_lab_d65(np.array([1.0, 1.0, 1.0]))
    assert abs(white[0] - 100.0) < 1e-4
    assert abs(white[1]) < 1e-2 and abs(white[2]) < 1e-2
    # black -> L*=0
    black = cp.srgb_to_lab_d65(np.array([0.0, 0.0, 0.0]))
    assert abs(black[0]) < 1e-6
    # mid gray 0.5 sRGB -> L* around 53.4
    gray = cp.srgb_to_lab_d65(np.array([0.5, 0.5, 0.5]))
    assert 53.0 < gray[0] < 54.0
    assert abs(gray[1]) < 1e-3 and abs(gray[2]) < 1e-3


def test_chroma_hue_and_masks():
    # pure warm push: +b* -> hue near 90 deg, positive chroma
    lab = np.array([50.0, 0.0, 10.0])
    assert abs(cp.chroma(lab) - 10.0) < 1e-9
    assert abs(cp.hue_deg(lab) - 90.0) < 1e-6
    labs = np.array([[80.0, 0.0, 0.0], [10.0, 0.0, 0.0]])
    assert cp.highlight_mask(labs).tolist() == [True, False]
    assert cp.shadow_mask(labs).tolist() == [False, True]


def test_identity_lut_zero_deltae():
    from eval.cube_io import identity_grid

    grid = identity_grid(17).reshape(-1, 3)
    dE = cp.deltae2000_srgb(grid, grid)
    assert float(np.max(dE)) == 0.0


def test_lab_srgb_roundtrip_in_gamut():
    # in-gamut sRGB colors survive srgb->lab->srgb within a tight tolerance
    rng = np.random.default_rng(0)
    rgb = rgb = np.clip(0.15 + 0.7 * rng.random((500, 3)), 0.0, 1.0)
    lab = cp.srgb_to_lab_d65(rgb)
    back = cp.lab_d65_to_srgb(lab)
    assert np.max(np.abs(back - rgb)) < 1e-4


def test_lab_delta_b_is_warm():
    # +b* shift maps to a real sRGB color; re-measured b* stays higher
    base = cp.srgb_to_lab_d65(np.array([0.5, 0.5, 0.5]))
    warm = base.copy()
    warm[2] += 12.0
    warm_rgb = cp.lab_d65_to_srgb(warm)
    assert cp.srgb_to_lab_d65(warm_rgb)[2] > base[2] + 6.0


def test_to_canonical_srgb_uint8_and_clip():
    img = np.array([[0, 128, 255]], dtype=np.uint8)
    out = cp.to_canonical_srgb(img)
    assert out.dtype == np.float32
    assert abs(out[0, 0] - 0.0) < 1e-6 and abs(out[0, 2] - 1.0) < 1e-6
    over = cp.to_canonical_srgb(np.array([1.5, -0.2, 0.5], dtype=np.float32))
    assert over.max() <= 1.0 and over.min() >= 0.0


# --- AdobeRGB(1998) <-> sRGB color management -------------------------------------
def test_adobe_srgb_roundtrip_in_gamut_identity():
    # sRGB is inside AdobeRGB's gamut, so srgb->adobe->srgb must round-trip to ~identity.
    rng = np.random.default_rng(0)
    xs = rng.random((500, 3))
    rt = cp.adobe_rgb_to_srgb(cp.srgb_to_adobe_rgb(xs))
    assert np.max(np.abs(rt - xs)) < 1e-9


def test_adobe_neutral_gray_stays_neutral():
    # both spaces share the D65 white point -> a neutral gray stays neutral (ΔE00 ~ 0).
    g = np.array([[0.5, 0.5, 0.5]])
    back = cp.adobe_rgb_to_srgb(cp.srgb_to_adobe_rgb(g))
    dE = float(cp.ciede2000(cp.srgb_to_lab_d65(g), cp.srgb_to_lab_d65(back)).reshape(-1)[0])
    assert dE < 1e-6
    # AdobeRGB encodes gray slightly lower than sRGB (steeper gamma), never above.
    assert cp.srgb_to_adobe_rgb(g)[0, 0] <= 0.5 + 1e-9


def test_adobe_wide_gamut_color_clips_into_srgb():
    # AdobeRGB pure green is outside sRGB -> managed result must be a valid clipped sRGB color.
    out = cp.adobe_rgb_to_srgb(np.array([[0.0, 1.0, 0.0]]))
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out[0, 1] > out[0, 0] and out[0, 1] > out[0, 2]  # still green-dominant
