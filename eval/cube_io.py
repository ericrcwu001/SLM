"""Canonical ``.cube`` serialization + the encoded-sRGB identity grid.

Pure and deterministic — no model, no color-management. This locks the pinned
serialization format now (cheap, high value); it is exercised by its own unit tests
and by the decoder/CLI later, and is NOT on the live decode-disabled eval path.

Serialization contract ``cube_v1_size17_domain01_rgb_rfast_f10_lf``
(model_architecture.md "Canonical LUT Domain" / detailed_behavior_spec.md):
  * ``LUT_3D_SIZE 17``, ``DOMAIN_MIN 0 0 0``, ``DOMAIN_MAX 1 1 1``
  * table order: RGB with **R changing fastest**, then G, then B
  * fixed 10-decimal float formatting, LF line endings, UTF-8, no timestamps/comments

Tensor convention (pinned, audit A4): LUTs are stored as ``lut[r, g, b, channel]`` on
a 17-node grid; node ``i`` maps to input value ``i / (N-1)``. Serialization walks
``b`` (outer), ``g`` (mid), ``r`` (inner) so R varies fastest — consistent with the
``.cube`` R-fastest table order.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .schemas import CUBE_SERIALIZATION_VERSION

GRID_SIZE = 17


def cube_serialization_version() -> str:
    return CUBE_SERIALIZATION_VERSION


def identity_grid(size: int = GRID_SIZE) -> np.ndarray:
    """Encoded-sRGB identity LUT: ``grid[r,g,b] = [r/(N-1), g/(N-1), b/(N-1)]``.

    Node values are ``i/(size-1)`` (audit item B1).
    """
    axis = np.linspace(0.0, 1.0, size, dtype=np.float64)
    r = axis[:, None, None]
    g = axis[None, :, None]
    b = axis[None, None, :]
    grid = np.empty((size, size, size, 3), dtype=np.float64)
    grid[..., 0] = np.broadcast_to(r, (size, size, size))
    grid[..., 1] = np.broadcast_to(g, (size, size, size))
    grid[..., 2] = np.broadcast_to(b, (size, size, size))
    return grid


def residual_to_absolute(residual: np.ndarray, identity: np.ndarray | None = None) -> np.ndarray:
    if identity is None:
        identity = identity_grid(residual.shape[0])
    return residual + identity


def absolute_to_residual(absolute: np.ndarray, identity: np.ndarray | None = None) -> np.ndarray:
    if identity is None:
        identity = identity_grid(absolute.shape[0])
    return absolute - identity


def _fmt(x: float) -> str:
    # normalize -0.0 -> 0.0 so byte output is deterministic
    v = float(x) + 0.0
    return f"{v:.10f}"


def serialize_cube(lut_abs: np.ndarray) -> bytes:
    """Serialize a canonical absolute LUT (``[N,N,N,3]``) to ``.cube`` bytes."""
    if lut_abs.ndim != 4 or lut_abs.shape[3] != 3 or len(set(lut_abs.shape[:3])) != 1:
        raise ValueError(f"expected [N,N,N,3] LUT, got shape {lut_abs.shape}")
    n = lut_abs.shape[0]
    lines = [
        f"LUT_3D_SIZE {n}",
        "DOMAIN_MIN 0 0 0",
        "DOMAIN_MAX 1 1 1",
    ]
    # R fastest: b outer, g mid, r inner
    for b in range(n):
        for g in range(n):
            for r in range(n):
                rr, gg, bb = lut_abs[r, g, b]
                lines.append(f"{_fmt(rr)} {_fmt(gg)} {_fmt(bb)}")
    text = "\n".join(lines) + "\n"
    return text.encode("utf-8")


def write_cube(path: str, lut_abs: np.ndarray) -> None:
    with open(path, "wb") as fh:
        fh.write(serialize_cube(lut_abs))


def cube_bytes_hash(lut_abs: np.ndarray) -> str:
    """SHA-256 of the canonical serialized bytes."""
    return hashlib.sha256(serialize_cube(lut_abs)).hexdigest()


def parse_cube(data: bytes | str) -> tuple[np.ndarray, dict]:
    """Parse ``.cube`` bytes back into ``lut[r,g,b,3]`` + a header dict.

    Reads the R-fastest table order (b outer, g mid, r inner). Used for roundtrip
    tests and for parsing renderer/frontier ``.cube`` outputs later.
    """
    if isinstance(data, bytes):
        data = data.decode("utf-8")
    size: int | None = None
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]
    rows: list[tuple[float, float, float]] = []
    for raw in data.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        key = parts[0].upper()
        if key == "LUT_3D_SIZE":
            size = int(parts[1])
        elif key == "DOMAIN_MIN":
            domain_min = [float(x) for x in parts[1:4]]
        elif key == "DOMAIN_MAX":
            domain_max = [float(x) for x in parts[1:4]]
        elif key in ("TITLE", "LUT_1D_SIZE", "LUT_3D_INPUT_RANGE"):
            continue
        else:
            rows.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if size is None:
        raise ValueError("missing LUT_3D_SIZE")
    if len(rows) != size**3:
        raise ValueError(f"expected {size**3} rows, got {len(rows)}")
    lut = np.empty((size, size, size, 3), dtype=np.float64)
    i = 0
    for b in range(size):
        for g in range(size):
            for r in range(size):
                lut[r, g, b] = rows[i]
                i += 1
    header = {
        "size": size,
        "domain_min": domain_min,
        "domain_max": domain_max,
        "serialization_version": CUBE_SERIALIZATION_VERSION,
    }
    return lut, header
