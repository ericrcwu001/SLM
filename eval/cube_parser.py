"""Strict parser for a *frontier model's* raw ``.cube`` output.

The prompt-to-LUT eval harness normally scores 64 VQ code tokens (eval/output_parsers.py).
A prompted frontier model (Opus / Sonnet) cannot emit those tokens — no VQ tokenizer exists
yet — so for the prompted-frontier baseline it emits a raw ``.cube`` LUT directly, which this
module parses into a canonical absolute LUT ``[17,17,17,3]`` for behavioral scoring.

This is a thin, strict validator on top of :func:`eval.cube_io.parse_cube` (which is the
authority for the canonical R-fastest table order). "Strict" per the pilot spec means the
LUT itself must be the canonical 17^3, domain-0..1 grid; we tolerate only *wrapping* noise
around it (markdown ``` fences, a TITLE line, trailing prose) because rejecting a good LUT
for a stray backtick would measure formatting compliance, not LUT quality.

Classification (mirrors output_parsers.parse_output so the boundary metrics are shared):
  * exact ``<unsupported>``                         -> kind="unsupported"
  * a valid canonical 17^3 .cube                    -> kind="raw_lut"
  * anything else (wrong size/domain, truncated,
    out-of-range, non-finite, unparseable, mixed)   -> kind="invalid"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import cube_io
from .vocab import UNSUPPORTED

RAW_LUT = "raw_lut"
UNSUPPORTED_KIND = "unsupported"
INVALID = "invalid"

EXPECTED_SIZE = cube_io.GRID_SIZE  # 17
# node values must lie in the display-encoded [0,1] domain; allow a little float slop
# from the model, then clip. Beyond this band the grid is garbage, not rounding noise.
_RANGE_TOL = 0.02


@dataclass
class ParsedCube:
    kind: str  # RAW_LUT | UNSUPPORTED_KIND | INVALID
    lut_abs: Optional[np.ndarray] = None  # [N,N,N,3] absolute LUT when kind == RAW_LUT
    size: Optional[int] = None
    errors: list[str] = field(default_factory=list)

    @property
    def syntax_pass(self) -> bool:
        return self.kind in (RAW_LUT, UNSUPPORTED_KIND)


def _is_triple(parts: list[str]) -> Optional[tuple[float, float, float]]:
    """Return a float triple if the first three tokens parse as floats, else None."""
    if len(parts) < 3:
        return None
    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


def parse_frontier_cube(text: Optional[str], expected_size: int = EXPECTED_SIZE) -> ParsedCube:
    if text is None:
        return ParsedCube(INVALID, errors=["null_output"])

    s = text.strip()
    if s == UNSUPPORTED:
        return ParsedCube(UNSUPPORTED_KIND)
    if not s:
        return ParsedCube(INVALID, errors=["empty_output"])
    # a mixed refusal-plus-LUT output is not a clean decision either way
    if UNSUPPORTED in s and "LUT_3D_SIZE" in s.upper():
        return ParsedCube(INVALID, errors=["mixed_unsupported_and_lut"])

    size: Optional[int] = None
    domain_min = [0.0, 0.0, 0.0]
    domain_max = [1.0, 1.0, 1.0]
    rows: list[tuple[float, float, float]] = []
    for raw in s.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        parts = line.split()
        key = parts[0].upper()
        if key == "LUT_3D_SIZE":
            try:
                size = int(parts[1])
            except (IndexError, ValueError):
                return ParsedCube(INVALID, errors=["bad_lut_3d_size"])
        elif key == "DOMAIN_MIN" and len(parts) >= 4:
            domain_min = [float(x) for x in parts[1:4]]
        elif key == "DOMAIN_MAX" and len(parts) >= 4:
            domain_max = [float(x) for x in parts[1:4]]
        elif key in ("TITLE", "LUT_1D_SIZE", "LUT_3D_INPUT_RANGE"):
            continue
        else:
            triple = _is_triple(parts)
            if triple is not None:
                rows.append(triple)
            # non-numeric prose lines are ignored (wrapping tolerance)

    if size is None:
        return ParsedCube(INVALID, errors=["no_lut_3d_size_header"])
    if size != expected_size:
        return ParsedCube(INVALID, size=size, errors=[f"size_{size}_not_{expected_size}"])

    need = size**3
    if len(rows) < need:
        return ParsedCube(INVALID, size=size, errors=[f"truncated_{len(rows)}_of_{need}_rows"])
    rows = rows[:need]  # tolerate trailing prose beyond the table

    # strict domain: must be the canonical 0..1 grid
    if not (np.allclose(domain_min, [0, 0, 0], atol=1e-3) and np.allclose(domain_max, [1, 1, 1], atol=1e-3)):
        return ParsedCube(INVALID, size=size, errors=[f"non_canonical_domain:{domain_min}->{domain_max}"])

    arr = np.array(rows, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        return ParsedCube(INVALID, size=size, errors=["non_finite_values"])
    if arr.min() < -_RANGE_TOL or arr.max() > 1.0 + _RANGE_TOL:
        return ParsedCube(INVALID, size=size,
                          errors=[f"out_of_range:[{arr.min():.4f},{arr.max():.4f}]"])

    # Reconstruct a clean canonical .cube and let cube_io.parse_cube own the R-fastest
    # ordering (single source of truth for the table layout).
    body = "\n".join(f"{r:.10f} {g:.10f} {b:.10f}" for (r, g, b) in rows)
    clean = f"LUT_3D_SIZE {size}\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n{body}\n"
    try:
        lut_abs, _ = cube_io.parse_cube(clean)
    except Exception as exc:  # noqa: BLE001 - any reconstruction failure is an invalid output
        return ParsedCube(INVALID, size=size, errors=[f"parse_error:{exc}"])

    lut_abs = np.clip(lut_abs, 0.0, 1.0)
    return ParsedCube(RAW_LUT, lut_abs=lut_abs, size=size)
