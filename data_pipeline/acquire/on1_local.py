"""ON1 free LUT pack connector (local directory).

Ingests the user's ``ON1_All_LUTs`` pack from a local dir (``$SLM_ON1_DIR`` or default
``~/Downloads/ON1_All_LUTs``): copies each category's ``.cube`` LUTs into ``luts/raw/on1/<cat>/``
and registers them with category-derived style/scene tags. 80/90 are AdobeRGB-authored (headers
name ``AdobeRGB1998.icc``) -> ``declared_domain="adobe_rgb"`` (color-managed at canonicalize); the
rest default to sRGB. ``family="smaller_public_packs"`` so it inherits that source's selection cap.
Resumable (skips cubes already copied).
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from . import downloaders as dl
from .base import AcquireLimits, AcquireReport, RawArtifact, utcnow_iso

_DEFAULT_DIR = "~/Downloads/ON1_All_LUTs"

# category (from "ON1 <Category> LUTs" dir) -> style/scene tags. Kept as descriptive scene/style
# labels (not attribute claims) so they don't trip the tag<->behavior direction-magnitude checks.
_CATEGORY_TAGS = {
    "black & white": ["black_and_white", "monochrome"],
    "cinematic": ["cinematic"],
    "color boost": ["vibrant", "color_boost"],
    "landscape": ["landscape"],
    "lifestyle & commercial": ["lifestyle", "commercial"],
    "lutify.me": ["cinematic", "film"],
    "moody": ["moody"],
    "nature & wildlife": ["nature", "wildlife"],
    "portrait": ["portrait"],
}


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _category_of(parts) -> str | None:
    for p in parts:
        m = re.match(r"ON1 (.+) LUTs", p)
        if m:
            return m.group(1)
    return None


class ON1Connector:
    source_pack_id = "on1_lut_packs"
    family = "smaller_public_packs"

    def __init__(self, src_dir: str | None = None):
        self.src_dir = Path(os.path.expanduser(
            src_dir or os.environ.get("SLM_ON1_DIR", _DEFAULT_DIR)))

    def verify(self) -> tuple[bool, str]:
        if not self.src_dir.exists():
            return False, f"ON1 dir not found: {self.src_dir}"
        cubes = list(self.src_dir.rglob("*.cube"))
        return (len(cubes) > 0), f"{len(cubes)} .cube files in {self.src_dir.name}"

    @staticmethod
    def _header_info(path: Path) -> tuple[str | None, bool]:
        """Return (preset title, is_adobe_rgb) from the cube header comments."""
        title, adobe = None, False
        try:
            with open(path, "r", errors="ignore") as fh:
                for _ in range(40):
                    line = fh.readline()
                    if not line or line.strip().upper().startswith("LUT_3D_SIZE"):
                        break
                    m = re.match(r"#\s*Preset:\s*(.+)", line)
                    if m:
                        title = m.group(1).strip()
                    if "adobergb" in line.lower() or "adobe rgb" in line.lower():
                        adobe = True
        except Exception:  # noqa: BLE001
            pass
        return title, adobe

    def acquire(self, raw_root, limits: AcquireLimits) -> AcquireReport:
        report = AcquireReport(source_pack_id=self.source_pack_id)
        if not self.src_dir.exists():
            report.status = "skipped"
            report.note = f"dir not found: {self.src_dir}"
            return report
        dest_root = Path(raw_root) / "on1"
        cubes = sorted(self.src_dir.rglob("*.cube"))
        if limits.max_items is not None:
            cubes = cubes[: limits.max_items]
        ts = utcnow_iso()
        seen: set[str] = set()
        for src in cubes:
            report.attempted += 1
            rel_parts = src.relative_to(self.src_dir).parts
            category = _category_of([self.src_dir.name, *rel_parts])
            cat_slug = _slug(category) if category else "misc"
            tags = _CATEGORY_TAGS.get((category or "").lower(), [cat_slug])
            title, adobe = self._header_info(src)
            name = _slug(title or src.stem)
            lut_id = f"on1_{cat_slug}_{name}"
            if lut_id in seen:            # same preset exported for multiple programs -> once
                report.skipped += 1
                continue
            seen.add(lut_id)
            dest = dest_root / cat_slug / f"{name}.cube"
            if dest.exists() and dest.stat().st_size > 0:
                report.skipped += 1
            else:
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dest)
                report.acquired += 1
            report.artifacts.append(RawArtifact(
                kind="lut_file", source_pack_id=self.source_pack_id, family=self.family,
                declared_domain=("adobe_rgb" if adobe else "srgb"),
                license="ON1 free LUT pack (personal-use)",
                source_url=f"local:ON1_All_LUTs/{cat_slug}",
                file_hash=(dl.sha256_file(dest) if dest.exists() else None),
                download_timestamp=ts, lut_id=lut_id, file_path=str(dest),
                derivation_method="cube", author_uploader_pack_id="on1",
                gold_tags=tags, style_bundle=cat_slug,
                extra={"title": title, "adobe_rgb": adobe},
            ))
        report.status = "ok" if (report.acquired or report.skipped) else "partial"
        return report
