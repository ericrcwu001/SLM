"""Scraped-web LUT connector (local ingestion of the discovery-swarm downloads).

Registers the LUTs pulled into ``luts/raw/web/`` + ``luts/raw/shutterstock_log_luts/`` by the
web-scraping workflows. Handles ``.cube`` (derivation_method=cube) and HaldCLUT ``.png``
(derivation_method=haldclut). Content-hash dedup against the existing corpus AND within the scrape
(the swarm re-fetched some packs from several sites, and gmic.eu / freshluts.com overlap our
existing gmic/freshluts sources). ``.3dl`` / ``.look`` are skipped (no parser yet). Colour domain is
assumed sRGB; genuine log/video LUTs will be correctly rejected by the sRGB-display gates.

Personal-use ingestion, non-redistribution (per the project owner). No network I/O.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from ..lut_ops import hald_level_and_edge
from .base import AcquireLimits, AcquireReport, RawArtifact, utcnow_iso

_SCRAPE_SUBDIRS = ("web", "shutterstock_log_luts", "gated")
# folder/name tokens worth keeping as informational style hints
_STYLE_HINTS = {
    "cinematic", "vintage", "retro", "film", "moody", "portrait", "landscape", "nature",
    "wildlife", "teal", "orange", "bw", "mono", "black", "white", "warm", "cool", "fade",
    "faded", "matte", "kodak", "fuji", "polaroid", "wedding", "travel", "urban", "vlog",
    "drone", "log", "rec709", "blockbuster", "analog", "sepia", "duotone",
}


def _sha256(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:  # noqa: BLE001
        return None


def _is_hald_png(path: Path) -> bool:
    try:
        from PIL import Image

        w, h = Image.open(path).size
        if w != h:
            return False
        hald_level_and_edge(w)  # raises if side is not level**3
        return True
    except Exception:  # noqa: BLE001
        return False


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60]


class ScrapedWebConnector:
    source_pack_id = "scraped_web"
    family = "scraped_web"

    def __init__(self, registry_path: str | Path | None = None):
        # existing provenance, used to seed the content-hash dedup set
        self.registry_path = Path(registry_path) if registry_path else None

    def verify(self) -> tuple[bool, str]:
        return True, "local scraped-web ingestion"

    def _existing_hashes(self, raw_root: Path) -> set[str]:
        """sha256 of every already-registered LUT *file* (cube/hald/generated), to skip re-adds."""
        seen: set[str] = set()
        reg = self.registry_path or (raw_root.parent.parent / "data" / "raw_registry" / "provenance.jsonl")
        if not Path(reg).exists():
            return seen
        for line in open(reg, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            if d.get("derivation_method") in ("cube", "haldclut", "generated"):
                p = d.get("derivation_path")
                if p and Path(p).exists():
                    h = _sha256(Path(p))
                    if h:
                        seen.add(h)
        return seen

    def acquire(self, raw_root, limits: AcquireLimits) -> AcquireReport:
        report = AcquireReport(source_pack_id=self.source_pack_id)
        raw_root = Path(raw_root)
        roots = [raw_root / s for s in _SCRAPE_SUBDIRS if (raw_root / s).exists()]
        if not roots:
            report.status = "skipped"
            report.note = "no scraped dirs (luts/raw/web, shutterstock_log_luts)"
            return report

        seen = self._existing_hashes(raw_root)
        report.note = f"{len(seen)} existing-LUT hashes seeded for dedup"
        ts = utcnow_iso()
        skipped_formats = 0
        files: list[Path] = []
        for root in roots:
            files.extend(sorted(root.rglob("*")))

        for f in files:
            if not f.is_file():
                continue
            ext = f.suffix.lower()
            if ext == ".cube":
                method = "cube"
            elif ext == ".png" and _is_hald_png(f):
                method = "haldclut"
            else:
                if ext in (".3dl", ".look", ".png"):
                    skipped_formats += 1
                continue
            report.attempted += 1
            h = _sha256(f)
            if h is None:
                report.failed += 1
                continue
            if h in seen:                      # exact-content dup (existing corpus or earlier in scrape)
                report.skipped += 1
                continue
            seen.add(h)
            rel = f.relative_to(raw_root)
            domain = rel.parts[1] if rel.parts[0] == "web" and len(rel.parts) > 1 else rel.parts[0]
            tokens = {_slug(t) for part in rel.parts[:-1] for t in re.split(r"[^A-Za-z0-9]+", part)}
            tags = sorted(t for t in tokens if t in _STYLE_HINTS)
            report.artifacts.append(RawArtifact(
                kind="lut_file", source_pack_id=self.source_pack_id, family=self.family,
                declared_domain="srgb", license=f"scraped:{domain} (personal-use, non-redistribution)",
                source_url=f"web:{domain}", file_hash=h, download_timestamp=ts,
                lut_id=f"web_{_slug(domain)}_{_slug(f.stem)}_{h[:8]}",
                file_path=str(f), derivation_method=method, author_uploader_pack_id=_slug(domain),
                gold_tags=tags, style_bundle=_slug(domain),
                extra={"scraped_domain": domain, "rel_path": str(rel)},
            ))
            report.acquired += 1

        report.status = "ok" if report.acquired else "partial"
        report.note = (f"{report.acquired} new, {report.skipped} dup, {skipped_formats} unparsable "
                       f"(.3dl/.look), {len(seen)} total distinct hashes")
        return report
