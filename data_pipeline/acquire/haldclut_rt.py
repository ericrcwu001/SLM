"""RawTherapee Film Simulation HaldCLUT connector (primary real source).

Direct archive: ``rawtherapee.com/shared/HaldCLUT.zip`` (~402 MB, ~292 sRGB-8bit HaldCLUT
PNGs). Downloaded once (resumable), extracted, each PNG -> a ``lut_file`` RawArtifact under
``luts/raw/haldclut/rawtherapee/``.
"""

from __future__ import annotations

from pathlib import Path

from . import downloaders as dl
from .base import AcquireLimits, AcquireReport, RawArtifact, utcnow_iso

_URLS = [
    "https://rawtherapee.com/shared/HaldCLUT.zip",
    "http://rawtherapee.com/shared/HaldCLUT.zip",
]
_LICENSE = "RawTherapee Film Simulation Collection (published for use; see rawpedia Film_Simulation)"


class RawTherapeeHaldConnector:
    source_pack_id = "gmic_rawtherapee_haldclut"
    family = "gmic_rawtherapee"

    def __init__(self, session=None, urls=None):
        self.session = session
        self.urls = urls or _URLS

    def verify(self) -> tuple[bool, str]:
        for url in self.urls:
            ok, note = dl.http_head_ok(url, session=self.session)
            if ok:
                return True, f"{url} {note}"
        return False, "no RawTherapee HaldCLUT.zip URL reachable"

    def acquire(self, raw_root, limits: AcquireLimits) -> AcquireReport:
        raw_root = Path(raw_root)
        dest_dir = raw_root / "haldclut" / "rawtherapee"
        dl_dir = raw_root / "haldclut" / "_download"
        report = AcquireReport(source_pack_id=self.source_pack_id)

        zip_path = None
        for url in self.urls:
            try:
                zip_path = dl.http_download(url, dl_dir / "HaldCLUT.zip", session=self.session)
                report.note = f"downloaded {url}"
                break
            except Exception as e:  # noqa: BLE001
                report.note = f"download failed: {e}"
        if zip_path is None:
            report.status = "failed"
            return report

        try:
            pngs = dl.extract_zip(zip_path, dest_dir, suffixes=[".png"], max_items=limits.max_items)
        except Exception as e:  # noqa: BLE001
            report.status = "failed"
            report.note = f"extract failed: {e}"
            return report

        ts = utcnow_iso()
        for png in pngs:
            report.attempted += 1
            try:
                art = RawArtifact(
                    kind="lut_file", source_pack_id=self.source_pack_id, family=self.family,
                    declared_domain="srgb", license=_LICENSE, source_url=self.urls[0],
                    file_hash=dl.sha256_file(png), download_timestamp=ts,
                    lut_id=f"rt_{png.stem}", file_path=str(png), derivation_method="haldclut",
                    author_uploader_pack_id="rawtherapee_film_simulation",
                )
                report.artifacts.append(art)
                report.acquired += 1
            except Exception as e:  # noqa: BLE001
                report.failed += 1
                report.note = f"artifact error: {e}"
        report.status = "ok" if report.acquired else "failed"
        return report
