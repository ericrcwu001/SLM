"""G'MIC color-presets HaldCLUT connector (best-effort, bounded).

Scrapes ``gmic.eu/color_presets/`` for HaldCLUT ``.png`` links and downloads a bounded
sample under ``luts/raw/haldclut/gmic/``. Best-effort: if the page scheme changes the
connector reports ``partial``/``failed`` gracefully (RawTherapee already provides the bulk
of real film LUTs).
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin

from . import downloaders as dl
from .base import AcquireLimits, AcquireReport, RawArtifact, utcnow_iso

_INDEX = "https://gmic.eu/color_presets/"
_LICENSE = "G'MIC color presets (published for download; gmic.eu)"


class GmicHaldConnector:
    source_pack_id = "gmic_rawtherapee_haldclut"
    family = "gmic_rawtherapee"

    def __init__(self, session=None, index_url: str = _INDEX):
        self.session = session
        self.index_url = index_url

    def verify(self) -> tuple[bool, str]:
        return dl.http_head_ok(self.index_url, session=self.session)

    def _png_links(self, html: str) -> list[str]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        hrefs: set[str] = set()
        for tag, attr in (("a", "href"), ("img", "src")):
            for el in soup.find_all(tag):
                val = el.get(attr)
                if val and val.lower().endswith(".png"):
                    hrefs.add(urljoin(self.index_url, val))
        return sorted(hrefs)

    def acquire(self, raw_root, limits: AcquireLimits) -> AcquireReport:
        report = AcquireReport(source_pack_id=self.source_pack_id, note="gmic")
        dest = Path(raw_root) / "haldclut" / "gmic"
        sess = self.session or dl.polite_session()
        try:
            html = sess.get(self.index_url, timeout=60).text
        except Exception as e:  # noqa: BLE001
            report.status = "failed"
            report.note = f"index fetch failed: {e}"
            return report

        links = self._png_links(html)
        if not links:
            report.status = "partial"
            report.note = "no .png links discovered on color_presets index"
            return report
        if limits.max_items is not None:
            links = links[: limits.max_items]

        ts = utcnow_iso()
        for url in links:
            report.attempted += 1
            name = url.rsplit("/", 1)[-1]
            try:
                png = dl.http_download(url, dest / name, session=sess)
                report.artifacts.append(RawArtifact(
                    kind="lut_file", source_pack_id=self.source_pack_id, family=self.family,
                    declared_domain="srgb", license=_LICENSE, source_url=url,
                    file_hash=dl.sha256_file(png), download_timestamp=ts,
                    lut_id=f"gmic_{Path(name).stem}", file_path=str(png),
                    derivation_method="haldclut", author_uploader_pack_id="gmic",
                ))
                report.acquired += 1
            except Exception:  # noqa: BLE001
                report.failed += 1
        report.status = "ok" if report.acquired else "partial"
        return report
