"""Procedural-filler connector: wraps the local generator as a Stage-2 source."""

from __future__ import annotations

from pathlib import Path

from ..sources import procedural as proc
from .base import AcquireLimits, AcquireReport, RawArtifact, utcnow_iso


class ProceduralConnector:
    source_pack_id = "procedural_fillers_v1"
    family = "controlled_procedural"

    def __init__(self, magnitudes: tuple[float, ...] = (0.6, 1.0, 1.4)):
        self.magnitudes = magnitudes

    def verify(self) -> tuple[bool, str]:
        return True, "local generator"

    def acquire(self, raw_root, limits: AcquireLimits) -> AcquireReport:
        report = AcquireReport(source_pack_id=self.source_pack_id, note="generated locally")
        out_dir = Path(raw_root) / "procedural"
        gens = proc.generate(out_dir, magnitudes=self.magnitudes)
        if limits.max_items is not None:
            gens = gens[: limits.max_items]
        ts = utcnow_iso()
        for g in gens:
            report.attempted += 1
            report.artifacts.append(RawArtifact(
                kind="lut_file", source_pack_id=self.source_pack_id, family=self.family,
                declared_domain="srgb", license="generated (n/a)", source_url="generated",
                file_hash=g.file_hash, download_timestamp=ts, lut_id=g.lut_id,
                file_path=str(g.path), derivation_method="generated",
                author_uploader_pack_id="procedural",
                gold_tags=list(g.gold_tags), style_bundle=g.style, attribute=g.attribute,
                usage_prior_bucket=g.usage_prior_bucket, procedural_filler=True,
            ))
            report.acquired += 1
        report.status = "ok"
        return report
