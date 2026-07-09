"""Smaller public LUT packs — gated until concrete pack URLs are added to the inventory."""

from __future__ import annotations

from .base import AcquireLimits, AcquireReport


class PublicPacksConnector:
    source_pack_id = "public_lut_packs_misc"
    family = "smaller_public_packs"

    def verify(self) -> tuple[bool, str]:
        return False, "no concrete pack URLs configured (manual opt-in)"

    def acquire(self, raw_root, limits: AcquireLimits) -> AcquireReport:
        return AcquireReport(source_pack_id=self.source_pack_id, status="skipped",
                             note="add concrete pack URLs to source_inventory before enabling")
