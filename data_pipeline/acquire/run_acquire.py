"""Stage 2 orchestrator: verify + fetch each enabled source, write raw provenance + report.

Autonomous and bounded. Maps ``source_pack_id`` -> connector(s), fetches under the artifact
root's ``luts/raw/``, writes raw rows to ``data/raw_registry/``, and emits
``acquisition_report.json``. Per-source failure is graceful (recorded, never fatal).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ..errors import RequiresManualOptIn
from ..paths import artifact_paths
from ..registry import RegistryStore
from .base import AcquireLimits, AcquireReport
from .fivek_hf import FiveKHFConnector
from .fivek_kaggle import FiveKKaggleConnector
from .freshluts import FreshLutsConnector
from .haldclut_gmic import GmicHaldConnector
from .haldclut_rt import RawTherapeeHaldConnector
from .on1_local import ON1Connector
from .ppr10k_hf import PPR10KHFConnector
from .procedural_gen import ProceduralConnector
from .public_packs import PublicPacksConnector
from .scraped_web import ScrapedWebConnector

# source_pack_id -> ordered list of connector factories (primary first).
CONNECTORS = {
    "procedural_fillers_v1": [ProceduralConnector],
    "gmic_rawtherapee_haldclut": [RawTherapeeHaldConnector, GmicHaldConnector],
    "ppr10k_expert_abc": [PPR10KHFConnector],
    "fivek_expert_abcde": [FiveKKaggleConnector, FiveKHFConnector],  # Kaggle .jpg primary, HF fallback
    "freshluts_public": [FreshLutsConnector],
    "on1_lut_packs": [ON1Connector],
    "scraped_web": [ScrapedWebConnector],
    "public_lut_packs_misc": [PublicPacksConnector],
}

# Packs whose extra connectors are FALLBACKS (only run if the primary acquires nothing), NOT additive
# mirrors. Without this, both FiveK connectors run and DOUBLE-INGEST the same dataset. Additive packs
# (e.g. gmic_rawtherapee_haldclut, which sets secondary_max_items) are intentionally NOT listed here.
FALLBACK_SOURCES = {"fivek_expert_abcde"}

# Bounded, ToS-respecting defaults. FreshLUTs uncapped per user permission.
DEFAULT_ACQUIRE = {
    "procedural_fillers_v1": {"enabled": True, "max_items": None},
    "gmic_rawtherapee_haldclut": {"enabled": True, "max_items": 300, "secondary_max_items": 60},
    "ppr10k_expert_abc": {"enabled": True, "max_items": 150},
    "fivek_expert_abcde": {"enabled": True, "max_items": 150},
    "freshluts_public": {"enabled": True, "max_items": None},
    "on1_lut_packs": {"enabled": True, "max_items": None},
    "scraped_web": {"enabled": True, "max_items": None},
    "public_lut_packs_misc": {"enabled": False, "max_items": 0},
}


def _load_acquire_config(config_path: str | None) -> dict:
    if not config_path:
        return {k: dict(v) for k, v in DEFAULT_ACQUIRE.items()}
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    acq = data.get("acquisition") or {}
    merged = {k: dict(v) for k, v in DEFAULT_ACQUIRE.items()}
    for k, v in acq.items():
        merged.setdefault(k, {}).update(v or {})
    return merged


def run_acquire(config_path: str | None = None, out_root: str | None = None,
                only: list[str] | None = None, full: bool = False,
                rate_limit_s: float = 0.0) -> dict:
    paths = artifact_paths(out_root).ensure()
    acq = _load_acquire_config(config_path)
    store = RegistryStore(paths.raw_registry)

    reports: list[AcquireReport] = []

    def emit(r: AcquireReport) -> None:
        reports.append(r)
        print(f"[acquire] {r.source_pack_id}: status={r.status} acquired={r.acquired} "
              f"failed={r.failed} skipped={r.skipped} :: {r.note}", flush=True)

    for pack_id, factories in CONNECTORS.items():
        if only and pack_id not in only:
            continue
        cfg = acq.get(pack_id, {})
        if not cfg.get("enabled", False):
            emit(AcquireReport(pack_id, status="skipped", note="disabled in config"))
            continue

        max_items = None if full else cfg.get("max_items")
        for i, factory in enumerate(factories):
            connector = factory()
            limits = AcquireLimits(
                max_items=(None if full else (cfg.get("secondary_max_items") if i else max_items)),
                rate_limit_s=rate_limit_s,
            )
            print(f"[acquire] starting {pack_id} ({connector.__class__.__name__}) "
                  f"max_items={limits.max_items}", flush=True)
            try:
                ok, note = connector.verify()
                if not ok:
                    emit(AcquireReport(pack_id, status="skipped", note=f"verify: {note}"))
                    continue
                report = connector.acquire(paths.luts_raw, limits)
            except RequiresManualOptIn as e:
                emit(AcquireReport(pack_id, status="skipped", note=str(e)))
                continue
            except Exception as e:  # noqa: BLE001
                emit(AcquireReport(pack_id, status="failed", note=f"error: {e}"))
                continue
            for art in report.artifacts:
                store.add(art.to_registry_row())
            emit(report)
            # Fallback packs: once the primary connector acquires anything, do NOT run the remaining
            # (fallback) connectors — they are mirrors of the same dataset, not additive sources.
            if pack_id in FALLBACK_SOURCES and report.acquired > 0:
                break

    summary = {
        "artifact_root": str(paths.root),
        "raw_registry": str(store.path),
        "sources": [r.summary() for r in reports],
        "total_acquired": sum(r.acquired for r in reports),
    }
    report_path = paths.raw_registry / "acquisition_report.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Stage 2: autonomous source acquisition.")
    ap.add_argument("--config", default=None, help="pipeline config yaml (acquisition section)")
    ap.add_argument("--out", default=None, help="artifact root (default: cwd / $SLM_ARTIFACT_ROOT)")
    ap.add_argument("--sources", default=None, help="comma-separated source_pack_ids to run")
    ap.add_argument("--full", action="store_true", help="lift sampled caps (large downloads)")
    ap.add_argument("--rate-limit", type=float, default=0.0, help="min seconds between requests")
    args = ap.parse_args(argv)
    only = args.sources.split(",") if args.sources else None
    summary = run_acquire(args.config, args.out, only=only, full=args.full,
                          rate_limit_s=args.rate_limit)
    print(json.dumps(summary, indent=2))
    for s in summary["sources"]:
        print(f"  [{s['status']:>7}] {s['source_pack_id']:<28} "
              f"acquired={s['acquired']} failed={s['failed']} skipped={s['skipped']} :: {s['note']}")
    print(f"total acquired: {summary['total_acquired']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
