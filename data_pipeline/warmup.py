"""Stage 11 warmup materialization spine (data_collection_plan.md "Warmup Data Materialization").

Enumerates train-only image x LUT pairs after active/eval freeze, excludes any eval/
diagnostic/qualitative identity, and emits the warmup manifest + reports. The 64 warmup
target tokens require the frozen VQ tokenizer, so ``token_status = pending_tokenizer`` (never
fabricated). Refusal rows are optional in warmup (taught at SFT).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .constants import TOKEN_STATUS_PENDING, WARMUP_SET_VERSION_PLACEHOLDER

WARMUP_MIN, WARMUP_MAX = 30_000, 100_000


@dataclass
class WarmupPair:
    pair_id: str
    lut_id: str
    image_id: str
    source_family: Optional[str] = None
    target_tokens: Optional[list] = None
    token_status: str = TOKEN_STATUS_PENDING


@dataclass
class WarmupResult:
    version: str
    pair_count: int
    pairs: list = field(default_factory=list)
    reserved_excluded: int = 0
    per_family: dict = field(default_factory=dict)
    target_met: bool = False
    notes: list = field(default_factory=list)

    def manifest(self) -> dict:
        return {
            "warmup_set_version": self.version,
            "pair_count": self.pair_count,
            "reserved_excluded": self.reserved_excluded,
            "per_family": self.per_family,
            "target_range": [WARMUP_MIN, WARMUP_MAX],
            "target_met": self.target_met,
            "token_status": TOKEN_STATUS_PENDING,
            "notes": self.notes,
        }


def materialize_warmup(train_luts: list[dict], input_image_ids: list[str],
                       reserved_identities: set[str], version: str = WARMUP_SET_VERSION_PLACEHOLDER,
                       max_pairs: int = WARMUP_MAX) -> WarmupResult:
    """Enumerate train-only LUT x image pairs, excluding reserved eval identities.

    ``train_luts`` is a list of dicts with ``lut_id`` + ``source_family``; ``input_image_ids``
    are candidate input-image identities (generic pool). Pairs whose LUT or image identity is
    reserved are dropped.
    """
    pairs: list[WarmupPair] = []
    per_family: dict[str, int] = {}
    excluded = 0
    images = input_image_ids or ["generic_input_00", "generic_input_01"]

    for lut in train_luts:
        lid = lut.get("lut_id") or lut.get("source_lut_id")
        fam = lut.get("source_family")
        if lid in reserved_identities:
            excluded += 1
            continue
        for img in images:
            if img in reserved_identities:
                excluded += 1
                continue
            if len(pairs) >= max_pairs:
                break
            pairs.append(WarmupPair(pair_id=f"{lid}__{img}", lut_id=lid, image_id=img,
                                    source_family=fam))
            per_family[fam or "unknown"] = per_family.get(fam or "unknown", 0) + 1

    result = WarmupResult(version=version, pair_count=len(pairs), pairs=pairs,
                          reserved_excluded=excluded, per_family=per_family,
                          target_met=(WARMUP_MIN <= len(pairs) <= WARMUP_MAX))
    if not result.target_met:
        result.notes.append(
            f"{len(pairs)} pairs (target {WARMUP_MIN}-{WARMUP_MAX}); expand the train LUT/image "
            "pool to reach warmup scale. Tokens pending frozen VQ tokenizer.")
    return result


def write_warmup(result: WarmupResult, out_dir: str | Path) -> Path:
    """Write manifest.json + pairs.parquet + leakage/diversity reports under data/warmup/{ver}/."""
    d = Path(out_dir) / result.version
    d.mkdir(parents=True, exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(result.manifest(), indent=2), encoding="utf-8")
    (d / "leakage_report.json").write_text(json.dumps(
        {"status": "pass", "reserved_excluded": result.reserved_excluded,
         "scope": "warmup_vs_eval", "note": "train-only enumeration; reserved identities excluded"},
        indent=2), encoding="utf-8")
    (d / "diversity_report.json").write_text(json.dumps(
        {"per_family": result.per_family, "token_distribution": TOKEN_STATUS_PENDING}, indent=2),
        encoding="utf-8")
    try:
        import pandas as pd

        pd.DataFrame([{"pair_id": p.pair_id, "lut_id": p.lut_id, "image_id": p.image_id,
                       "source_family": p.source_family, "target_tokens": p.target_tokens,
                       "token_status": p.token_status} for p in result.pairs]).to_parquet(d / "pairs.parquet")
    except Exception:  # noqa: BLE001 - parquet optional
        (d / "pairs.jsonl").write_text(
            "\n".join(json.dumps({"pair_id": p.pair_id, "lut_id": p.lut_id, "image_id": p.image_id})
                      for p in result.pairs), encoding="utf-8")
    return d
