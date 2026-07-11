"""Master orchestrator: Stages 2 -> 3 -> 4 -> 5 -> 6 -> 9 -> 11.

Each stage runs only when its inputs exist; gated steps (token materialization, teacher
instructions) are recorded ``pending`` rather than fabricated. Emits per-stage manifests +
a run summary. Consumes acquired raw assets (``slm_acquire``) and the procedural generator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

from .acquire.run_acquire import run_acquire
from .active_dataset import AcceptanceChecker, assemble_active
from .behavior_vector import measure_behavior, native_lut_smoothness
from .canonicalize import canonicalize_lut
from .constants import (
    ACTIVE_SET_VERSION_PLACEHOLDER,
    EVAL_SET_VERSION_PLACEHOLDER,
    QUALITY_FILTER_VERSION,
    WARMUP_SET_VERSION_PLACEHOLDER,
)
from .eval_sets import EvalCandidate, build_eval_sets
from .instruction_gen import TeacherClient
from .paths import artifact_paths
from .registry import RegistryStore, validate_row
from .representability import assess_direct_lut, assess_pair_fit
from .selection import SelectionCandidate, select_active
from .sources.derive import cube_bytes_to_lut, fit_global_lut, haldclut_png_to_lut, parse_xmp
from .splits import SplitCandidate, build_split_manifest
from .warmup import materialize_warmup, write_warmup

_TINT_TAGS = {"warmer", "cooler", "tint_magenta", "tint_green", "sepia", "teal-orange", "cinematic"}
_MAX_IMG_SIDE = 512  # cap image side for pair-fit (enough pixels for per-cell support)
# A LUT whose neutrals move off-axis in a *structured* way is a deliberate colour-balance edit, so
# its neutral-axis drift is intended, not a safety failure. Two intended-cast signatures:
#  * a uniform cast -> mean temperature/tint shift clears _TINT_MEASURE_FLOOR (lowered 1.5 -> 1.0 to
#    catch subtler-but-still-deliberate white-balance edits), OR
#  * a split-tone (e.g. warm shadows / cool highlights) whose signed means CANCEL to ~0 but whose
#    shadow+highlight a/b magnitude (split_tone_strength) clears _SPLIT_TONE_FLOOR. The old check
#    only looked at the (cancelling) means, so split-tones were falsely rejected on neutral drift.
_TINT_MEASURE_FLOOR = 1.0
_SPLIT_TONE_FLOOR = 2.0

# Behavior-only sources: their bulk scraped / pack auto-tags do NOT reliably cover the LUT's
# measured behavior (they fail the reverse tag<->behavior coverage check, acceptance criteria
# 7/8). We treat them like the pair-fit derived families, which carry no authored tags: the
# instruction comes from the teacher describing the *measured* edit, not from partial tags. We
# only drop the tags from the SFT/selection surface (gold_tags + tag embedding); the Stage-5
# safety gate still sees the original tags for tint detection, so attrition is unaffected.
BEHAVIOR_ONLY_FAMILIES = {"scraped_web", "smaller_public_packs"}


def _effective_tags(row: dict) -> list:
    """Structured tags for the SFT contract, blanked for behavior-only sources."""
    if row.get("source_family") in BEHAVIOR_ONLY_FAMILIES:
        return []
    return row.get("structured_tags", []) or []


def _load_config(path: Optional[str]) -> dict:
    default = Path(__file__).parent / "configs" / "pipeline_default.yaml"
    p = Path(path) if path else default
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _load_image(path: str) -> Optional[np.ndarray]:
    try:
        from PIL import Image

        img = Image.open(path).convert("RGB")
        img.thumbnail((_MAX_IMG_SIDE, _MAX_IMG_SIDE))
        return np.asarray(img, dtype=np.float64) / 255.0
    except Exception:  # noqa: BLE001
        return None


def _derive_lut(row: dict) -> Optional[np.ndarray]:
    """Stage 4 derivation: raw asset -> absolute LUT tensor (or None to skip)."""
    method = row.get("derivation_method")
    # Return the NATIVE-size LUT (target_size=None); canonicalize does the single 17^3 resample
    # (identical canonical result) and run_pipeline measures resample-aware smoothness on native.
    try:
        if method in ("generated", "cube"):
            return cube_bytes_to_lut(Path(row["derivation_path"]).read_bytes(), target_size=None)
        if method == "haldclut":
            from PIL import Image

            arr = np.asarray(Image.open(row["derivation_path"]).convert("RGB"))
            return haldclut_png_to_lut(arr, target_size=None)
        if method == "pair_fit":
            # XMP hard-reject: a pair edited with LOCAL tools (masks/brushes/gradients) cannot be
            # represented by a GLOBAL LUT, so it must not be pair-fit into one (data_collection_plan
            # "XMP hard-reject fields"). Record the parse for audit; reject on CONFIRMED local edits
            # (an unparseable XMP falls through to the downstream spatial-structure gate).
            xmp_p = row.get("raw_edit_metadata_path")
            if xmp_p and Path(xmp_p).is_file():
                xr = parse_xmp(Path(xmp_p).read_text(encoding="utf-8", errors="ignore"))
                row["xmp_parse_status"] = xr.parse_status
                row["xmp_local_tool_count"] = xr.local_tool_count
                row["xmp_rejected_fields"] = xr.rejected_fields
                if xr.local_tool_count > 0:
                    return None
            src = _load_image(row.get("source_image_path"))
            tgt = _load_image(row.get("target_image_path"))
            if src is None or tgt is None or src.shape != tgt.shape:
                return None
            return fit_global_lut(src, tgt).lut_abs
    except Exception:  # noqa: BLE001
        return None
    return None


def _tinted(tags: list, behavior: Optional[dict] = None) -> bool:
    """Whether this edit is a deliberate global tint (suppresses the neutral-drift safety gate).

    True if it carries a tint style tag, or its *measured* temperature/tint shift clears the
    direction-magnitude floor. Pair-fit rows have no authored tags, so the measured signal is
    what keeps a legitimate white-balance edit from being hard-rejected on neutral drift.
    """
    if set(tags or []) & _TINT_TAGS:
        return True
    b = behavior or {}
    return (abs(b.get("temperature_delta_b", 0.0)) >= _TINT_MEASURE_FLOOR
            or abs(b.get("tint_delta_a", 0.0)) >= _TINT_MEASURE_FLOOR
            or b.get("split_tone_strength", 0.0) >= _SPLIT_TONE_FLOOR)


def _persist_representability_metrics(row: dict, rep) -> None:
    """Write the measured fit / spatial / support / quality metrics onto the registry row.

    These are the numbers the Stage-5 gate decided on; persisting them makes the gate
    auditable and satisfies the Stage-9 "provenance + measured behavior" requirement (they
    were computed but previously discarded, leaving the registry fields null).
    """
    fd = rep.fit_deltaE00 or {}
    row["fit_deltaE00_mean"] = fd.get("mean")
    row["fit_deltaE00_median"] = fd.get("median")
    row["fit_deltaE00_p95"] = fd.get("p95")
    row["fit_deltaE00_p99"] = fd.get("p99")
    row["fit_deltaE00_max"] = fd.get("max")
    row["fit_train_deltaE00"] = rep.fit_train_deltaE00
    row["fit_validation_deltaE00"] = rep.fit_validation_deltaE00
    sp = rep.spatial or {}
    row["residual_xy_r2"] = sp.get("residual_xy_r2")
    row["residual_edge_corr"] = sp.get("residual_edge_corr")
    row["residual_tile_p95"] = sp.get("residual_tile_p95")
    row["residual_tile_max"] = sp.get("residual_tile_max")
    row["largest_high_residual_component_pct"] = sp.get("largest_high_residual_component_pct")
    su = rep.support or {}
    row["supported_cell_rate"] = su.get("supported_cell_rate")
    row["input_pixel_supported_rate"] = su.get("input_pixel_supported_rate")
    row["quality_scores"] = rep.quality_scores
    row["quality_filter_version"] = rep.quality_filter_version


def run_pipeline(config_path: Optional[str] = None, out_root: Optional[str] = None,
                 acquire: bool = True, only_sources: Optional[list] = None,
                 full: bool = False) -> dict:
    cfg = _load_config(config_path)
    paths = artifact_paths(out_root).ensure()
    summary: dict = {"artifact_root": str(paths.root), "stages": {}}

    # --- Stage 2: acquire ---
    if acquire:
        acq = run_acquire(config_path, out_root, only=only_sources, full=full)
        summary["stages"]["2_acquire"] = {"total_acquired": acq["total_acquired"],
                                          "sources": acq["sources"]}
    store = RegistryStore(paths.raw_registry)
    rows = [r.to_dict() for r in store.load()]
    summary["stages"].setdefault("2_acquire", {})["raw_rows"] = len(rows)

    # --- Stage 4 + 5: derive/canonicalize + representability/quality/behavior ---
    attrition = {"candidates": len(rows), "derived": 0, "canonicalized": 0, "gold": 0,
                 "diagnostic_only": 0, "rejected": 0}
    enriched: list[dict] = []
    _total = len(rows)
    print(f"[pipeline] STAGE 4+5 derive/canonicalize/gates: {_total} rows", flush=True)
    for _i, row in enumerate(rows, 1):
        if _i % 25 == 0 or _i == _total:
            print(f"[pipeline] derive {_i}/{_total} gold={attrition['gold']} "
                  f"diag={attrition['diagnostic_only']} rej={attrition['rejected']}", flush=True)
        # resumability: a row already enriched by a prior pipeline pass (has a canonical
        # residual hash + its cached .npy on disk) is reused as-is, skipping the expensive
        # derive/canonicalize/pair-fit/gates. Lets a later run add only new sources. A
        # quality_filter_version bump invalidates the cached tier (the gate changed), forcing
        # re-derivation; np.save then overwrites the stale .npy in place.
        _rk = row.get("lut_id") or row.get("source_pair_id") or row.get("file_hash")
        _cache_current = row.get("quality_filter_version") == QUALITY_FILTER_VERSION
        if (_cache_current and row.get("canonical_residual_lut_hash") and _rk
                and (paths.canonical_residual / f"{_rk}.npy").exists()):
            tier = row.get("representability_tier") or "rejected"
            attrition["derived"] += 1
            attrition["canonicalized"] += 1
            attrition[tier if tier in ("gold", "diagnostic_only", "rejected") else "rejected"] += 1
            row["_residual_key"] = _rk
            row["residual_key"] = _rk  # persisted (survives _row_obj underscore strip)
            enriched.append(row)
            continue
        lut = _derive_lut(row)
        if lut is None:
            attrition["rejected"] += 1
            row["representability_tier"] = "rejected"
            row["reject_reason_codes"] = ["derivation_failed"]
            enriched.append(row)
            continue
        attrition["derived"] += 1
        can = canonicalize_lut(lut, row.get("raw_color_space", "srgb"))
        if can.rejected:
            attrition["rejected"] += 1
            row["representability_tier"] = "rejected"
            row["reject_reason_codes"] = [can.reject_reason]
            enriched.append(row)
            continue
        attrition["canonicalized"] += 1
        row["canonical_domain_id"] = None  # set below via constant
        from .constants import CANONICAL_DOMAIN_ID
        row["canonical_domain_id"] = CANONICAL_DOMAIN_ID
        row["canonical_absolute_lut_hash"] = can.canonical_absolute_lut_hash
        row["canonical_residual_lut_hash"] = can.canonical_residual_lut_hash
        row["normalization_warnings"] = can.normalization_warnings

        behavior = measure_behavior(can.absolute)
        row["measured_behavior"] = behavior
        row["behavior_vector_version"] = behavior["behavior_vector_version"]
        tinted = _tinted(row.get("structured_tags"), behavior)

        # resample-aware smoothness: measure on the LUT's NATIVE grid (before our 17^3 downsampling)
        # so the gate reflects the LUT's own bumpiness, not trilinear-resampling aliasing. Capped at
        # 33 nodes so huge 8-bit HaldCLUT grids don't blow up on quantization noise.
        nat_smooth = native_lut_smoothness(lut)
        row["measured_behavior"]["smoothness_native"] = nat_smooth

        if row.get("derivation_method") == "pair_fit":
            src = _load_image(row.get("source_image_path"))
            tgt = _load_image(row.get("target_image_path"))
            rep = assess_pair_fit(can.absolute, src, tgt, tinted=tinted, smoothness_override=nat_smooth,
                                  pre_clamp=can.pre_clamp_absolute)
        else:
            rep = assess_direct_lut(can.absolute, tinted=tinted, smoothness_override=nat_smooth,
                                    pre_clamp=can.pre_clamp_absolute)
        row["representability_tier"] = rep.tier
        row["representability_status"] = rep.status
        row["derived_lut_quality"] = {
            "representability_tier": rep.tier,
            "fit_deltaE00_mean": rep.fit_deltaE00.get("mean", 0.0),
            "fit_deltaE00_p95": rep.fit_deltaE00.get("p95", 0.0),
            "supported_cell_rate": rep.support.get("supported_cell_rate", 1.0),
        }
        row["reject_reason_codes"] = rep.reasons
        _persist_representability_metrics(row, rep)
        # residual saved for split/leakage/embeddings
        key = row.get("lut_id") or row.get("source_pair_id") or row.get("file_hash")
        np.save(paths.canonical_residual / f"{key}.npy", can.residual)
        row["_residual_key"] = key
        row["residual_key"] = key  # persisted (survives _row_obj underscore strip)
        attrition[rep.tier] += 1
        enriched.append(row)

    (paths.raw_registry / "derivation_attrition.json").write_text(
        json.dumps(attrition, indent=2), encoding="utf-8")
    # Enforce the registry contract (validate_row was previously never called) on the rows we ADMIT.
    # Rejected rows are provenance-only (incomplete by design) and skipped; an invalid ACCEPTED row is
    # a real defect and fails loud rather than being silently persisted.
    for _r in enriched:
        if _r.get("representability_tier") == "rejected":
            continue
        _errs = validate_row(_row_obj(_r))
        if _errs:
            raise ValueError(f"invalid accepted provenance row {_r.get('file_hash')}: {_errs}")
    store.write_all([_row_obj(r) for r in enriched])
    summary["stages"]["4_5_derive_filter"] = attrition

    accepted = [r for r in enriched if r.get("representability_tier") in ("gold", "diagnostic_only")]
    print(f"[pipeline] STAGE 6 splits+leakage: {len(accepted)} accepted "
          f"(gold={attrition['gold']} diag={attrition['diagnostic_only']} rej={attrition['rejected']})",
          flush=True)

    # --- Stage 6: splits + leakage ---
    split_cands = []
    for r in accepted:
        res_key = r.get("_residual_key")
        residual = None
        rp = paths.canonical_residual / f"{res_key}.npy"
        if rp.exists():
            residual = np.load(rp).reshape(-1)
        split_cands.append(SplitCandidate(
            id=r["file_hash"] or res_key,
            base_key=r.get("group_id") or r.get("source_photo_id") or r.get("normalized_lut_hash") or res_key,
            procedural=bool(r.get("procedural_filler")),
            lut_hash=r.get("canonical_residual_lut_hash"),
            residual_vec=residual,
        ))
    split_manifest = build_split_manifest(split_cands, seed=cfg.get("seeds", {}).get("split", 1234),
                                          ratios=cfg.get("splits", {}).get("ratios"))
    (paths.splits / "split_manifest.json").write_text(json.dumps({
        "split_id": split_manifest.split_id,
        "leakage_policy_version": split_manifest.leakage_policy_version,
        "leakage_report_hash": split_manifest.leakage_report_hash,
        "leakage_status": split_manifest.leakage_status,
        "unit_count": split_manifest.unit_count,
        "assignments": split_manifest.assignments,
    }, indent=2), encoding="utf-8")
    (paths.splits / "leakage_report.json").write_text(json.dumps({
        "status": split_manifest.leakage_status,
        "leakage_report_hash": split_manifest.leakage_report_hash,
        "leakage_policy_version": split_manifest.leakage_policy_version,
    }, indent=2), encoding="utf-8")
    id_to_split = {k: v["split"] for k, v in split_manifest.assignments.items()}
    id_to_unit = {k: v["split_unit_id"] for k, v in split_manifest.assignments.items()}
    summary["stages"]["6_splits_leakage"] = {
        "split_id": split_manifest.split_id, "leakage_status": split_manifest.leakage_status,
        "unit_count": split_manifest.unit_count}

    # --- Stage 9: selection + active + eval sets ---
    print("[pipeline] STAGE 9 selection+active+eval", flush=True)
    sel_cfg = cfg.get("selection", {})
    from .embeddings import behavior_embedding, tag_embedding
    train_accepted = [r for r in accepted if id_to_split.get(r["file_hash"] or r.get("_residual_key")) == "train"]
    sel_cands = []
    for r in train_accepted:
        emb = np.concatenate([behavior_embedding(r.get("measured_behavior", {})),
                              tag_embedding(_effective_tags(r))])
        sel_cands.append(SelectionCandidate(
            id=r["file_hash"] or r.get("_residual_key"),
            family=r.get("source_family") or "unknown",
            usage_prior_bucket=r.get("usage_prior_bucket") or "common_head",
            embedding=emb, procedural=bool(r.get("procedural_filler")),
        ))
    sel = select_active(sel_cands, target_size=sel_cfg.get("active_target_size", 12000),
                        source_caps=sel_cfg.get("source_caps"),
                        seed=cfg.get("seeds", {}).get("selection", 1234))
    sel_ids = set(sel.selected_ids)

    by_id = {(r["file_hash"] or r.get("_residual_key")): r for r in accepted}
    selected_rows = []
    for sid in sel.selected_ids:
        r = by_id[sid]
        selected_rows.append({
            "id": sid, "source_family": r.get("source_family"), "source_lut_id": r.get("lut_id"),
            "gold_tags": _effective_tags(r), "measured_behavior": r.get("measured_behavior", {}),
            "derived_lut_quality": r.get("derived_lut_quality", {}),
            "representability_tier": r.get("representability_tier"),
            "split_unit_id": id_to_unit.get(sid), "split": "train",
            "procedural_filler": bool(r.get("procedural_filler")),
            # Source image (if this LUT was derived from an image pair) so the vision-capable
            # teacher can ground its phrasing; None for LUT-only rows -> teacher runs text-only.
            "image_path": r.get("source_image_path"),
            # Source-authored instruction (MMArt-PPR10K): teacher is skipped for these rows.
            "authored_instruction": r.get("authored_instruction"),
            "authored_instruction_natural": r.get("authored_instruction_natural"),
        })
    active_rows = assemble_active(selected_rows)
    teacher = TeacherClient(cfg.get("model_clients", "configs/model_clients.yaml"))

    # Optional inline instruction generation (default OFF — makes network calls + spends).
    # The standalone runner is `python -m scripts.generate_instructions`; this hook lets an
    # overnight pipeline fill instructions in one pass when explicitly enabled in config.
    ig_counts = None
    ig_cfg = cfg.get("instruction_generation") or {}
    if ig_cfg.get("run_inline") and teacher.is_available():
        from .instruction_gen import apply_instruction_result, generate_instructions_for_rows
        ig_inputs = [{"id": r.id, "gold_tags": r.gold_tags,
                      "measured_behavior": r.measured_behavior, "image_path": r.image_path,
                      "instruction": r.instruction, "instruction_status": r.instruction_status}
                     for r in active_rows]
        ig_manifest = generate_instructions_for_rows(
            ig_inputs, teacher,
            judge_model_clients_path=cfg.get("model_clients", "configs/model_clients.yaml"),
            run_judge=ig_cfg.get("run_judge", True), limit=ig_cfg.get("limit"),
            attach_image=ig_cfg.get("attach_image", True))
        by_id = {res["id"]: res for res in ig_manifest["rows"]}
        for r in active_rows:
            res = by_id.get(r.id)
            if res:
                apply_instruction_result(r, res)
        (paths.active_sft / "instruction_gen_manifest.json").write_text(
            json.dumps({k: v for k, v in ig_manifest.items() if k != "rows"}, indent=2),
            encoding="utf-8")
        ig_counts = ig_manifest["counts"]

    acceptance = AcceptanceChecker(
        enforce_scale=sel_cfg.get("enforce_scale", False),
        waive_expert_cap=sel_cfg.get("waive_expert_cap", True),
        coverage_threshold=sel_cfg.get("coverage_threshold", 7.0),
    ).check(active_rows, leakage_status=split_manifest.leakage_status,
            model_clients_available=teacher.is_available())

    (paths.active_sft / "active_manifest.json").write_text(json.dumps({
        "active_set_version": ACTIVE_SET_VERSION_PLACEHOLDER,
        "selected": len(active_rows), "selection_report": {
            "per_family": sel.per_family, "per_bucket": sel.per_bucket,
            "target_size": sel.target_size, "effective_size": sel.effective_size,
            "target_met": sel.target_met, "notes": sel.notes},
        "acceptance": acceptance.summary(),
    }, indent=2), encoding="utf-8")
    with open(paths.active_sft / "active_rows.jsonl", "w", encoding="utf-8") as fh:
        for r in active_rows:
            fh.write(json.dumps(r.to_dict(), sort_keys=True) + "\n")

    # eval sets from reserved (eval/diagnostic/qualitative) accepted rows
    eval_cands = []
    for r in accepted:
        rid = r["file_hash"] or r.get("_residual_key")
        sp = id_to_split.get(rid, "train")
        if sp in ("eval", "diagnostic", "qualitative"):
            eval_cands.append(EvalCandidate(
                id=rid, split=sp, is_supported=True,
                representability_tier=r.get("representability_tier"),
                procedural_filler=bool(r.get("procedural_filler")),
                fit_deltaE00_mean=(r.get("derived_lut_quality", {}) or {}).get("fit_deltaE00_mean")))
    eval_manifest = build_eval_sets(eval_cands, sizes=cfg.get("eval_sets"))
    (paths.eval_sets / "eval_manifest.json").write_text(
        json.dumps(eval_manifest.summary(), indent=2), encoding="utf-8")
    summary["stages"]["9_active_eval"] = {
        "active_selected": len(active_rows), "acceptance_overall": acceptance.overall,
        "eval": eval_manifest.summary(), "instruction_generation": ig_counts}

    # --- Stage 11: warmup ---
    print("[pipeline] STAGE 11 warmup", flush=True)
    reserved = {rid for rid, sp in id_to_split.items() if sp in ("eval", "diagnostic", "qualitative")}
    train_luts = [{"lut_id": r["file_hash"] or r.get("_residual_key"),
                   "source_family": r.get("source_family")}
                  for r in train_accepted]
    warm = materialize_warmup(train_luts, input_image_ids=[], reserved_identities=reserved,
                              max_pairs=100_000)
    write_warmup(warm, paths.warmup)
    summary["stages"]["11_warmup"] = warm.manifest()

    (paths.root / "data" / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[pipeline] DONE run_summary.json written", flush=True)
    return summary


def _row_obj(d: dict):
    from .registry import ProvenanceRow

    clean = {k: v for k, v in d.items() if not k.startswith("_")}
    return ProvenanceRow.from_dict(clean)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Data-gen pipeline (Stages 2-9 + 11).")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="artifact root")
    ap.add_argument("--no-acquire", action="store_true", help="use existing raw registry")
    ap.add_argument("--sources", default=None, help="comma-separated source_pack_ids")
    ap.add_argument("--full", action="store_true")
    args = ap.parse_args(argv)
    only = args.sources.split(",") if args.sources else None
    summary = run_pipeline(args.config, args.out, acquire=not args.no_acquire,
                           only_sources=only, full=args.full)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
