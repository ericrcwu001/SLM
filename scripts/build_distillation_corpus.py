"""Build a self-distillation corpus by harvesting best-of-N winners (docs/collapse_fix/03).

For each TRAINING supported row, sample best-of-N from a harvest adapter, and if the reranker-best
sample clears an ABSOLUTE fidelity bar ``--tau`` (NOT compared to the gold codes — gold is ~0.89 and
unreachable; see the doc), rewrite the row's ``target_tokens`` to that reachable trajectory. Holdout
rows and unsupported/refuse rows are copied UNCHANGED (the model is never run on them) so the
behavioral eval stays honest. Then SFT a fresh adapter on the distilled corpus (ReST/RFT/expert
iteration) — a stable MLE objective that moves the model's own free-running distribution.

The oracle@N gate put P6 coverage at ~0.30, so ``--tau`` defaults to 0.30 (a 0.5 default would replace
almost nothing). Iterate: harvest with the distilled adapter and repeat while the oracle climbs.

Writes ``<out-dir>/active_rows.jsonl`` (backup + atomic replace) + a manifest ``distillation`` block,
and a resumable per-row cache. ``distill_row`` is pure (no torch) and unit-tested.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

from sft.example import is_supported_materialized, load_rows
from sft.holdout import is_holdout_row


def _assistant_target(codes) -> str:
    """MUST equal scripts.materialize_target_tokens._assistant_target (kept torch-free for pure tests)."""
    return "<lut_bos> " + " ".join(f"<lut_{int(c):03d}>" for c in codes) + " <lut_eos>"


def distill_row(row: dict, best_codes, best_fid, tau: float) -> dict:
    """Pure transform. Rewrite ONLY when a valid, reachable winner clears the ABSOLUTE bar ``tau``;
    otherwise return the row unchanged (keep gold). Never compares to the row's gold fidelity."""
    if best_codes is None or len(best_codes) != 64:            # 64-guard (invariant 6)
        return row
    if (best_fid or 0.0) < tau:                                # absolute bar (invariant 2), NOT vs gold
        return row
    return {**row,
            "target_tokens": [int(c) for c in best_codes],
            "assistant_target": _assistant_target(best_codes),
            "token_status": "distilled"}


def _load_cache(path: Path) -> dict:
    cache: dict = {}
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                cache[r["id"]] = r
    return cache


def run(args) -> int:
    from sft.score_tokens import _load_config
    cfg = _load_config(args.config)   # only the locked pixel/quant knobs matter for loading

    src = Path(args.source_rows)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_rows_p = out_dir / "active_rows.jsonl"
    cache_p = out_dir / "harvest_cache.jsonl"

    rows = load_rows(str(src))
    cache = _load_cache(cache_p)
    print(f"[distill] source={src} rows={len(rows)} cached_winners={len(cache)} tau={args.tau}")

    from eval.best_of_n import best_of_n_for_row
    from sft.loader import load_eval_model
    model, processor = load_eval_model(cfg, args.resized_model, args.adapter)

    counts: Counter = Counter()
    fids: list[float] = []
    out_rows: list[dict] = []
    n_train = sum(1 for r in rows if r.get("is_supported") and not is_holdout_row(r)
                  and is_supported_materialized(r))
    done = 0
    cache_fh = open(cache_p, "a", encoding="utf-8")
    try:
        for row in rows:
            if not row.get("is_supported"):
                out_rows.append(row); counts["unsupported"] += 1; continue
            if is_holdout_row(row):                                    # sacred — never touch/generate
                out_rows.append(row); counts["holdout"] += 1; continue
            if not is_supported_materialized(row):
                out_rows.append(row); counts["not_materialized"] += 1; continue
            if args.limit and done >= args.limit:                      # smoke cap: leave the rest as-is
                out_rows.append(row); counts["skipped_limit"] += 1; continue

            rid = row.get("id")
            if rid in cache:                                           # resume
                best_codes, best_fid = cache[rid]["codes"], cache[rid]["fid"]
            else:
                codes, rec = best_of_n_for_row(model, processor, row, n=args.n,
                                               input_field=cfg.input_field, chunk=args.chunk,
                                               sampling={"temperature": args.temperature, "top_p": args.top_p},
                                               device=model.device,
                                               fast=getattr(args, "fast_reward", False))
                best_codes, best_fid = codes, (rec or {}).get("behavioral_fidelity")
                cache_fh.write(json.dumps({"id": rid, "codes": best_codes, "fid": best_fid}) + "\n")
                cache_fh.flush()
            if best_fid is not None:
                fids.append(best_fid)
            new_row = distill_row(row, best_codes, best_fid, args.tau)
            counts["replaced" if new_row is not row else "kept_gold"] += 1
            out_rows.append(new_row)
            done += 1
            if done % 50 == 0:
                print(f"[distill] {done}/{n_train} train rows | replaced={counts['replaced']} "
                      f"kept_gold={counts['kept_gold']}")
    finally:
        cache_fh.close()

    mean_fid = float(sum(fids) / len(fids)) if fids else None
    print(f"[distill] DONE counts={dict(counts)} mean_best_of_N={mean_fid}")
    if args.dry_run:
        print("[distill][dry-run] wrote nothing (cache updated).")
        return 0

    # Write the distilled corpus (backup + .tmp + atomic replace; sort_keys for byte-stable output).
    if out_rows_p.exists():
        shutil.copy2(out_rows_p, out_rows_p.with_suffix(".jsonl.bak"))
    tmp = out_rows_p.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in out_rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(tmp, out_rows_p)
    print(f"[distill][write] {out_rows_p} ({len(out_rows)} rows)")

    # Manifest with a distillation provenance block (copy source manifest if present).
    man_out = out_dir / "active_manifest.json"
    man: dict = {}
    src_man = src.parent / "active_manifest.json"
    if src_man.is_file():
        man = json.loads(src_man.read_text(encoding="utf-8"))
    man["distillation"] = {
        "source_rows": str(src), "harvest_adapter": args.adapter, "n": args.n,
        "temperature": args.temperature, "top_p": args.top_p, "tau": args.tau,
        "counts": dict(counts), "mean_best_of_N": mean_fid,
    }
    man_out.write_text(json.dumps(man, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[distill][write] {man_out} (distillation block)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source-rows", default="data/active_sft/active_rows.jsonl",
                    help="P6 corpus to read (NOT a training config — avoids reading the distilled path)")
    ap.add_argument("--out-dir", default="data/active_sft_distilled",
                    help="where the distilled corpus is written")
    ap.add_argument("--config", default="configs/candidate_two_stage.json",
                    help="config for loading + conditioning input_field (NOT the source path)")
    ap.add_argument("--resized-model", default="models/base_resized")
    ap.add_argument("--adapter", required=True, help="harvest adapter (P6, or the prior distill round)")
    ap.add_argument("--n", type=int, default=16, help="best-of-N samples per training row")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--tau", type=float, default=0.30, help="absolute fidelity bar to accept a winner")
    ap.add_argument("--limit", type=int, default=0, help="cap TRAINING rows harvested (0=all); smoke lever")
    ap.add_argument("--fast-reward", action="store_true",
                    help="score best-of-N via the batched device-aware eval.fast_reward.score_batch "
                         "(one batched GPU decode + reduced measurement; parity-verified). Default off.")
    ap.add_argument("--dry-run", action="store_true", help="harvest + report; write no corpus")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
