"""Generate the unsupported / refusal corpus (teacher) and integrate it into the dataset.

Produces (image, natural refusal prompt, ``<unsupported>``) rows across all unsupported
categories + mixed boundary prompts (docs/detailed_behavior_spec.md "Unsupported Prompt Space";
data_collection_plan.md "Prompt Difficulty Mix": unsupported = 5-10% of the active SFT set).

Pipeline:
  1. Build a leakage-safe image pool: source images NOT used by any supported active row, split
     disjointly into an eval-reserved slice and a train slice (one unique image per row).
  2. Deterministic balanced plan: round-robin over the 11 out_of_scope categories + 3 out_of_gamut
     categories (ADR 0023) + 6 mixed families. All are refuse rows; rows carry route=refuse and a
     refuse_kind of out_of_scope/out_of_gamut. Image paths are stored RELATIVE (portable) so the
     refusal rows resolve against $SLM_ARTIFACT_ROOT on Colab and train (AUDIT F2 fix).
  3. Teacher phrases each request (:class:`UnsupportedTeacherClient`); a deterministic validator
     (:func:`validate_unsupported_prompt`) rejects any phrasing that lost its category cue.
  4. Idempotently assemble rows: write ``unsupported_rows.jsonl`` (all) + manifest, append train
     rows to ``active_rows.jsonl`` (replacing any prior ``unsup_*`` rows), and stage the
     eval-reserved rows to ``unsupported_eval_rows.jsonl`` for the Stage 9 eval freeze.

Resumable/idempotent: teacher results cache to ``--out`` keyed by id; re-running skips done ids
and re-assembles artifacts from the cache. ``--dry-run`` prints prompts with no API call.

Usage:
    python -m scripts.generate_unsupported --dry-run --limit 4
    python -m scripts.generate_unsupported --n-train 300 --n-eval 250
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path
from typing import Optional

from data_pipeline.active_dataset import SftRow
from data_pipeline.unsupported_gen import (
    MIXED_FAMILIES,
    PURE_CATEGORIES,
    SUPPORTED_ATTRS,
    UnsupportedTeacherClient,
    validate_unsupported_prompt,
)
from eval.refuse_taxonomy import (
    OUT_OF_GAMUT_CATEGORIES,
    REFUSE_OUT_OF_SCOPE,
    ROUTE_REFUSE,
    refuse_kind_for_category,
)

_ACTIVE_ROWS = "data/active_sft/active_rows.jsonl"
_SEED = 20260709
# token/tokenizer fields are not applicable to a refusal row (target is the literal <unsupported>).
_NA = "not_applicable"


def to_portable_image_path(path: str) -> str:
    """Return a corpus-relative (portable) image path — the fix for AUDIT F2 (ADR 0023).

    The unsupported rows previously stored ABSOLUTE paths (``/Users/.../luts/raw/...``) which fail to
    resolve on Colab (``sft.example.resolve_image`` returns an absolute path unchanged, so the file
    is missing and every refusal row is skipped every epoch). Supported rows store repo-relative
    paths like ``luts/raw/...`` that resolve against ``$SLM_ARTIFACT_ROOT`` (the staged corpus). We
    anchor on the ``luts/`` corpus root so the result is identical to the supported-row convention
    regardless of the machine's repo location. Already-relative paths pass through normalized.
    """
    p = str(path).replace(os.sep, "/")
    marker = "/luts/"
    idx = p.find(marker)
    if idx != -1:
        return p[idx + 1:]          # from "luts/..." onward (drop the leading slash)
    if p.startswith("luts/"):
        return p
    return os.path.relpath(p, os.getcwd()).replace(os.sep, "/") if os.path.isabs(p) else p


def _supported_images() -> set[str]:
    imgs: set[str] = set()
    if os.path.exists(_ACTIVE_ROWS):
        for line in open(_ACTIVE_ROWS, encoding="utf-8"):
            ip = json.loads(line).get("image_path")
            if ip:
                imgs.add(os.path.abspath(ip))
    return imgs


def _source_image_pool() -> list[str]:
    """All candidate source images (ppr10k + fivek), absolute, sorted for determinism."""
    pats = [
        "luts/raw/ppr10k/**/before.jpg",
        "luts/raw/fivek*/**/*.jpg",
        "luts/raw/fivek*/**/*.png",
    ]
    found: set[str] = set()
    for p in pats:
        for f in glob.glob(p, recursive=True):
            if os.path.isfile(f):
                found.add(os.path.abspath(f))
    return sorted(found)


def _split_unit(img: str) -> str:
    p = Path(img)
    return f"unsup:{p.parent.name}_{p.stem}"


def build_plan(n_train: int, n_eval: int) -> list[dict]:
    """Deterministic, leakage-safe, category-balanced plan (one unique free image per row)."""
    supported = _supported_images()
    free = [f for f in _source_image_pool() if f not in supported]
    need = n_train + n_eval
    if len(free) < need:
        raise SystemExit(f"only {len(free)} free images (need {need}); lower --n-train/--n-eval")
    rng = random.Random(_SEED)
    rng.shuffle(free)
    eval_imgs, train_imgs = free[:n_eval], free[n_eval:n_eval + n_train]

    # Refuse buckets, round-robin balanced: out_of_scope pure + out_of_gamut (ADR 0023) + mixed.
    # All are refuse rows (target <unsupported>); refuse_kind distinguishes out_of_scope/out_of_gamut.
    buckets = [("pure", c) for c in PURE_CATEGORIES] + \
              [("gamut", c) for c in OUT_OF_GAMUT_CATEGORIES] + \
              [("mixed", i) for i in range(len(MIXED_FAMILIES))]

    def _items(n: int, split: str, imgs: list[str], headline: bool) -> list[dict]:
        out = []
        for i in range(n):
            kind, key = buckets[i % len(buckets)]
            item = {"image_path": imgs[i], "split": split, "headline_eligible": headline,
                    "split_unit_id": _split_unit(imgs[i]), "route": ROUTE_REFUSE}
            if kind in ("pure", "gamut"):
                item.update(mixed=False, category=key,
                            unsupported_components=[key], supported_components=[],
                            refuse_kind=refuse_kind_for_category(key))
            else:
                fam = MIXED_FAMILIES[key]
                attr_pair = SUPPORTED_ATTRS[i % len(SUPPORTED_ATTRS)]
                item.update(mixed=True, category=fam["category"],
                            component_category=fam["component_category"],
                            unsupported_components=[fam["unsupported_component"]],
                            supported_components=[attr_pair[0]],
                            supported_attr=attr_pair[0], _attr_pair=attr_pair,
                            refuse_kind=REFUSE_OUT_OF_SCOPE)
            out.append(item)
        return out

    plan: list[dict] = []
    for i, it in enumerate(_items(n_eval, "eval", eval_imgs, True), start=1):
        plan.append({**it, "id": f"unsup_eval_{i:06d}"})
    for i, it in enumerate(_items(n_train, "train", train_imgs, False), start=1):
        plan.append({**it, "id": f"unsup_train_{i:06d}"})
    return plan


def _row_from(plan_item: dict, prompt: str) -> SftRow:
    return SftRow(
        id=plan_item["id"], is_supported=False, source_family="unsupported_teacher",
        image_path=to_portable_image_path(plan_item["image_path"]),   # relative -> resolves on Colab
        instruction=prompt, instruction_natural=prompt,
        instruction_status="teacher_generated", assistant_target="<unsupported>", target_tokens=[],
        token_status=_NA, canonical_domain_id=None, tokenizer_version=_NA,
        vq_codebook_sha256=_NA, vq_decoder_sha256=_NA,
        split_unit_id=plan_item["split_unit_id"], split=plan_item["split"],
        headline_eligible=bool(plan_item["headline_eligible"]),
        support_label="unsupported", unsupported_category=plan_item["category"],
        unsupported_components=list(plan_item["unsupported_components"]),
        mixed_prompt=bool(plan_item.get("mixed")),
        route=plan_item.get("route", ROUTE_REFUSE),
        refuse_kind=plan_item.get("refuse_kind") or refuse_kind_for_category(plan_item["category"]),
    )


def _load_done(path: str) -> dict:
    done: dict[str, dict] = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line)
                if r.get("id"):
                    done[r["id"]] = r
    return done


def _assemble(cache: dict, plan: list[dict], paths: dict) -> dict:
    """Rebuild artifacts from the cache (idempotent). Returns a summary dict."""
    by_id = {p["id"]: p for p in plan}
    accepted = [r for r in cache.values()
                if r.get("status") == "generated" and r.get("id") in by_id]
    train_rows, eval_rows, all_rows = [], [], []
    for r in accepted:
        row = _row_from(by_id[r["id"]], r["prompt"]).to_dict()
        all_rows.append(row)
        (eval_rows if by_id[r["id"]]["split"] == "eval" else train_rows).append(row)

    def _dump(path: str, rows: list[dict]):
        with open(path, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    _dump(paths["all"], all_rows)
    _dump(paths["eval"], eval_rows)

    # Idempotently merge train rows into active_rows.jsonl: drop any prior unsup_* rows first.
    if os.path.exists(_ACTIVE_ROWS):
        kept = [l for l in open(_ACTIVE_ROWS, encoding="utf-8")
                if not json.loads(l).get("id", "").startswith("unsup_")]
        with open(_ACTIVE_ROWS, "w", encoding="utf-8") as fh:
            for l in kept:
                fh.write(l if l.endswith("\n") else l + "\n")
            for row in train_rows:
                fh.write(json.dumps(row, sort_keys=True) + "\n")

    import collections
    cat = collections.Counter(by_id[r["id"]]["category"] for r in accepted)
    return {"generated": len(accepted), "train_rows": len(train_rows),
            "eval_rows": len(eval_rows), "by_category": dict(cat),
            "cache_total": len(cache)}


def run(out: str, config: str, n_train: int, n_eval: int, limit: Optional[int],
        attach_image: bool, dry_run: bool) -> int:
    plan = build_plan(n_train, n_eval)
    teacher = UnsupportedTeacherClient(config, attach_image=attach_image)
    if not dry_run and not teacher.is_available():
        print(f"[unsup] teacher_primary not available in {config}; pass --dry-run to inspect.")
        return 2

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cache = {} if dry_run else _load_done(out)
    if cache:
        print(f"[unsup] resuming: {len(cache)} rows already in {out}")
    sink = None if dry_run else open(out, "a", encoding="utf-8")
    tally = {"generated": 0, "rejected": 0, "error": 0, "dry_run": 0, "skipped": 0}
    processed = 0

    for item in plan:
        rid = item["id"]
        if rid in cache:
            tally["skipped"] += 1
            continue
        if limit is not None and processed >= limit:
            break
        processed += 1
        if dry_run:
            from data_pipeline.unsupported_gen import build_messages
            msgs = build_messages(item, attach_image=attach_image)
            print(f"\n===== {rid} [{item['category']}"
                  f"{'/mixed' if item.get('mixed') else ''}] =====")
            print(msgs[0]["content"][:200], "...")
            print("USER:", [p for p in msgs[1]["content"] if p.get("type") == "text"][0]["text"])
            tally["dry_run"] += 1
            continue
        try:
            gen = teacher.generate(item)
        except Exception as exc:  # noqa: BLE001
            rec = {"id": rid, "status": "error", "error": f"{type(exc).__name__}: {exc}"}
            tally["error"] += 1
        else:
            ok, issues = validate_unsupported_prompt(gen["prompt"], item)
            rec = {"id": rid, "status": "generated" if ok else "rejected",
                   "prompt": gen["prompt"], "category": item["category"],
                   "mixed": bool(item.get("mixed")), "validation_ok": ok,
                   "validation_issues": issues, "extra": gen.get("extra"),
                   "provenance": gen.get("provenance")}
            tally["generated" if ok else "rejected"] += 1
        cache[rid] = rec
        sink.write(json.dumps(rec, sort_keys=True) + "\n")
        sink.flush()
        status = rec["status"]
        note = f" issues={rec.get('validation_issues')}" if rec.get("validation_issues") else ""
        err = f" {rec.get('error')}" if rec.get("error") else ""
        print(f"  {rid}: {status}{note}{err}  {rec.get('prompt','')[:70]}")
    if sink:
        sink.close()

    print(f"[unsup] {tally}")
    if dry_run:
        return 0

    summary = _assemble(cache, plan, {
        "all": "data/active_sft/unsupported_rows.jsonl",
        "eval": "data/active_sft/unsupported_eval_rows.jsonl"})
    manifest = {"plan_size": len(plan), "n_train": n_train, "n_eval": n_eval,
                "seed": _SEED, "tally": tally, **summary}
    Path("data/active_sft/unsupported_gen_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[unsup] assembled: {summary}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate unsupported/refusal rows (teacher).")
    ap.add_argument("--out", default="data/active_sft/unsupported_cache.jsonl")
    ap.add_argument("--config", default="configs/model_clients.yaml")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=250)
    ap.add_argument("--limit", type=int, default=None, help="first N not-yet-done plan items")
    ap.add_argument("--no-image", action="store_true", help="text-only teacher (skip source image)")
    ap.add_argument("--dry-run", action="store_true", help="print prompts, no API call")
    args = ap.parse_args(argv)
    return run(args.out, args.config, args.n_train, args.n_eval, args.limit,
               attach_image=not args.no_image, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
