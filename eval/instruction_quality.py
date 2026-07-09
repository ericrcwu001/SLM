"""Evaluate the teacher/judge interaction (docs/adr/0004-teacher-prompt-quality-gates.md).

The judge is non-authoritative, so "eval the interaction" means: is the judge a well-calibrated
advisory filter that adds signal the authoritative deterministic tag<->behavior gate cannot,
without sharing the teacher's blind spots? Two lenses:

  A. agreement_matrix -- cross-tab the deterministic gate (validation_ok) against the judge
     verdict over an instruction-gen run (scripts/generate_instructions.py output). The
     off-diagonal is the whole story: det-FAIL/judge-PASS = the judge missed a measurable
     inconsistency; det-PASS/judge-FAIL = the judge's independent value-add (local claims,
     leakage) OR a false positive to audit. Free — reuses data the pipeline already emits.

  C. synthetic negatives -- deliberately corrupt clean instructions (flip a tag's direction,
     inject a local edit / impossible preservation / aesthetic ranking, diverge concise vs
     natural) and measure per-defect catch rate. Confirms the interaction actually rejects the
     bad cases. Direction flips must be caught deterministically; the language defects are the
     judge's job. The judge is pluggable (``judge_fn``) so this runs offline in tests.

CLI:
    python -m eval.instruction_quality --manifest data/active_sft/instructions.jsonl
    python -m eval.instruction_quality --self-check            # synth negatives + catch rate
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Callable, Optional

from data_pipeline.instruction_gen import _TAG_BEHAVIOR, validate_tags_against_behavior

# --- A. deterministic x judge agreement matrix -------------------------------------
def _det_status(row: dict) -> str:
    v = row.get("validation_ok")
    return "pass" if v is True else ("fail" if v is False else "n/a")


def _judge_status(row: dict) -> str:
    j = row.get("judge")
    if not isinstance(j, dict):
        return "not_run"
    return str(j.get("status", "?"))


def agreement_matrix(rows: list[dict]) -> dict:
    """Cross-tab deterministic gate vs judge verdict over instruction-gen per-row results."""
    cells: dict[tuple[str, str], int] = defaultdict(int)
    examples: dict[tuple[str, str], list] = defaultdict(list)
    authored = 0
    for r in rows:
        if r.get("instruction_status") == "source_authored":
            authored += 1
            continue
        key = (_det_status(r), _judge_status(r))
        cells[key] += 1
        if len(examples[key]) < 8:
            examples[key].append(r.get("id"))

    def g(d: str, j: str) -> int:
        return cells.get((d, j), 0)

    return {
        "n_rows": len(rows),
        "authored_skipped": authored,
        # diagonals (agreement)
        "agree_accept": g("pass", "pass") + g("pass", "not_run"),
        "agree_reject": g("fail", "fail") + g("fail", "not_run"),
        # off-diagonals (the interesting part)
        "judge_missed_measurable": g("fail", "pass"),   # det caught it, judge waved it through
        "judge_only_flag": g("pass", "fail"),           # judge's independent flag or false positive
        "judge_not_evaluated": g("pass", "not_evaluated") + g("fail", "not_evaluated"),
        "cells": {f"det={d}|judge={j}": n for (d, j), n in sorted(cells.items())},
        "disagreement_examples": {
            "judge_missed_measurable": examples.get(("fail", "pass"), []),
            "judge_only_flag": examples.get(("pass", "fail"), []),
        },
    }


def load_manifest_rows(path: str) -> list[dict]:
    """Load per-row instruction-gen results from a JSONL (scripts.generate_instructions --out)."""
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# --- C. synthetic-negative catch-rate harness --------------------------------------
# opposite directional tag (share a behavior key with the opposite required sign).
_BY_KEY: dict[str, dict[int, str]] = {}
for _t, (_k, _s) in _TAG_BEHAVIOR.items():
    _BY_KEY.setdefault(_k, {})[_s] = _t
_OPPOSITE = {t: _BY_KEY[k].get(-s) for t, (k, s) in _TAG_BEHAVIOR.items() if _BY_KEY[k].get(-s)}

DEFAULT_SEEDS: list[dict] = [
    {"id": "seed_warm_muted", "gold_tags": ["warmer", "muted"],
     "concise": "Make it warmer and more muted.",
     "natural": "Give it a warm, gentle, muted look.",
     "measured_behavior": {"temperature_delta_b": 4.0, "chroma_delta": -3.0}},
    {"id": "seed_bright_contrast", "gold_tags": ["brighter", "more_contrast"],
     "concise": "Brighten it and add contrast.",
     "natural": "Make it punchier and brighter.",
     "measured_behavior": {"mean_l_delta": 5.0, "contrast_l_spread_delta": 4.0}},
    {"id": "seed_cinematic", "gold_tags": ["cinematic", "warmer"],
     "concise": "Give it a warm cinematic grade.",
     "natural": "A warm, filmic, cinematic mood.",
     "measured_behavior": {"temperature_delta_b": 3.0, "chroma_delta": -2.0,
                           "contrast_l_spread_delta": 3.0}},
]


def _flip_direction(rec: dict) -> Optional[dict]:
    tags = list(rec["gold_tags"])
    flipped = None
    for i, t in enumerate(tags):
        if t in _OPPOSITE:
            flipped = (t, _OPPOSITE[t])
            tags[i] = _OPPOSITE[t]
            break
    if not flipped:
        return None
    concise = rec["concise"].replace(flipped[0], flipped[1])
    return {**rec, "gold_tags": tags, "concise": concise}


def _inject_local(rec: dict) -> dict:
    return {**rec, "concise": rec["concise"] + " Also blur the background and sharpen the subject's eyes."}


def _impossible_preservation(rec: dict) -> dict:
    return {**rec, "concise": rec["concise"] + " Keep the sky exactly the same while doing this."}


def _aesthetic(rec: dict) -> dict:
    return {**rec, "concise": rec["concise"] + " Make it look the most beautiful and best possible."}


def _divergence(rec: dict) -> dict:
    # natural describes a different edit than concise
    return {**rec, "natural": "Make it cooler and much darker."}


# (name, corruptor, gate expected to catch it)
DEFECTS: list[tuple[str, Callable[[dict], Optional[dict]], str]] = [
    ("wrong_direction", _flip_direction, "deterministic"),
    ("local_edit", _inject_local, "judge"),
    ("impossible_preservation", _impossible_preservation, "judge"),
    ("aesthetic_ranking", _aesthetic, "judge"),
    ("concise_natural_divergence", _divergence, "judge"),
]


def synthesize_negatives(seeds: Optional[list[dict]] = None) -> list[dict]:
    """Build labeled negatives from clean seeds: one per (seed, defect)."""
    seeds = seeds if seeds is not None else DEFAULT_SEEDS
    out: list[dict] = []
    for s in seeds:
        for name, fn, catcher in DEFECTS:
            corrupted = fn(s)
            if corrupted is None:
                continue
            out.append({"defect": name, "expected_catcher": catcher, "record": corrupted})
    return out


def deterministic_catches(record: dict) -> bool:
    ok, _issues = validate_tags_against_behavior(
        record.get("gold_tags", []), record.get("measured_behavior", {}))
    return not ok


def default_judge_fn(model_clients_path: str = "configs/model_clients.yaml") -> Callable:
    """A judge_fn that calls the real judge; returns None if the judge is unavailable."""
    from eval import judge_client

    def _fn(record: dict):
        if not judge_client.is_available(model_clients_path):
            return None
        return judge_client.score_instruction(
            record.get("concise", ""), record.get("natural", "") or "",
            record.get("gold_tags", []), record.get("measured_behavior", {}),
            model_clients_path=model_clients_path)
    return _fn


def _judge_flags(result) -> bool:
    if result is None:
        return False
    if isinstance(result, bool):
        return result
    return getattr(result, "status", None) == "fail"


def run_catch_rate(negatives: list[dict], judge_fn: Optional[Callable] = None,
                   use_deterministic: bool = True) -> dict:
    """Per-defect catch rate: a negative is 'caught' if the deterministic gate rejects it or the
    judge flags it. ``judge_fn(record) -> LayerResult|bool|None``; None (no judge) counts as
    not-flagged so the deterministic-only floor is still measured."""
    by: dict[str, dict] = defaultdict(lambda: {"n": 0, "caught": 0, "by_det": 0, "by_judge": 0})
    for neg in negatives:
        d, rec = neg["defect"], neg["record"]
        det = use_deterministic and deterministic_catches(rec)
        judge = _judge_flags(judge_fn(rec)) if judge_fn is not None else False
        by[d]["n"] += 1
        by[d]["by_det"] += int(det)
        by[d]["by_judge"] += int(judge)
        by[d]["caught"] += int(det or judge)
    out = {d: {**v, "catch_rate": (v["caught"] / v["n"]) if v["n"] else None}
           for d, v in by.items()}
    total = sum(v["n"] for v in by.values())
    caught = sum(v["caught"] for v in by.values())
    out["_overall"] = {"n": total, "caught": caught,
                       "catch_rate": (caught / total) if total else None}
    return out


def _load_seeds(path: Optional[str]) -> list[dict]:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            return [json.loads(l) for l in fh if l.strip()]
    return DEFAULT_SEEDS


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Eval the teacher/judge interaction (A + C).")
    ap.add_argument("--manifest", help="instruction-gen per-row JSONL -> agreement matrix (A)")
    ap.add_argument("--self-check", action="store_true", help="synthetic-negative catch rate (C)")
    ap.add_argument("--seeds", default="eval/fixtures/instruction_seeds.jsonl")
    ap.add_argument("--config", default="configs/model_clients.yaml")
    args = ap.parse_args(argv)

    if args.manifest:
        rows = load_manifest_rows(args.manifest)
        print(json.dumps({"agreement_matrix": agreement_matrix(rows)}, indent=2))
    if args.self_check:
        from eval import judge_client

        negatives = synthesize_negatives(_load_seeds(args.seeds))
        have_judge = judge_client.is_available(args.config)
        judge_fn = default_judge_fn(args.config) if have_judge else None
        rates = run_catch_rate(negatives, judge_fn=judge_fn)
        print(json.dumps({
            "judge_available": have_judge,
            "note": None if have_judge else "judge unavailable -> deterministic-only floor "
                    "(only wrong_direction is catchable without the judge)",
            "catch_rate": rates,
        }, indent=2))
    if not args.manifest and not args.self_check:
        ap.error("pass --manifest and/or --self-check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
