"""Build the static Best-of-N showcase artifact for the webapp demo page.

The deployable inference win is BEST-OF-N: sample N candidate 64-code sequences, decode + score
each against the REQUESTED spec with the deployable reranker (:func:`eval.behavioral_fidelity.
rerank_key` — behavioral fidelity + collapse/entropy tie-breaks, *no target LUT*), and return the
best. That path needs the Qwen2.5-VL generator on a GPU, so it cannot run in a browser. This script
bakes a precomputed, self-contained artifact (JSON + rendered before/after PNGs) that the static
page ``webapp/static/best-of-n.html`` steps through client-side — no backend, no model at view time.

Two provenance modes, identical output schema:

* **default (GPU-free)** — ``candidate_source: "corpus_reranker_demo"``. Candidates are real LUT looks
  drawn from the corpus (``data/active_sft/active_rows.jsonl``), decoded by the *frozen* VQ decoder
  and scored by the *real* reranker. This demonstrates the deployable reranker + decoder + scorer on
  real data with zero model weights. It does NOT claim the candidates are the generator's own samples.
* ``--from-model --adapter <path>`` (GPU/MPS) — ``candidate_source: "model_sampled"``. Loads the real
  generator and samples N candidates per prompt (:func:`sft.generate.generate_codes_batch`) plus a
  greedy baseline, scoring each with the same reranker. This makes the stepper fully live.

The aggregate fidelity ladder + oracle@N curve are the REAL measured numbers from the P6 held-out
slice (``notebooks/phase1_behavioral_score.ipynb`` / ``phase2_oracle_at_n.ipynb``); they are copied
verbatim, never fabricated.

Usage::

    python -m scripts.build_best_of_n_showcase                 # GPU-free reranker demo (default)
    python -m scripts.build_best_of_n_showcase --examples 3 --pool 5
    python -m scripts.build_best_of_n_showcase --from-model \\
        --adapter models/sft_adapters/p6_twostage_d0f9c744_smokefull --n 8
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from data_pipeline.attribute_spec import ground_truth_attribute_spec_text
from eval.behavioral_fidelity import decode_codes, rerank_key, score_from_lut, score_generation
from webapp.lut import apply_lut, export_cube, save_image

_REPO = Path(__file__).resolve().parent.parent
_ROWS = _REPO / "data/active_sft/active_rows.jsonl"
_REFS = _REPO / "webapp/assets/references"
_OUT_DIR = _REPO / "webapp/static/best_of_n"
_OUT_JSON = _REPO / "webapp/static/best_of_n_showcase.json"

# The REAL measured story (P6 adapter p6_twostage_d0f9c744_smokefull), copied verbatim from the
# notebook outputs — NOT recomputed here. See docs/collapse_fix/README.md.
AGGREGATE = {
    "source": (
        "P6 adapter (p6_twostage_d0f9c744_smokefull) on the held-out slice; "
        "notebooks/phase1_behavioral_score.ipynb (fidelity ladder, n=64) and "
        "notebooks/phase2_oracle_at_n.ipynb (oracle@N, n=32)."
    ),
    "ladder": [
        {"label": "Teacher-forced argmax", "fidelity": 0.708, "collapse_rate": 0.0,
         "kind": "mirage", "note": "Optimistic ceiling — feeds the gold prefix at every step, so it never sees the model commit to its own trajectory."},
        {"label": "Free-running greedy", "fidelity": 0.159, "collapse_rate": 0.94,
         "kind": "baseline", "note": "The shipped baseline. Over-commits to one dominant code and collapses to a low-fidelity LUT that mostly misses the request (exposure bias)."},
        {"label": "Free-running sample t=0.7", "fidelity": 0.091, "collapse_rate": 0.14,
         "kind": "single", "note": "A single hot sample is diverse but individually often points the wrong way."},
        {"label": "Best-of-N reranked (t=0.7)", "fidelity": 0.287, "collapse_rate": 0.03,
         "kind": "winner", "note": "Sample 32, decode + score each by behavioral fidelity, keep the best. Nearly doubles greedy."},
        {"label": "Best-of-N reranked (t=1.0)", "fidelity": 0.307, "collapse_rate": 0.0,
         "kind": "winner", "note": "A hotter temperature covers more good trajectories; the reranker captures them."},
        {"label": "Real-corpus codes", "fidelity": 0.89, "collapse_rate": 0.0,
         "kind": "ceiling", "note": "Fidelity of the ground-truth codes themselves — the metric's practical ceiling."},
    ],
    "oracle_at_n": {
        # oracle@k = mean over rows of the max fidelity among the first k samples (reranker ceiling);
        # best_of_N = what rerank_key actually picks (the deploy number).
        "t=0.7": {"1": 0.122, "4": 0.182, "8": 0.217, "16": 0.250, "32": 0.287, "best_of_N": 0.287},
        "t=1.0": {"1": 0.083, "4": 0.180, "8": 0.230, "16": 0.267, "32": 0.307, "best_of_N": 0.307},
    },
    "greedy_baseline": 0.159,
    "greedy_collapse_rate": 0.94,
    "best_of_n_fidelity": 0.307,        # best_of_N pick at t=1.0
    "best_pick_collapse_rate": 0.0,     # collapse rate of the reranked pick at t=1.0
}

# Curated diverse examples (real corpus rows). Each is keyed by a substring of its spec so selection
# is deterministic and spans distinct looks; falls back to the first matching grade row.
_EXAMPLE_HINTS = [
    ("split_strength", "Split-toned warmth"),
    ("more_contrast", "Punchier contrast"),
    ("cooler", "Cooler cast"),
    ("chroma=-", "Muted / faded"),
    ("brighter", "Lifted & brighter"),
]


def _load_rows() -> list[dict]:
    return [json.loads(line) for line in _ROWS.read_text().splitlines() if line.strip()]


def _grade_rows(rows: list[dict]) -> list[dict]:
    out = []
    for row in rows:
        spec = ground_truth_attribute_spec_text(row) or ""
        if spec.startswith("route=grade") and len(row.get("target_tokens", [])) == 64:
            row["_spec_text"] = spec
            out.append(row)
    return out


def _pick_examples(grade_rows: list[dict], k: int) -> list[dict]:
    picked, used = [], set()
    for hint, _title in _EXAMPLE_HINTS:
        if len(picked) >= k:
            break
        for row in grade_rows:
            if row["id"] in used:
                continue
            if hint in row["_spec_text"]:
                picked.append(row)
                used.add(row["id"])
                break
    # Backfill if hints under-fill.
    for row in grade_rows:
        if len(picked) >= k:
            break
        if row["id"] not in used:
            picked.append(row)
            used.add(row["id"])
    return picked[:k]


def _thumb(path: Path, max_edge: int) -> Image.Image:
    img = Image.open(path).convert("RGB")
    img.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    return img


def _render(image: Image.Image, lut: np.ndarray, dest: Path) -> None:
    arr = np.asarray(image, dtype=np.float64) / 255.0
    graded = apply_lut(arr, lut)
    save_image(np.rint(graded * 255.0).astype(np.uint8), dest)


def _dominant_collapse_codes(codes: list[int]) -> list[int]:
    """The greedy exposure-bias failure mode: over-commit to a single dominant code."""
    vals, counts = np.unique(np.asarray(codes), return_counts=True)
    dominant = int(vals[int(counts.argmax())])
    return [dominant] * 64


def _score_codes(codes: list[int], spec_text: str) -> tuple[np.ndarray, dict]:
    """Deploy-path scoring: decode + score against the requested spec, NO target LUT."""
    rec = score_generation(codes, spec_text)  # no target_codes -> no ΔE leakage (matches deploy)
    lut = decode_codes(codes)
    return lut, rec


def _candidate_entry(index, source, label, codes, rec, is_winner, hero_rel, note=None):
    cs = rec.get("code_stats") or {}
    entry = {
        "index": index,
        "source": source,
        "label": label,
        "behavioral_fidelity": rec.get("behavioral_fidelity"),
        "collapsed": bool(rec.get("collapsed")),
        "entropy_norm": cs.get("entropy_norm"),
        "dominant_share": cs.get("dominant_share"),
        "residual_norm": rec.get("residual_norm"),
        "is_winner": is_winner,
        "graded_url": hero_rel,
        "codes_preview": [int(c) for c in codes[:12]],
    }
    if note:
        entry["note"] = note
    return entry


def build_corpus_demo(examples: int, pool: int, hero_edge: int, ref_edge: int, seed: int) -> dict:
    rng = random.Random(seed)
    grade_rows = _grade_rows(_load_rows())
    chosen = _pick_examples(grade_rows, examples)
    # A shared pool of "distractor" looks (other real corpus targets) to rerank against.
    distractor_pool = [r for r in grade_rows if r["id"] not in {c["id"] for c in chosen}]
    rng.shuffle(distractor_pool)

    out_examples = []
    for ei, row in enumerate(chosen):
        ex_dir = _OUT_DIR / f"ex{ei}"
        ex_dir.mkdir(parents=True, exist_ok=True)
        spec_text = row["_spec_text"]
        hero_src = _REPO / row["image_path"]
        hero = _thumb(hero_src, hero_edge)
        save_image(hero, ex_dir / "hero_original.png")

        # Candidate pool: the on-spec target + (pool-1) distinct distractor looks + 1 collapse mode.
        entries_src = [("corpus:self", "On-spec look", row["target_tokens"])]
        for j in range(pool - 1):
            dr = distractor_pool[(ei * pool + j) % len(distractor_pool)]
            entries_src.append((f"corpus:alt{j}", "Alternative look", dr["target_tokens"]))
        collapse_codes = _dominant_collapse_codes(row["target_tokens"])
        entries_src.append(("synthetic:collapse", "Collapse mode (illustrative)", collapse_codes))

        order = list(range(len(entries_src)))
        rng.shuffle(order)  # so the reranker's pick is not trivially first

        scored = []  # (order_index, source, label, codes, rec, lut)
        for out_idx, src_idx in enumerate(order):
            source, label, codes = entries_src[src_idx]
            lut, rec = _score_codes(codes, spec_text)
            scored.append((out_idx, source, label, codes, rec, lut))

        winner = max(scored, key=lambda t: rerank_key(t[4]))
        winner_index = winner[0]

        candidates = []
        for out_idx, source, label, codes, rec, lut in scored:
            hero_rel = f"best_of_n/ex{ei}/cand{out_idx}_graded.png"
            _render(hero, lut, _OUT_DIR / f"ex{ei}/cand{out_idx}_graded.png")
            note = None
            if source == "synthetic:collapse":
                note = ("Illustrative greedy failure mode: a single dominant code fills all 64 positions. "
                        "It scores low fidelity against the request; the reranker's collapse flag is a "
                        "tie-break (not a hard veto), so it also loses on ties.")
            candidates.append(_candidate_entry(out_idx, source, label, codes, rec,
                                               out_idx == winner_index, hero_rel, note))

        # Consistency check: apply the WINNER LUT to a few reference photos.
        winner_lut = winner[5]
        refs = []
        for ref_name, ref_file in (("Portrait", "portrait.jpg"), ("City", "city.jpg"), ("Landscape", "landscape.jpg")):
            ref_path = _REFS / ref_file
            if not ref_path.is_file():
                continue
            ref_img = _thumb(ref_path, ref_edge)
            save_image(ref_img, _OUT_DIR / f"ex{ei}/ref_{ref_name.lower()}_original.png")
            _render(ref_img, winner_lut, _OUT_DIR / f"ex{ei}/ref_{ref_name.lower()}_graded.png")
            refs.append({
                "name": ref_name,
                "original_url": f"best_of_n/ex{ei}/ref_{ref_name.lower()}_original.png",
                "graded_url": f"best_of_n/ex{ei}/ref_{ref_name.lower()}_graded.png",
            })

        export_cube(winner_lut, _OUT_DIR / f"ex{ei}/winner.cube")

        out_examples.append({
            "id": row["id"][:12],
            "instruction": row.get("instruction") or row.get("instruction_natural") or "",
            "spec_text": spec_text,
            "hero_original_url": f"best_of_n/ex{ei}/hero_original.png",
            "winner_cube_url": f"best_of_n/ex{ei}/winner.cube",
            "n": len(candidates),
            "winner_index": winner_index,
            "candidates": candidates,
            "references": refs,
        })

    return {
        "meta": {
            "candidate_source": "corpus_reranker_demo",
            "reranker": "eval.behavioral_fidelity.rerank_key (behavioral fidelity + collapse/entropy/ΔE tie-breaks; no target LUT)",
            "decoder": "frozen VQ-VAE at tokenizer/final/",
            "note": ("Candidates are REAL LUT looks from the corpus, decoded and scored by the deployable "
                     "reranker. This demonstrates the reranker + decoder + scorer on real data with no "
                     "model weights. It is NOT the generator's own samples — regenerate with --from-model "
                     "for the live sampled candidates."),
        },
        "aggregate": AGGREGATE,
        "examples": out_examples,
    }


def build_from_model(adapter: str, resized: str, examples: int, n: int, chunk: int,
                     temperature: float, top_p: float, hero_edge: int, ref_edge: int, seed: int) -> dict:
    """Live path: sample N candidates from the real generator per prompt (needs GPU/MPS)."""
    import torch
    from sft.example import input_text_for, resolve_image
    from sft.generate import generate_codes, generate_codes_batch
    from sft.loader import load_eval_model
    from sft.score_tokens import _load_config

    # Seed sampling for reproducibility. (Determinizes on CPU/CUDA; MPS/Metal has residual
    # nondeterminism, so an MPS re-run may still differ slightly.)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = _load_config("configs/candidate_two_stage.json")
    model, processor = load_eval_model(cfg, resized, adapter)
    device = getattr(model, "device", None)

    grade_rows = _grade_rows(_load_rows())
    chosen = _pick_examples(grade_rows, examples)

    out_examples = []
    for ei, row in enumerate(chosen):
        ex_dir = _OUT_DIR / f"ex{ei}"
        ex_dir.mkdir(parents=True, exist_ok=True)
        spec_text = row["_spec_text"]
        image = resolve_image(row["image_path"])
        cond_text = input_text_for(row, "attribute_spec_text")

        hero = _thumb(_REPO / row["image_path"], hero_edge)
        save_image(hero, ex_dir / "hero_original.png")

        sampled = generate_codes_batch(model, processor, image=image, text=cond_text, n=n,
                                       sampling={"temperature": temperature, "top_p": top_p},
                                       chunk=chunk, device=device)
        greedy = generate_codes(model, processor, image=image, text=cond_text, sampling=None, device=device)

        entries_src = []
        if greedy is not None and len(greedy) == 64:
            entries_src.append(("model:greedy", "Free-running greedy", list(greedy)))
        for j, codes in enumerate(c for c in sampled if c is not None and len(c) == 64):
            entries_src.append((f"model:sample{j}", f"Sample t={temperature}", list(codes)))

        scored = []
        for out_idx, (source, label, codes) in enumerate(entries_src):
            lut, rec = _score_codes(codes, spec_text)
            scored.append((out_idx, source, label, codes, rec, lut))
        # Winner among the SAMPLED candidates only (greedy is shown as the baseline, not a rerank pick).
        sampled_scored = [t for t in scored if t[1] != "model:greedy"] or scored
        winner = max(sampled_scored, key=lambda t: rerank_key(t[4]))
        winner_index = winner[0]

        candidates = []
        for out_idx, source, label, codes, rec, lut in scored:
            _render(hero, lut, ex_dir / f"cand{out_idx}_graded.png")
            note = "Free-running greedy baseline (not a rerank candidate)." if source == "model:greedy" else None
            candidates.append(_candidate_entry(out_idx, source, label, codes, rec,
                                               out_idx == winner_index, f"best_of_n/ex{ei}/cand{out_idx}_graded.png", note))

        refs = []
        for ref_name, ref_file in (("Portrait", "portrait.jpg"), ("City", "city.jpg"), ("Landscape", "landscape.jpg")):
            ref_path = _REFS / ref_file
            if not ref_path.is_file():
                continue
            ref_img = _thumb(ref_path, ref_edge)
            save_image(ref_img, ex_dir / f"ref_{ref_name.lower()}_original.png")
            _render(ref_img, winner[5], ex_dir / f"ref_{ref_name.lower()}_graded.png")
            refs.append({"name": ref_name,
                         "original_url": f"best_of_n/ex{ei}/ref_{ref_name.lower()}_original.png",
                         "graded_url": f"best_of_n/ex{ei}/ref_{ref_name.lower()}_graded.png"})

        export_cube(winner[5], ex_dir / "winner.cube")
        out_examples.append({
            "id": row["id"][:12], "instruction": row.get("instruction", ""), "spec_text": spec_text,
            "hero_original_url": f"best_of_n/ex{ei}/hero_original.png",
            "winner_cube_url": f"best_of_n/ex{ei}/winner.cube",
            "n": len(candidates), "winner_index": winner_index,
            "candidates": candidates, "references": refs,
        })

    return {
        "meta": {
            "candidate_source": "model_sampled",
            "adapter": adapter,
            "sampling": {"n": n, "temperature": temperature, "top_p": top_p},
            "reranker": "eval.behavioral_fidelity.rerank_key (no target LUT)",
            "decoder": "frozen VQ-VAE at tokenizer/final/",
            "note": "Candidates are the generator's own free-running samples, reranked by behavioral fidelity.",
        },
        "aggregate": AGGREGATE,
        "examples": out_examples,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--examples", type=int, default=3, help="number of showcase prompts")
    ap.add_argument("--pool", type=int, default=5, help="candidate-pool size per prompt (corpus mode)")
    ap.add_argument("--hero-edge", type=int, default=720, help="max edge of the hero before/after render")
    ap.add_argument("--ref-edge", type=int, default=380, help="max edge of the reference thumbnails")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--from-model", action="store_true", help="sample from the real generator (needs GPU/MPS)")
    ap.add_argument("--adapter", default="models/sft_adapters/p6_twostage_d0f9c744_smokefull")
    ap.add_argument("--resized-model", default="models/base_resized")
    ap.add_argument("--n", type=int, default=8, help="samples per prompt (--from-model)")
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    args = ap.parse_args(argv)

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.from_model:
        payload = build_from_model(args.adapter, args.resized_model, args.examples, args.n, args.chunk,
                                   args.temperature, args.top_p, args.hero_edge, args.ref_edge, args.seed)
    else:
        payload = build_corpus_demo(args.examples, args.pool, args.hero_edge, args.ref_edge, args.seed)

    _OUT_JSON.write_text(json.dumps(payload, indent=2))
    src = payload["meta"]["candidate_source"]
    print(f"[showcase] wrote {_OUT_JSON.relative_to(_REPO)} ({len(payload['examples'])} examples, source={src})")
    print(f"[showcase] images under {_OUT_DIR.relative_to(_REPO)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
