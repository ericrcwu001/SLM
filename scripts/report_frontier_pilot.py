"""Generate the consolidated markdown report for the prompted-frontier LUT baseline pilot.

Reads the source-of-truth caches (data/eval/frontier_<config>.jsonl) produced by
scripts.generate_frontier_luts, re-scores every LUT (eval.cube_parser + eval.frontier_scoring),
attaches list-price cost + per-cell wall-clock (from the run logs in /tmp, best-effort), and
writes docs/frontier_baseline_results.md.

Regenerable: quantitative tables come from the caches; only per-cell timing depends on the
/tmp run logs still existing (older cells fall back to "n/a").

Usage: python -m scripts.report_frontier_pilot
"""

from __future__ import annotations

import json
import os
import re

from eval.cube_parser import parse_frontier_cube
from eval.frontier_scoring import score_lut
from eval.schemas import load_rows

OUT = "docs/frontier_baseline_results.md"

# --- pinned run facts (list prices $/1M; input tokens = per-model probe; see caveats) ---
CONFIGS = ["opus_4_8", "sonnet_4_6", "gpt_5_5", "gpt_5_5_xhigh", "gemini_3_1_pro"]
LABEL = {"opus_4_8": "Opus 4.8", "sonnet_4_6": "Sonnet 4.6", "gpt_5_5": "GPT-5.5 (medium)",
         "gpt_5_5_xhigh": "GPT-5.5 (xhigh)", "gemini_3_1_pro": "Gemini 3.1 Pro"}
SLUG = {"opus_4_8": "claude-group/claude-opus-4-8", "sonnet_4_6": "claude-group/claude-sonnet-4-6",
        "gpt_5_5": "openai-group/gpt-5.5", "gpt_5_5_xhigh": "openai-group/gpt-5.5",
        "gemini_3_1_pro": "gemini-group/gemini-3.1-pro"}
EFFORT = {"opus_4_8": "medium", "sonnet_4_6": "medium", "gpt_5_5": "medium",
          "gpt_5_5_xhigh": "xhigh", "gemini_3_1_pro": "(none)"}
PRICE = {"opus_4_8": (5, 25), "sonnet_4_6": (3, 15), "gpt_5_5": (5, 30),
         "gpt_5_5_xhigh": (5, 30), "gemini_3_1_pro": (2, 12)}
IN_EST = {"opus_4_8": 1277, "sonnet_4_6": 971, "gpt_5_5": 849, "gpt_5_5_xhigh": 849,
          "gemini_3_1_pro": 1908}

MAIN_ROWS = ["eval_sup_000001", "eval_sup_000002", "eval_sup_000003",
             "eval_sup_000004", "eval_sup_000005"]
PROBE_ROWS = ["eval_sup_000025", "eval_sup_000022", "eval_unsup_000013"]


def _logmap() -> dict:
    m = {("opus_4_8", "eval_sup_000001"): "/tmp/frontier_1row_opus_4_8.log",
         ("sonnet_4_6", "eval_sup_000001"): "/tmp/frontier_1row.log",
         ("gpt_5_5", "eval_sup_000001"): "/tmp/frontier_v2_gpt_5_5.log",
         ("gemini_3_1_pro", "eval_sup_000001"): "/tmp/frontier_v2_gemini_3_1_pro.log"}
    for c in CONFIGS:
        for r, rid in enumerate(MAIN_ROWS[1:], start=2):
            m[(c, rid)] = f"/tmp/frontier_row{r}_{c}.log"
    return m


def _elapsed(path: str):
    try:
        mt = re.search(r"elapsed_s=(\d+)", open(path).read())
        return int(mt.group(1)) if mt else None
    except (FileNotFoundError, KeyError, TypeError):
        return None


def _ratio(recs: dict) -> float:
    c = t = 0
    for d in recs.values():
        o = d["provenance"].get("output_tokens")
        if o:
            c += len(d["text"] or ""); t += o
    return c / t if t else 3.0


def _cell(cfg, rid, recs, ratio, meta, logs):
    d = recs.get(rid)
    if not d:
        return None
    p = parse_frontier_cube(d["text"])
    pv = d["provenance"]
    out = pv.get("output_tokens") or round(len(d["text"] or "") / ratio)
    intok = pv.get("input_tokens") or IN_EST[cfg]
    pin, pout = PRICE[cfg]
    cost = intok / 1e6 * pin + out / 1e6 * pout
    sec = _elapsed(logs.get((cfg, rid), ""))
    res, score = p.kind, ""
    if meta[rid]["is_supported"]:
        if p.kind == "raw_lut":
            s = score_lut(p.lut_abs, meta[rid]["gold_tags"])
            ok = s.direction.status == "pass" and s.safety.status == "pass"
            res = "PASS" if ok else ("valid/dir-fail" if s.direction.status != "not_evaluated" else "valid/unscored")
            det = " ".join(f"{t}={'ok' if s.direction.per_tag[t]['pass'] else 'MISS'}"
                           for t in s.direction.per_tag)
            score = det or "(style: not scored)"
        elif p.kind == "unsupported":
            res = "refused(WRONG)"
        else:
            res = f"invalid({p.errors[0] if p.errors else '?'})"
    else:  # unsupported gold: correct iff refused
        res = "refuse(CORRECT)" if p.kind == "unsupported" else f"{p.kind}(WRONG-false-support)"
    return {"res": res, "kind": p.kind, "out": out, "cost": cost, "sec": sec, "score": score,
            "finish": pv.get("finish_reason")}


def main() -> int:
    meta = {r.id: {"instruction": r.instruction, "gold_tags": r.gold_tags,
                   "is_supported": r.is_supported, "mixed": bool(r.mixed_prompt)}
            for r in load_rows("data/eval/smoke_rows.jsonl")}
    logs = _logmap()
    caches = {}
    for c in CONFIGS:
        path = f"data/eval/frontier_{c}.jsonl"
        caches[c] = {json.loads(l)["row_id"]: json.loads(l) for l in open(path)} if os.path.exists(path) else {}

    L = []
    w = L.append
    w("# Prompted-Frontier LUT Baseline — Pilot Results\n")
    w("**Task:** given a source image + a global color-grade instruction, emit a canonical "
      "17³ `.cube` LUT (4,913 rows) or the token `<unsupported>`. This is the runnable form of "
      "the prompted-frontier baseline (#8) in `docs/adr/0011-baseline-comparisons.md`.\n")
    w("**Setup:** models reached via the TrueFoundry OpenAI-compatible gateway "
      "(`configs/model_clients.yaml`), prompt `frontier_raw_cube_v2`, effort as noted. "
      "Eval fixtures are synthetic 32×32 patches upscaled to 256². Generated by "
      "`scripts/generate_frontier_luts.py`; scored by `eval/cube_parser.py` + "
      "`eval/frontier_scoring.py`; this file by `scripts/report_frontier_pilot.py`.\n")
    w("**Scoring:** L0 boundary (refuse vs attempt), L1/valid-`.cube` (complete canonical "
      "17³ parse), L4 direction (gold tag → Lab axis + sign, ≥1.5 Lab magnitude), L6 safety "
      "(monotone / non-clipping / smooth / non-identity). **L5 target fidelity is not "
      "evaluated** — the frozen eval rows carry no target LUTs. A LUT is a PASS iff it is a "
      "valid 17³ `.cube` AND direction-correct AND safe.\n")

    w("## Configurations\n")
    w("| config | slug | effort | price in/out ($/1M) | input tok |")
    w("|---|---|---|---|---|")
    for c in CONFIGS:
        pin, pout = PRICE[c]
        w(f"| {LABEL[c]} | `{SLUG[c]}` | {EFFORT[c]} | ${pin}/${pout} | ~{IN_EST[c]} |")
    w("")

    # main grid
    w("## Main grid — 5 single-attribute rows\n")
    w("Rows: r1 warmer · r2 cooler · r3 more_magenta · r4 more_green · r5 brighter.\n")
    w("| config | r1 | r2 | r3 | r4 | r5 | valid | cost | compute |")
    w("|---|---|---|---|---|---|---|---|---|")
    grand_cost = 0.0
    for c in CONFIGS:
        rat = _ratio(caches[c]); cells = []; nv = tc = tt = 0
        for rid in MAIN_ROWS:
            cell = _cell(c, rid, caches[c], rat, meta, logs)
            if cell is None:
                cells.append("—"); continue
            nv += int(cell["res"] == "PASS"); tc += cell["cost"]; tt += (cell["sec"] or 0)
            mark = "✅" if cell["res"] == "PASS" else ("⛔ref" if "refuse" in cell["res"] and "CORRECT" not in cell["res"] else "✗")
            cells.append(f"{mark} ${cell['cost']:.2f}" + (f" {cell['sec']}s" if cell["sec"] else ""))
        grand_cost += tc
        w(f"| {LABEL[c]} | " + " | ".join(cells) + f" | **{nv}/5** | ${tc:.2f} | {tt//60}m |")
    w(f"\n*(✅ = valid+correct LUT, ✗ = truncated/invalid, ⛔ref = refused. Grand cost of main "
      f"grid ≈ ${grand_cost:.2f}.)*\n")

    # detailed per-cell scores for the completed LUTs
    w("## Direction scores for completed LUTs\n")
    w("| config | row | Lab movement | dir/safety |")
    w("|---|---|---|---|")
    for c in CONFIGS:
        rat = _ratio(caches[c])
        for rid in MAIN_ROWS:
            d = caches[c].get(rid)
            if not d:
                continue
            p = parse_frontier_cube(d["text"])
            if p.kind != "raw_lut":
                continue
            s = score_lut(p.lut_abs, meta[rid]["gold_tags"]); b = s.behavior
            mv = (f"b*={b['temperature_delta_b']:+.1f} a*={b['tint_delta_a']:+.1f} "
                  f"L*={b['mean_l_delta']:+.1f} spread={b['contrast_l_spread_delta']:+.1f} "
                  f"chroma={b['chroma_delta']:+.1f}")
            w(f"| {LABEL[c]} | {rid.split('_')[-1]} {meta[rid]['gold_tags']} | {mv} | "
              f"{s.direction.status}/{s.safety.status} |")
    w("")

    # probe rows
    w("## Behavior-type probe (Opus / Sonnet / GPT-5.5 xhigh only)\n")
    for rid in PROBE_ROWS:
        m = meta[rid]
        kind = "MIXED UNSUPPORTED" if not m["is_supported"] else ("COMPOSITE" if len(m["gold_tags"]) == 2 else "NAMED STYLE")
        w(f"**{kind} — “{m['instruction']}”**  (tags: {m['gold_tags'] or 'unsupported'})\n")
        w("| config | result | detail | cost |")
        w("|---|---|---|---|")
        for c in ["opus_4_8", "sonnet_4_6", "gpt_5_5_xhigh"]:
            cell = _cell(c, rid, caches[c], _ratio(caches[c]), meta, logs)
            if cell is None:
                w(f"| {LABEL[c]} | — | (not run) | — |"); continue
            w(f"| {LABEL[c]} | {cell['res']} | {cell['score'] or cell['kind']} | ${cell['cost']:.2f} |")
        w("")

    # findings
    w("## Findings\n")
    for line in [
        "- **Color intent is never the failure — completion is.** Every LUT that parsed as a "
        "complete 17³ passed direction + safety. Failures are always truncation or refusal, "
        "never a wrong-direction LUT.",
        "- **Reliability (single-attribute rows):** Opus 4.8 4/5 > Sonnet 4.6 3/5 > "
        "GPT-5.5 xhigh 2/5 > GPT-5.5 medium 0/5 = Gemini 3.1 Pro 0/5.",
        "- **Effort gates GPT-5.5:** medium refuses/bails (0/5); xhigh engages (2/5 + a 93% "
        "near-miss) at ~15–20 min/row.",
        "- **Gemini 3.1 Pro is structurally blocked:** its route hard-caps output ~65K tokens "
        "(~2,000 rows), so a full ~100K-token LUT is impossible regardless of prompt.",
        "- **Composites work (3/3)** — all three models got both axes right simultaneously "
        "(warmer + softer-contrast).",
        "- **Named styles are the worst case (0/3)** — sepia truncated for all three; the extra "
        "decomposition reasoning eats the token budget *earlier* (Opus paid ~$3 for nothing).",
        "- **Boundary/refusal is the models' strength (3/3)** — all correctly refused the mixed "
        "trap (“warmer AND remove background”) for ~$0.01 in seconds, resisting the supported lure.",
        "- **The token ceiling is the dominant failure mode** across every supported behavior; "
        "refusal (near-zero output) is where frontier prompting shines.",
    ]:
        w(line)
    w("\n## Implication (ADR-0011)\n")
    w("Prompting frontier models *can* produce good global-color LUTs, so the project's win is "
      "not “only fine-tuning does the behavior.” It is **reliability + a tractable output budget "
      "(64 VQ tokens vs ~100K raw floats) + cost/latency/local**: a usable frontier LUT costs "
      "~$1.4–2.4 and 8–17 min and lands ≤80% of the time (best model), versus a small local "
      "model targeting 64 tokens, sub-second, ~$0.00x, deterministic.\n")

    w("## Caveats\n")
    for line in [
        "- Synthetic 32×32 fixtures with repeated prompts — probes behavior, not a real eval "
        "distribution.",
        "- Costs use provider **list prices**; the TrueFoundry gateway may bill differently, and "
        "prompt-cache discounts on the repeated system prompt could reduce the (small) input cost.",
        "- Output tokens are captured; **input tokens are a per-model probe estimate** (~0.8–1.9K, "
        "≈constant across rows). One GPT xhigh cell returned no usage → its output tokens are "
        "estimated from text length.",
        "- Per-cell timing is best-effort from run logs; some GPT-xhigh rows ran in a combined "
        "job and show no per-cell time.",
        "- Raw generations are preserved in `data/eval/frontier_*.jsonl`; re-score / regenerate "
        "via `python -m eval.run_frontier_eval` and `python -m scripts.report_frontier_pilot`.",
    ]:
        w(line)
    w("")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"wrote {OUT} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
