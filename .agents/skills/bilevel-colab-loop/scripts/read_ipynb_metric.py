"""Read the eval result back from the LOCAL notebook after a remote Colab run.

The Colab extension writes remote cell outputs into the local ``.ipynb`` file (once saved), so the
``METRIC=<accuracy>`` line printed by ``sft.bilevel_bridge`` is readable here as a plain file op — no
screenshots/OCR. This parses the notebook's cell outputs and reports a machine-usable verdict.

Detects, across ALL code-cell outputs (stream text + error tracebacks):
  * the LAST ``METRIC=<number>`` sentinel (what the loop optimizes),
  * the LAST ``bridge_summary`` / ``sft_summary`` / ``score_summary`` JSON (for the steps>0 guard),
  * failure tokens (``[bridge][ABORT]``, ``[sft][ABORT]``, ``[score][ABORT]``, ``Traceback``).

Emits ``status`` = ok | failed | no_metric so the loop never mistakes silence/scrollback for success.
``--min-mtime <epoch>`` is a staleness guard: fail if the notebook wasn't saved after that time (i.e.
outputs may be from a previous run — wait for autosave/Save, then re-read).

Pure stdlib (json). Usage:
    python read_ipynb_metric.py --notebook notebooks/sft_stage7_run.ipynb
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

METRIC_RE = re.compile(r"(?i)\bMETRIC\s*=\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
_SUMMARY_KEYS = ("bridge_summary", "sft_summary", "score_summary")
_FAIL_TOKENS = ("[bridge][ABORT]", "[sft][ABORT]", "[score][ABORT]", "Traceback (most recent call last)")


def _output_text(nb: dict) -> str:
    chunks: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        for out in cell.get("outputs", []) or []:
            if "text" in out:  # stream
                t = out["text"]
                chunks.append("".join(t) if isinstance(t, list) else str(t))
            data = out.get("data", {})
            if "text/plain" in data:  # execute_result / display_data
                t = data["text/plain"]
                chunks.append("".join(t) if isinstance(t, list) else str(t))
            if out.get("output_type") == "error":  # exception in a cell
                tb = out.get("traceback") or []
                chunks.append("\n".join(tb) if isinstance(tb, list) else str(tb))
                chunks.append(f"{out.get('ename', '')}: {out.get('evalue', '')}")
    return "\n".join(chunks)


def _last_summary(text: str) -> dict:
    found: dict = {}
    for line in text.splitlines():
        line = line.strip()
        for key in _SUMMARY_KEYS:
            if line.startswith('{"' + key + '"') or f'"{key}"' in line[:40]:
                try:
                    obj = json.loads(line).get(key, {})
                    if isinstance(obj, dict):
                        found = {"which": key, **obj}
                except (ValueError, TypeError):
                    pass
    return found


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--notebook", required=True)
    ap.add_argument("--min-mtime", type=float, default=None,
                    help="fail if the notebook mtime is <= this epoch (stale/unsaved outputs)")
    args = ap.parse_args(argv)

    nb_path = Path(args.notebook)
    if args.min_mtime is not None and nb_path.stat().st_mtime <= args.min_mtime:
        print(json.dumps({"status": "stale", "reason": "notebook not saved since the run; wait for "
                          "autosave/Save then re-read", "metric": None}))
        return 3

    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    text = _output_text(nb)
    summary = _last_summary(text)
    metrics = METRIC_RE.findall(text)
    fails = [tok for tok in _FAIL_TOKENS if tok in text]

    metric = float(metrics[-1]) if metrics else None
    rows_trained = summary.get("rows_trained") if summary else None
    if metric is not None and not fails and (rows_trained is None or rows_trained):
        status, reason = "ok", ""
    elif metric is None:
        status = "no_metric"
        reason = "no METRIC= line in outputs" + (f"; failure token(s): {fails}" if fails else "")
    else:
        status = "failed"
        reason = f"failure token(s): {fails}" if fails else f"rows_trained={rows_trained}"

    print(json.dumps({"status": status, "metric": metric, "reason": reason,
                      "summary": summary, "failure_tokens": fails}))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
