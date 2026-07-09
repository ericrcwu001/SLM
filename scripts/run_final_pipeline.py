#!/usr/bin/env python3
"""Final full-corpus pipeline run (network-free).

Runs Stages 4->5->6->9->11 over the merged raw registry (--no-acquire) with the improved
smooth-fill derivation + tuned gates. Writes a sentinel on completion/failure so an external
monitor has a reliable terminal signal. Detached-friendly (nohup / PPID 1).
"""

from __future__ import annotations

import json
import pathlib
import sys
import traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DONE = ROOT / "data" / "raw_registry" / ".pipeline_done"
FAIL = ROOT / "data" / "raw_registry" / ".pipeline_failed"


def main() -> int:
    from data_pipeline.run_pipeline import run_pipeline

    DONE.unlink(missing_ok=True)
    FAIL.unlink(missing_ok=True)
    cfg = str(ROOT / "data_pipeline" / "configs" / "overnight_full.yaml")
    try:
        summary = run_pipeline(cfg, str(ROOT), acquire=False)
    except Exception as e:  # noqa: BLE001
        FAIL.write_text(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")
        print(f"[pipeline] FAILED {type(e).__name__}: {e}", flush=True)
        return 1
    DONE.write_text(json.dumps(summary.get("stages", {}), indent=2))
    print("[pipeline] SENTINEL written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
