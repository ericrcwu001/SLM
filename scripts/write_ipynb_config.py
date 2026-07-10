"""Inject a bilevel candidate into the LOCAL notebook's config cell.

The Colab VS Code/Cursor extension keeps the ``.ipynb`` FILE local while executing cells on the
remote A100. So Codex edits the local file here; on the next "Run All" the edited cell runs remotely
and writes ``candidate.json`` onto the VM, which the eval cell (calling ``sft.bilevel_bridge``) reads.

Target cell = the code cell tagged ``candidate-config`` in ``metadata.tags`` OR whose first line is
``# @candidate-config``. Fails loudly if it can't find exactly that cell (never guesses a cell).

Pure stdlib (json only); prints a JSON status line. Usage:
    python write_ipynb_config.py --notebook notebooks/sft_stage7_run.ipynb \
        --params '{"lora_r": 24, "learning_rate_lora": 0.0003}'
    python write_ipynb_config.py --notebook nb.ipynb --params-file proposal.json  # reads .params too
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MARKER = "# @candidate-config"
VM_CANDIDATE_PATH = "/content/SLM/candidate.json"


def _load_params(args) -> dict:
    if args.params_file:
        raw = json.loads(Path(args.params_file).read_text(encoding="utf-8"))
        params = raw.get("params", raw) if isinstance(raw, dict) else raw
    else:
        params = json.loads(args.params)
    if not isinstance(params, dict):
        raise ValueError("params must be a JSON object")
    return params


def _is_target(cell: dict, tag: str) -> bool:
    if cell.get("cell_type") != "code":
        return False
    if tag in (cell.get("metadata", {}).get("tags") or []):
        return True
    src = cell.get("source") or []
    first = (src[0] if isinstance(src, list) and src else src if isinstance(src, str) else "")
    return first.strip().startswith(MARKER)


def _cell_source(params: dict) -> list[str]:
    payload = json.dumps(params, indent=2)
    body = (
        f"{MARKER}  (auto-written by the improve loop; do not hand-edit)\n"
        "import json, pathlib\n"
        f"_CANDIDATE = {payload!r}\n"
        f'pathlib.Path("{VM_CANDIDATE_PATH}").write_text(_CANDIDATE)\n'
        'print("candidate.json written:", _CANDIDATE)\n'
    )
    return body.splitlines(keepends=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--notebook", required=True)
    ap.add_argument("--params", help="candidate params as a JSON object string")
    ap.add_argument("--params-file", help="JSON file (flat params or {'params': {...}})")
    ap.add_argument("--tag", default="candidate-config")
    args = ap.parse_args(argv)
    if not args.params and not args.params_file:
        print(json.dumps({"error": "provide --params or --params-file"})); return 2

    params = _load_params(args)
    nb_path = Path(args.notebook)
    nb = json.loads(nb_path.read_text(encoding="utf-8"))

    hits = [c for c in nb.get("cells", []) if _is_target(c, args.tag)]
    if len(hits) != 1:
        print(json.dumps({"error": f"expected exactly 1 cell tagged {args.tag!r} or starting with "
                                   f"{MARKER!r}, found {len(hits)}"}))
        return 1
    cell = hits[0]
    cell["source"] = _cell_source(params)
    cell["outputs"] = []
    cell["execution_count"] = None

    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"ok": True, "notebook": str(nb_path), "params": params,
                      "vm_candidate_path": VM_CANDIDATE_PATH}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
