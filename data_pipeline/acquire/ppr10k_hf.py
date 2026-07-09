"""PPR10K connector via the HuggingFace mirror ``JarvisArt/MMArt-PPR10k``.

Built on PPR10K; per-sample dir carries ``before.jpg``, ``processed.jpg``, ``config.xmp``,
``config.lua`` (Apache-2.0), plus the MMArt natural-language edit instructions in
``user_want_short`` / ``user_want_middle`` / ``user_want_long``. We download a bounded set of
samples and yield ``image_pair`` artifacts (source=before, target=processed, xmp=config.xmp)
for Stage-4 XMP-gate + pair-fit, and carry the authored instruction (short/middle as the
concise phrasing, long as the natural one) so the teacher is SKIPPED for these rows.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import downloaders as dl
from .base import AcquireLimits, AcquireReport, RawArtifact, utcnow_iso

_REPO = "JarvisArt/MMArt-PPR10k"
_LICENSE = "Apache-2.0 (JarvisArt/MMArt-PPR10k, built on PPR10K)"
_BEFORE_RE = re.compile(r"(^|/)before\.(jpg|jpeg|png)$", re.IGNORECASE)
# MMArt per-sample instruction files (extension unspecified upstream; match by stem).
_WANT_RE = {
    "short": re.compile(r"(^|/)user_want_short\b", re.IGNORECASE),
    "middle": re.compile(r"(^|/)user_want_middle\b", re.IGNORECASE),
    "long": re.compile(r"(^|/)user_want_long\b", re.IGNORECASE),
}


def _read_instruction_text(path) -> str | None:
    """Read an MMArt instruction file (plain text, or JSON with a text-ish field)."""
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:  # noqa: BLE001
        return None
    if not raw:
        return None
    if raw[0] in "{[":
        try:
            obj = json.loads(raw)
        except Exception:  # noqa: BLE001
            obj = None
        if isinstance(obj, dict):
            for k in ("text", "instruction", "prompt", "user_want", "content"):
                if obj.get(k):
                    return str(obj[k]).strip()
            for v in obj.values():
                if isinstance(v, str) and v.strip():
                    return v.strip()
        elif isinstance(obj, list) and obj and isinstance(obj[0], str):
            return obj[0].strip()
    return raw


class PPR10KHFConnector:
    source_pack_id = "ppr10k_expert_abc"
    family = "ppr10k_derived"

    def __init__(self, repo_id: str = _REPO, list_files_fn=None, download_fn=None):
        self.repo_id = repo_id
        self._list = list_files_fn or (lambda: dl.hf_list_files(self.repo_id, "dataset"))
        self._download = download_fn or (
            lambda fn, root: dl.hf_download_file(self.repo_id, fn, root, "dataset")
        )

    def verify(self) -> tuple[bool, str]:
        try:
            files = self._list()
            return (len(files) > 0), f"{len(files)} files listed"
        except Exception as e:  # noqa: BLE001
            return False, f"hf list failed: {e}"

    def acquire(self, raw_root, limits: AcquireLimits) -> AcquireReport:
        report = AcquireReport(source_pack_id=self.source_pack_id)
        root = Path(raw_root) / "ppr10k"
        try:
            files = self._list()
        except Exception as e:  # noqa: BLE001
            report.status = "failed"
            report.note = f"hf list failed: {e}"
            return report

        # group by sample dir = parent of a before.* file
        sample_dirs: dict[str, str] = {}
        for f in files:
            if _BEFORE_RE.search(f):
                sample_dirs[str(Path(f).parent)] = f
        ordered = sorted(sample_dirs.keys())
        if limits.max_items is not None:
            ordered = ordered[: limits.max_items]
        if not ordered:
            report.status = "partial"
            report.note = "no before.* samples found in repo listing"
            return report

        by_dir = {}
        for f in files:
            by_dir.setdefault(str(Path(f).parent), []).append(f)

        ts = utcnow_iso()
        for sdir in ordered:
            report.attempted += 1
            members = by_dir.get(sdir, [])
            before = next((m for m in members if _BEFORE_RE.search(m)), None)
            processed = next((m for m in members if re.search(r"processed\.(jpg|jpeg|png)$", m, re.I)), None)
            xmp = next((m for m in members if m.lower().endswith("config.xmp")), None)
            if not before or not processed:
                report.skipped += 1
                continue
            try:
                bpath = self._download(before, root)
                ppath = self._download(processed, root)
                xpath = self._download(xmp, root) if xmp else None
                # MMArt authored instructions (best-effort; failure to fetch text must not drop
                # the image pair — the teacher can still fill an un-authored row later).
                wants: dict[str, str] = {}
                for length, rx in _WANT_RE.items():
                    member = next((m for m in members if rx.search(m)), None)
                    if not member:
                        continue
                    try:
                        text = _read_instruction_text(self._download(member, root))
                    except Exception:  # noqa: BLE001
                        text = None
                    if text:
                        wants[length] = text
                concise = wants.get("short") or wants.get("middle") or wants.get("long")
                natural = wants.get("long") or wants.get("middle")
                if natural == concise:
                    natural = None
                report.artifacts.append(RawArtifact(
                    kind="image_pair", source_pack_id=self.source_pack_id, family=self.family,
                    declared_domain="srgb", license=_LICENSE, source_url=f"hf://{self.repo_id}",
                    file_hash=dl.sha256_file(ppath), download_timestamp=ts,
                    source_pair_id=sdir.replace("/", "_"), group_id=Path(sdir).name,
                    source_image_path=str(bpath), target_image_path=str(ppath),
                    xmp_path=(str(xpath) if xpath else None), derivation_method="pair_fit",
                    authored_instruction=concise, authored_instruction_natural=natural,
                    authored_instruction_source=(f"mmart_ppr10k:{'+'.join(sorted(wants))}"
                                                 if wants else None),
                ))
                report.acquired += 1
            except Exception:  # noqa: BLE001
                report.failed += 1
        report.status = "ok" if report.acquired else "partial"
        return report
