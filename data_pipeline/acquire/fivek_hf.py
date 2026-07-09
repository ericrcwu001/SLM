"""MIT-Adobe FiveK connector via a HuggingFace WebP mirror (default ``logasja/mit-adobe-fivek``).

The original dataset is >1 TB; the WebP mirror is far smaller. Layout varies across mirrors,
so we discover it: list files, keep images, classify each into a "source" vs "expert" role by
directory patterns, and pair by photo stem. Bounded to ``max_items`` pairs. If the layout is
unrecognized the connector reports ``partial`` with sample paths rather than crashing.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from . import downloaders as dl
from .base import AcquireLimits, AcquireReport, RawArtifact, utcnow_iso

_REPO = "logasja/mit-adobe-fivek"
_LICENSE = "MIT-Adobe FiveK research license (via HF WebP mirror)"
_IMG_RE = re.compile(r"\.(webp|jpg|jpeg|png|tif|tiff)$", re.IGNORECASE)
_SOURCE_PAT = re.compile(r"(raw|input|source|/a/|dng)", re.IGNORECASE)
_EXPERT_PAT = re.compile(r"(expert|tiff16_?[a-e]|/[b-e]/|retouch|target)", re.IGNORECASE)


class FiveKHFConnector:
    source_pack_id = "fivek_expert_abcde"
    family = "fivek_derived"

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

    @staticmethod
    def _stem_key(path: str) -> str:
        # strip trailing expert/role tokens to align source with its expert target
        stem = Path(path).stem
        stem = re.sub(r"[-_](expert)?[a-eA-E]$", "", stem)
        return re.sub(r"[-_](raw|input|source|target|retouch)$", "", stem, flags=re.I)

    def acquire(self, raw_root, limits: AcquireLimits) -> AcquireReport:
        report = AcquireReport(source_pack_id=self.source_pack_id)
        root = Path(raw_root) / "fivek"
        try:
            all_files = self._list()
        except Exception as e:  # noqa: BLE001
            report.status = "failed"
            report.note = f"hf list failed: {e}"
            return report
        files = [f for f in all_files if _IMG_RE.search(f)]
        if not files:
            if any(f.endswith(".parquet") for f in all_files):
                report.status = "partial"
                report.note = ("parquet-packed FiveK mirror (HF datasets, experts a-e); loose-image "
                               "pairing not wired. Extend with `datasets` to unpack raw->expert pairs.")
            else:
                report.status = "partial"
                report.note = "no image files in repo listing"
            return report

        groups: dict[str, dict[str, str]] = defaultdict(dict)
        for f in files:
            role = "expert" if _EXPERT_PAT.search(f) else ("source" if _SOURCE_PAT.search(f) else None)
            if role is None:
                continue
            groups[self._stem_key(f)].setdefault(role, f)

        pairs = [(k, v["source"], v["expert"]) for k, v in groups.items()
                 if "source" in v and "expert" in v]
        pairs.sort()
        if not pairs:
            report.status = "partial"
            report.note = f"layout_unrecognized; sample files: {files[:5]}"
            return report
        if limits.max_items is not None:
            pairs = pairs[: limits.max_items]

        ts = utcnow_iso()
        for stem, src_f, exp_f in pairs:
            report.attempted += 1
            try:
                spath = self._download(src_f, root)
                tpath = self._download(exp_f, root)
                report.artifacts.append(RawArtifact(
                    kind="image_pair", source_pack_id=self.source_pack_id, family=self.family,
                    declared_domain="srgb", license=_LICENSE, source_url=f"hf://{self.repo_id}",
                    file_hash=dl.sha256_file(tpath), download_timestamp=ts,
                    source_pair_id=f"fivek_{stem}", source_photo_id=stem, expert_id="mixed",
                    source_image_path=str(spath), target_image_path=str(tpath),
                    derivation_method="pair_fit",
                ))
                report.acquired += 1
            except Exception:  # noqa: BLE001
                report.failed += 1
        report.status = "ok" if report.acquired else "partial"
        return report
