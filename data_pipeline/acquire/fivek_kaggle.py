"""MIT-Adobe FiveK connector via the Kaggle ``.jpg`` mirror ``weipengzhang/adobe-fivek``.

Unlike the HF parquet mirror, this dataset ships loose ``.jpg`` files, giving a usable
source -> expert-C pairing for the global-LUT pair fit (ADR 0003).

Download uses a **direct authenticated request** to Kaggle's dataset-download endpoint
(creds from ``~/.kaggle/kaggle.json`` or ``KAGGLE_USERNAME``/``KAGGLE_KEY``/``KAGGLE_API_KEY``)
rather than the kaggle client library, whose paginated endpoints are incompatible between the
1.6.x client and Kaggle's 2.x server. The full zip is ~28.5 GB, so we require free-disk
headroom, download the zip only, index it, extract a bounded input<->expert-C sample, then
delete the zip.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import zipfile
from pathlib import Path

import requests

from . import downloaders as dl
from .base import AcquireLimits, AcquireReport, RawArtifact, utcnow_iso

_SLUG = "weipengzhang/adobe-fivek"
_DOWNLOAD_URL = f"https://www.kaggle.com/api/v1/datasets/download/{_SLUG}"
_MIN_FREE_BYTES = 33 * 1024**3     # headroom for the ~28.5 GB zip
_MAX_EXTRACT_PAIRS = 1000          # cap extraction even under --full (zip still on disk)
_LICENSE = "MIT-Adobe FiveK (research use; via Kaggle weipengzhang/adobe-fivek .jpg mirror)"
_IMG_RE = re.compile(r"\.(jpg|jpeg|png)$", re.IGNORECASE)
# input/source vs expert-C role classification (checked most-specific first).
_INPUT_PAT = re.compile(r"(^|/)(input|raw|source|original|a)(/|_|-)", re.IGNORECASE)
_EXPERTC_PAT = re.compile(r"(expert[_-]?c|tiff16[_-]?c|(^|/)c(/|_|-)|target)", re.IGNORECASE)


def _kaggle_creds() -> tuple[str, str] | None:
    """Return (username, key) from env or ~/.kaggle/kaggle.json, else None."""
    user = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_KEY")
    if user and key:
        return user, key
    kj = Path.home() / ".kaggle" / "kaggle.json"
    if kj.exists():
        try:
            d = json.loads(kj.read_text())
            if d.get("username") and d.get("key"):
                return d["username"], d["key"]
        except Exception:  # noqa: BLE001
            return None
    return None


def _stem_key(path: str) -> str:
    stem = Path(path).stem
    stem = re.sub(r"[-_]?(expert)?[-_]?[a-eA-E]$", "", stem)
    return re.sub(r"[-_](input|raw|source|target|retouch|original)$", "", stem, flags=re.I)


def pair_images(file_list: list[str]) -> list[tuple[str, str, str]]:
    """Pair input/source images with their expert-C counterpart by stem (pure; unit-tested)."""
    images = [f for f in file_list if _IMG_RE.search(f)]
    inputs: dict[str, str] = {}
    experts: dict[str, str] = {}
    for f in images:
        is_expert = bool(_EXPERTC_PAT.search(f))
        is_input = bool(_INPUT_PAT.search(f)) and not is_expert
        key = _stem_key(f)
        if is_expert:
            experts.setdefault(key, f)
        elif is_input:
            inputs.setdefault(key, f)
    return sorted((k, inputs[k], experts[k]) for k in inputs if k in experts)


class FiveKKaggleConnector:
    source_pack_id = "fivek_expert_abcde"
    family = "fivek_derived"

    def __init__(self, slug: str = _SLUG, list_files_fn=None, download_fn=None,
                 creds: tuple[str, str] | None = None):
        self.slug = slug
        self.download_url = f"https://www.kaggle.com/api/v1/datasets/download/{slug}"
        self._list = list_files_fn        # test hook
        self._download = download_fn      # test hook: returns extracted rel paths
        self._creds = creds

    def _get_creds(self):
        return self._creds or _kaggle_creds()

    def verify(self) -> tuple[bool, str]:
        if self._list is not None:
            return (len(self._list()) > 0), "injected"
        creds = self._get_creds()
        if creds is None:
            return False, "no Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY)"
        try:
            r = requests.get(self.download_url, auth=creds, stream=True, timeout=60,
                             allow_redirects=True)
            ok = r.status_code < 400
            size = int(r.headers.get("Content-Length", 0) or 0)
            r.close()
            return ok, f"HTTP {r.status_code}, ~{size/1e9:.1f} GB"
        except Exception as e:  # noqa: BLE001
            return False, f"kaggle download endpoint unreachable: {e}"

    def _download_zip(self, creds, dest: Path) -> Path:
        zip_path = dest / "adobe-fivek.zip"
        part = zip_path.with_suffix(".zip.part")
        with requests.get(self.download_url, auth=creds, stream=True, timeout=120,
                          allow_redirects=True) as r:
            r.raise_for_status()
            with open(part, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    if chunk:
                        fh.write(chunk)
        part.replace(zip_path)
        return zip_path

    def acquire(self, raw_root, limits: AcquireLimits) -> AcquireReport:
        report = AcquireReport(source_pack_id=self.source_pack_id)
        dest = Path(raw_root) / "fivek_kaggle"
        dest.mkdir(parents=True, exist_ok=True)

        if self._download is not None:
            file_list = self._download(dest)
            pairs = pair_images(file_list)
            if limits.max_items is not None:
                pairs = pairs[: limits.max_items]
        else:
            target = min(limits.max_items or _MAX_EXTRACT_PAIRS, _MAX_EXTRACT_PAIRS)
            # resume: if enough pairs are already extracted, skip the ~28.5 GB re-download
            existing = [str(p.relative_to(dest)) for p in dest.rglob("*")
                        if p.is_file() and _IMG_RE.search(p.name)]
            existing_pairs = pair_images(existing)
            if len(existing_pairs) >= target:
                pairs = existing_pairs[:target]
                report.note = f"resumed {len(pairs)} pairs from disk (no re-download)"
                return self._emit_pairs(report, dest, pairs)
            creds = self._get_creds()
            if creds is None:
                return AcquireReport(self.source_pack_id, status="skipped",
                                     note="no Kaggle credentials")
            free = shutil.disk_usage(dest).free
            if free < _MIN_FREE_BYTES:
                return AcquireReport(self.source_pack_id, status="skipped",
                                     note=f"insufficient disk: {free/1e9:.1f} GB free, "
                                          f"need >= {_MIN_FREE_BYTES/1e9:.0f} GB")
            try:
                zip_path = self._download_zip(creds, dest)
            except Exception as e:  # noqa: BLE001
                return AcquireReport(self.source_pack_id, status="failed",
                                     note=f"kaggle download failed: {e}")
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    names = zf.namelist()
                    all_pairs = pair_images(names)
                    cap = min(limits.max_items or _MAX_EXTRACT_PAIRS, _MAX_EXTRACT_PAIRS)
                    pairs = all_pairs[:cap]
                    for _, src_rel, exp_rel in pairs:
                        zf.extract(src_rel, dest)
                        zf.extract(exp_rel, dest)
            finally:
                zip_path.unlink(missing_ok=True)   # reclaim ~28.5 GB
            if not pairs:
                return AcquireReport(
                    self.source_pack_id, status="partial",
                    note=(f"no input/expert-C pairs in zip ({len(names)} files); "
                          f"top dirs: {sorted({n.split('/')[0] for n in names})[:8]}"))
            if len(all_pairs) > len(pairs):
                report.note = f"extracted {len(pairs)}/{len(all_pairs)} pairs (disk cap {_MAX_EXTRACT_PAIRS})"

        return self._emit_pairs(report, dest, pairs)

    def _emit_pairs(self, report: AcquireReport, dest: Path, pairs) -> AcquireReport:
        ts = utcnow_iso()
        for stem, src_rel, exp_rel in pairs:
            report.attempted += 1
            spath, tpath = dest / src_rel, dest / exp_rel
            if not spath.exists() or not tpath.exists():
                report.skipped += 1
                continue
            report.artifacts.append(RawArtifact(
                kind="image_pair", source_pack_id=self.source_pack_id, family=self.family,
                declared_domain="srgb", license=_LICENSE, source_url=f"kaggle://{self.slug}",
                file_hash=dl.sha256_file(tpath), download_timestamp=ts,
                source_pair_id=f"fivek_{stem}", source_photo_id=stem, expert_id="c",
                source_image_path=str(spath), target_image_path=str(tpath),
                derivation_method="pair_fit",
            ))
            report.acquired += 1
        report.status = "ok" if report.acquired else "partial"
        return report
