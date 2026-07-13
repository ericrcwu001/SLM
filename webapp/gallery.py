"""Persistent, shared gallery of generated grades for the web demo.

Each successful ``grade`` reuses the before/after previews and ``output.cube`` that the
pipeline already wrote to a run directory (no re-render), storing a downscaled, self-contained
copy under ``<root>/<id>/``.  The newest-first order lives in a single ``index.json`` written
atomically.  On Modal this ``root`` is a mounted Volume, so entries survive scale-to-zero; the
optional ``commit_hook`` lets the deploy layer force an immediate durable commit.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Callable

from PIL import Image

log = logging.getLogger("webapp.gallery")

# Downscale previews before persisting: a shared gallery must never store the raw upload, and
# small JPEGs keep the Volume bounded.  720px matches the grid card size with headroom.
_PREVIEW_MAX_EDGE = 720
_PREVIEW_QUALITY = 82


class GalleryStore:
    """A capped, newest-first store of grades on disk.

    Layout::

        <root>/index.json                 # ordered list[dict] of entry metadata, newest first
        <root>/<id>/before.jpg            # downscaled original
        <root>/<id>/after.jpg             # downscaled graded
        <root>/<id>/lut.cube              # the exported LUT
        <root>/<id>/meta.json            # the same dict stored in the index (provenance copy)
    """

    def __init__(
        self,
        root: Path,
        max_entries: int,
        commit_hook: Callable[[], None] | None = None,
        reload_hook: Callable[[], None] | None = None,
    ):
        self.root = Path(root)
        self.max_entries = max(1, int(max_entries))
        # Wired by the Modal deploy layer to Volume.commit for immediate cross-restart durability;
        # a no-op locally (a plain directory needs nothing beyond the filesystem write).
        self.commit_hook = commit_hook
        # Wired by the deploy layer (on the read-only edge node) to Volume.reload, so a list() picks
        # up grades another container just committed; a no-op locally / on the writer.
        self.reload_hook = reload_hook
        self.root.mkdir(parents=True, exist_ok=True)

    # -- paths ---------------------------------------------------------------
    @property
    def _index_path(self) -> Path:
        return self.root / "index.json"

    # -- reads ---------------------------------------------------------------
    def _read_index(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self._index_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return []
        except (ValueError, OSError) as exc:  # corrupt/half-written index must not break the app
            log.warning("gallery index unreadable, treating as empty: %s", exc)
            return []
        return data if isinstance(data, list) else []

    def list(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Return stored entries newest-first, optionally truncated to ``limit``."""
        if self.reload_hook is not None:
            try:
                self.reload_hook()   # sync the mounted Volume so another container's writes are visible
            except Exception as exc:  # freshness is best-effort; a stale read still returns entries
                log.warning("gallery reload_hook failed: %s", exc)
        entries = self._read_index()
        if limit is not None:
            entries = entries[: max(0, limit)]
        return entries

    # -- writes --------------------------------------------------------------
    def _write_index(self, entries: list[dict[str, Any]]) -> None:
        # Atomic replace so a concurrent /api/gallery read never sees a half-written file.
        tmp = self._index_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self._index_path)

    @staticmethod
    def _save_preview(src: Path, dest: Path) -> None:
        with Image.open(src) as img:
            preview = img.convert("RGB")
            preview.thumbnail((_PREVIEW_MAX_EDGE, _PREVIEW_MAX_EDGE), Image.Resampling.LANCZOS)
            preview.save(dest, format="JPEG", quality=_PREVIEW_QUALITY)

    def add_from_run(
        self,
        run_dir: str | Path,
        *,
        prompt: str,
        spec_text: str | None,
        quality: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Persist a gallery entry from the files a ``grade`` already wrote to ``run_dir``.

        Reuses ``user_image_original.png`` / ``user_image_graded.png`` / ``output.cube``; raises
        if any is missing so the caller can log and continue without a partial entry.
        """
        run_dir = Path(run_dir)
        before_src = run_dir / "user_image_original.png"
        after_src = run_dir / "user_image_graded.png"
        cube_src = run_dir / "output.cube"
        for required in (before_src, after_src, cube_src):
            if not required.is_file():
                raise FileNotFoundError(f"gallery source missing: {required}")

        # Reuse the run id (its parent dir name) so the gallery id lines up with server provenance.
        entry_id = run_dir.name
        entry_dir = self.root / entry_id
        # A retry with the same run id should overwrite cleanly rather than merge stale files.
        if entry_dir.exists():
            shutil.rmtree(entry_dir, ignore_errors=True)
        entry_dir.mkdir(parents=True, exist_ok=True)

        self._save_preview(before_src, entry_dir / "before.jpg")
        self._save_preview(after_src, entry_dir / "after.jpg")
        shutil.copyfile(cube_src, entry_dir / "lut.cube")

        entry = {
            "id": entry_id,
            "prompt": prompt,
            "spec_text": spec_text,
            "quality": quality,
            "created_at": time.time(),
            "before_url": f"/gallery/{entry_id}/before.jpg",
            "after_url": f"/gallery/{entry_id}/after.jpg",
            "cube_url": f"/gallery/{entry_id}/lut.cube",
        }
        (entry_dir / "meta.json").write_text(json.dumps(entry, ensure_ascii=False), encoding="utf-8")

        # Prepend (newest first), drop any prior copy of this id, then evict the tail.
        entries = [e for e in self._read_index() if e.get("id") != entry_id]
        entries.insert(0, entry)
        evicted, entries = entries[self.max_entries:], entries[: self.max_entries]
        for stale in evicted:
            stale_id = stale.get("id")
            if isinstance(stale_id, str) and stale_id:
                shutil.rmtree(self.root / stale_id, ignore_errors=True)
        self._write_index(entries)

        if self.commit_hook is not None:
            try:
                self.commit_hook()
            except Exception as exc:  # durability is best-effort; the write already landed on disk
                log.warning("gallery commit_hook failed: %s", exc)
        return entry
