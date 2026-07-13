from pathlib import Path

import pytest
from PIL import Image

from webapp.gallery import GalleryStore


def _make_run(tmp_path: Path, run_id: str, color=(120, 60, 30)) -> Path:
    """Create a fake run dir with the three files a grade leaves behind."""
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (48, 32), (10, 20, 30)).save(run_dir / "user_image_original.png")
    Image.new("RGB", (48, 32), color).save(run_dir / "user_image_graded.png")
    (run_dir / "output.cube").write_text("LUT_3D_SIZE 2\n0 0 0\n1 1 1\n", encoding="utf-8")
    return run_dir


def test_add_writes_previews_cube_and_metadata(tmp_path):
    store = GalleryStore(tmp_path / "gallery", max_entries=10)
    entry = store.add_from_run(
        _make_run(tmp_path, "aaa"),
        prompt="warm cinematic look",
        spec_text="route=grade | warmer=+2.0",
        quality={"behavioral_fidelity": 0.9, "collapsed": False, "fell_back_greedy": False},
    )
    assert entry["id"] == "aaa"
    assert entry["before_url"] == "/gallery/aaa/before.jpg"
    assert entry["after_url"] == "/gallery/aaa/after.jpg"
    assert entry["cube_url"] == "/gallery/aaa/lut.cube"
    assert isinstance(entry["created_at"], float)
    for name in ("before.jpg", "after.jpg", "lut.cube", "meta.json"):
        assert (store.root / "aaa" / name).is_file()


def test_list_is_newest_first(tmp_path):
    store = GalleryStore(tmp_path / "gallery", max_entries=10)
    store.add_from_run(_make_run(tmp_path, "aaa"), prompt="first", spec_text=None, quality=None)
    store.add_from_run(_make_run(tmp_path, "bbb"), prompt="second", spec_text=None, quality=None)
    assert [e["id"] for e in store.list()] == ["bbb", "aaa"]


def test_eviction_drops_and_rmtrees_oldest(tmp_path):
    store = GalleryStore(tmp_path / "gallery", max_entries=2)
    for rid in ("a", "b", "c"):
        store.add_from_run(_make_run(tmp_path, rid), prompt=rid, spec_text=None, quality=None)
    assert [e["id"] for e in store.list()] == ["c", "b"]
    assert not (store.root / "a").exists()  # oldest fully removed from disk
    assert (store.root / "b").is_dir() and (store.root / "c").is_dir()


def test_limit_truncates(tmp_path):
    store = GalleryStore(tmp_path / "gallery", max_entries=10)
    for rid in ("a", "b", "c"):
        store.add_from_run(_make_run(tmp_path, rid), prompt=rid, spec_text=None, quality=None)
    assert len(store.list(limit=2)) == 2
    assert store.list(limit=0) == []


def test_readding_same_id_replaces_without_duplicates(tmp_path):
    store = GalleryStore(tmp_path / "gallery", max_entries=10)
    store.add_from_run(_make_run(tmp_path, "dup"), prompt="v1", spec_text=None, quality=None)
    store.add_from_run(_make_run(tmp_path, "dup"), prompt="v2", spec_text=None, quality=None)
    entries = store.list()
    assert [e["id"] for e in entries] == ["dup"]
    assert entries[0]["prompt"] == "v2"


def test_list_tolerates_corrupt_index(tmp_path):
    store = GalleryStore(tmp_path / "gallery", max_entries=5)
    store.add_from_run(_make_run(tmp_path, "x"), prompt="x", spec_text=None, quality=None)
    (store.root / "index.json").write_text("{ not valid json", encoding="utf-8")
    assert store.list() == []


def test_list_on_empty_store(tmp_path):
    assert GalleryStore(tmp_path / "gallery", max_entries=5).list() == []


def test_missing_source_files_raise(tmp_path):
    store = GalleryStore(tmp_path / "gallery", max_entries=5)
    empty = tmp_path / "runs" / "empty"
    empty.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        store.add_from_run(empty, prompt="p", spec_text=None, quality=None)


def test_commit_hook_is_called_after_write(tmp_path):
    calls = []
    store = GalleryStore(tmp_path / "gallery", max_entries=5, commit_hook=lambda: calls.append(1))
    store.add_from_run(_make_run(tmp_path, "z"), prompt="z", spec_text=None, quality=None)
    assert calls == [1]


def test_commit_hook_failure_does_not_break_write(tmp_path):
    def boom():
        raise RuntimeError("volume unavailable")

    store = GalleryStore(tmp_path / "gallery", max_entries=5, commit_hook=boom)
    entry = store.add_from_run(_make_run(tmp_path, "z"), prompt="z", spec_text=None, quality=None)
    assert entry["id"] == "z"
    assert [e["id"] for e in store.list()] == ["z"]  # the write still landed
