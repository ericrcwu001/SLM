"""Offline tests for the Kaggle FiveK connector (pairing logic + graceful no-creds)."""

from data_pipeline.acquire.base import AcquireLimits
from data_pipeline.acquire.fivek_kaggle import FiveKKaggleConnector, pair_images


def test_pair_images_input_expertc_by_stem():
    files = [
        "input/a0001.jpg", "input/a0002.jpg", "input/a0003.jpg",
        "expertC/a0001.jpg", "expertC/a0002.jpg",
        "readme.txt",
    ]
    pairs = pair_images(files)
    stems = {p[0] for p in pairs}
    assert stems == {"a0001", "a0002"}  # a0003 has no expert-C match -> unpaired
    for stem, src, exp in pairs:
        assert "input" in src and "expertC" in exp


def test_pair_images_tiff16_c_layout():
    files = ["raw/photo_1.jpg", "tiff16_c/photo_1.jpg", "raw/photo_2.jpg", "tiff16_c/photo_2.jpg"]
    assert len(pair_images(files)) == 2


def test_pair_images_no_pairs_when_only_experts():
    # only expert renditions, no input/source set -> nothing to fit from
    assert pair_images(["expertC/x.jpg", "expertD/x.jpg"]) == []


def test_connector_verify_no_creds(monkeypatch):
    # force the no-credentials path deterministically (independent of ambient kaggle.json)
    monkeypatch.setattr("data_pipeline.acquire.fivek_kaggle._kaggle_creds", lambda: None)
    c = FiveKKaggleConnector()
    ok, note = c.verify()
    assert ok is False
    assert "credential" in note.lower() or "kaggle" in note.lower()


def test_connector_acquire_with_injected_download(tmp_path):
    # simulate an unzipped dataset on disk via the download hook
    (tmp_path / "fivek_kaggle" / "input").mkdir(parents=True)
    (tmp_path / "fivek_kaggle" / "expertC").mkdir(parents=True)
    for stem in ("p1", "p2"):
        (tmp_path / "fivek_kaggle" / "input" / f"{stem}.jpg").write_bytes(b"\xff\xd8sourcejpg")
        (tmp_path / "fivek_kaggle" / "expertC" / f"{stem}.jpg").write_bytes(b"\xff\xd8targetjpg")

    def fake_download(dest):
        return [str(p.relative_to(dest)) for p in dest.rglob("*") if p.is_file()]

    c = FiveKKaggleConnector(download_fn=fake_download, list_files_fn=lambda: ["x"])
    report = c.acquire(tmp_path, AcquireLimits(max_items=None))
    assert report.status == "ok"
    assert report.acquired == 2
    art = report.artifacts[0]
    assert art.kind == "image_pair" and art.expert_id == "c"
    assert art.derivation_method == "pair_fit"
