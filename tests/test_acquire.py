"""Acquisition tests — fully offline (fixtures + mocked sessions/downloads)."""

import re
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from data_pipeline.acquire import downloaders as dl
from data_pipeline.acquire import run_acquire as ra
from data_pipeline.acquire.base import AcquireLimits
from data_pipeline.acquire.freshluts import FreshLutsConnector
from data_pipeline.acquire.haldclut_rt import RawTherapeeHaldConnector
from data_pipeline.acquire.procedural_gen import ProceduralConnector
from data_pipeline.errors import RequiresManualOptIn
from data_pipeline.lut_ops import haldclut_to_lut, lut_to_hald
from eval.cube_io import identity_grid, serialize_cube


# --- fixtures ---------------------------------------------------------------------
def _make_hald_png(path: Path, level: int = 2):
    hald = (lut_to_hald(identity_grid(17), level=level) * 255).round().astype("uint8")
    Image.fromarray(hald, mode="RGB").save(path)


def _make_hald_zip(zip_path: Path, n: int = 3):
    tmp = zip_path.parent / "_pngs"
    tmp.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for i in range(n):
            p = tmp / f"film_{i}.png"
            _make_hald_png(p)
            zf.write(p, arcname=f"HaldCLUT/film_{i}.png")


# --- RawTherapee HaldCLUT ---------------------------------------------------------
def test_rawtherapee_haldclut_extract_and_decode(tmp_path, monkeypatch):
    zip_fixture = tmp_path / "HaldCLUT.zip"
    _make_hald_zip(zip_fixture, n=3)

    monkeypatch.setattr(dl, "http_head_ok", lambda *a, **k: (True, "ok"))
    monkeypatch.setattr(
        "data_pipeline.acquire.haldclut_rt.dl.http_download",
        lambda url, dest, **k: (Path(dest).parent.mkdir(parents=True, exist_ok=True)
                                or __import__("shutil").copy(zip_fixture, dest) or Path(dest)),
    )

    conn = RawTherapeeHaldConnector()
    ok, _ = conn.verify()
    assert ok
    report = conn.acquire(tmp_path / "raw", AcquireLimits(max_items=2))
    assert report.status == "ok"
    assert report.acquired == 2  # cap honored
    art = report.artifacts[0]
    assert art.kind == "lut_file" and art.declared_domain == "srgb"
    lut = haldclut_to_lut(np.asarray(Image.open(art.file_path)), 17)
    assert lut.shape == (17, 17, 17, 3)
    row = art.to_registry_row()
    assert row.source_family == "gmic_rawtherapee" and row.file_hash


def test_rawtherapee_resumable_skip(tmp_path, monkeypatch):
    zip_fixture = tmp_path / "HaldCLUT.zip"
    _make_hald_zip(zip_fixture, n=2)
    calls = {"n": 0}

    def fake_dl(url, dest, **k):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and dest.stat().st_size > 0:
            return dest  # resumable skip
        calls["n"] += 1
        __import__("shutil").copy(zip_fixture, dest)
        return dest

    monkeypatch.setattr("data_pipeline.acquire.haldclut_rt.dl.http_download", fake_dl)
    conn = RawTherapeeHaldConnector()
    conn.acquire(tmp_path / "raw", AcquireLimits(max_items=2))
    conn.acquire(tmp_path / "raw", AcquireLimits(max_items=2))
    assert calls["n"] == 1  # downloaded once, second run skipped


# --- procedural connector ---------------------------------------------------------
def test_procedural_connector(tmp_path):
    report = ProceduralConnector(magnitudes=(1.0,)).acquire(tmp_path / "raw", AcquireLimits(max_items=None))
    assert report.status == "ok" and report.acquired == 22
    assert all(a.procedural_filler for a in report.artifacts)
    row = report.artifacts[0].to_registry_row()
    assert row.procedural_filler is True and row.structured_tags


# --- ON1 local pack ---------------------------------------------------------------
def test_on1_local_connector(tmp_path):
    from data_pipeline.acquire.on1_local import ON1Connector

    src = tmp_path / "ON1_All_LUTs"
    body = serialize_cube(identity_grid(17))
    cine = src / "ON1 Cinematic LUTs" / "For Other Programs" / "Cube Files"
    cine.mkdir(parents=True)
    (cine / "c1.cube").write_bytes(
        b"# Preset: Cine One\n# Color profile: /x/AdobeRGB1998.icc\n" + body)
    port = src / "ON1 Portrait LUTs" / "For Other Programs" / "Cube Files"
    port.mkdir(parents=True)
    (port / "p1.cube").write_bytes(b"# Preset: Port One\n" + body)  # no profile -> sRGB

    conn = ON1Connector(src_dir=str(src))
    ok, _ = conn.verify()
    assert ok
    report = conn.acquire(tmp_path / "raw", AcquireLimits(max_items=None))
    assert report.status == "ok" and report.acquired == 2
    assert all(a.family == "smaller_public_packs" for a in report.artifacts)
    assert all(a.derivation_method == "cube" for a in report.artifacts)

    by_dom = {a.declared_domain for a in report.artifacts}
    assert by_dom == {"adobe_rgb", "srgb"}
    cine_art = next(a for a in report.artifacts if "cinematic" in a.gold_tags)
    assert cine_art.declared_domain == "adobe_rgb"
    port_art = next(a for a in report.artifacts if "portrait" in a.gold_tags)
    assert port_art.declared_domain == "srgb"
    assert Path(cine_art.file_path).exists()

    # resumable: a second pass copies nothing
    report2 = conn.acquire(tmp_path / "raw", AcquireLimits(max_items=None))
    assert report2.acquired == 0 and report2.skipped == 2


# --- FreshLUTs (mocked Devise flow) -----------------------------------------------
_CUBE_BYTES = serialize_cube(identity_grid(17))


class _FakeResp:
    def __init__(self, text="", status=200, content=None, headers=None):
        self.text = text
        self.status_code = status
        self.url = ""
        self.content = content if content is not None else text.encode()
        self.headers = headers or {}


class _FakeSession:
    SIGNIN = '<meta name="csrf-token" content="tok123"><form action="/users/sign_in">user[password]</form>'
    SIGNEDIN = '<a href="/users/sign_out">Sign out</a>'
    REAL_IDS = ("101", "102")   # real LUT pages carry the /downloadlut form; others are gaps

    def _lut_page(self, lid):
        return (f'<form action="/downloadlut?lutid={lid}&userid=1" method="post">'
                '<input type="hidden" name="authenticity_token" value="tok123"/>'
                '<input type="submit" value="Download LUT"/></form>'
                '<span>CC0 Creative Commons</span>')

    def get(self, url, **k):
        if "/users/sign_in" in url or "/users/sign_up" in url:
            return _FakeResp(self.SIGNIN)
        m = re.search(r"/luts/(\d+)", url)
        if m:
            lid = m.group(1)
            return _FakeResp(self._lut_page(lid) if lid in self.REAL_IDS else "<div>no lut</div>")
        return _FakeResp("")

    def post(self, url, data=None, **k):
        if "/downloadlut" in url:
            lid = re.search(r"lutid=(\d+)", url)
            lid = lid.group(1) if lid else "x"
            return _FakeResp(content=_CUBE_BYTES,
                             headers={"Content-Type": "application/octet-stream",
                                      "Content-Disposition": f'attachment; filename="lut_{lid}.cube"'})
        return _FakeResp(self.SIGNEDIN)


def test_freshluts_missing_creds_raises(tmp_path):
    conn = FreshLutsConnector(session=_FakeSession(), email=None, password=None, min_delay_s=0, jitter_s=0, scan_max_id=200)
    ok, note = conn.verify()
    assert not ok
    with pytest.raises(RequiresManualOptIn):
        conn.acquire(tmp_path / "raw", AcquireLimits(max_items=5))


def test_freshluts_signin_crawl_download(tmp_path):
    conn = FreshLutsConnector(session=_FakeSession(), email="a@b.co", password="pw", min_delay_s=0, jitter_s=0, scan_max_id=200)
    report = conn.acquire(tmp_path / "raw", AcquireLimits(max_items=None, rate_limit_s=0.0))
    assert report.status == "ok"
    assert report.acquired == 2
    # real .cube bytes written + validated
    got = (tmp_path / "raw" / "fresh_luts" / "freshluts" / "101.cube").read_bytes()
    assert b"LUT_3D_SIZE" in got
    assert any(a.extra.get("title", "").endswith(".cube") for a in report.artifacts)


def test_freshluts_resumable_skip(tmp_path):
    pre = tmp_path / "raw" / "fresh_luts" / "freshluts" / "101.cube"
    pre.parent.mkdir(parents=True, exist_ok=True)
    pre.write_bytes(_CUBE_BYTES)
    conn = FreshLutsConnector(session=_FakeSession(), email="a@b.co", password="pw", min_delay_s=0, jitter_s=0, scan_max_id=200)
    report = conn.acquire(tmp_path / "raw", AcquireLimits(max_items=None))
    assert report.skipped >= 1


# --- orchestrator (offline: procedural only) --------------------------------------
def test_run_acquire_offline_procedural(tmp_path):
    summary = ra.run_acquire(out_root=str(tmp_path), only=["procedural_fillers_v1"])
    assert summary["total_acquired"] > 0
    assert (tmp_path / "data" / "raw_registry" / "acquisition_report.json").exists()
    assert (tmp_path / "data" / "raw_registry" / "provenance.jsonl").exists()
    proc_src = [s for s in summary["sources"] if s["source_pack_id"] == "procedural_fillers_v1"]
    assert proc_src and proc_src[0]["status"] == "ok"


def test_run_acquire_disabled_source_skipped(tmp_path):
    summary = ra.run_acquire(out_root=str(tmp_path), only=["public_lut_packs_misc"])
    s = summary["sources"][0]
    assert s["status"] == "skipped"
