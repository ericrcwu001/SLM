"""Tests for data_pipeline.staging (slm_stage: pack / stage / push).

Matches repo conventions: module-local `_`-helpers (no conftest/fixtures), `tmp_path` isolation,
call the library functions directly (not argv), and never touch the network — the GCS backend is
exercised via an injected fake `run_fn` that emulates `gcloud storage cp/ls/cat` against a local
"bucket" dir, and the credentials-absent branch is forced by monkeypatching `_gcs_available`.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from data_pipeline.errors import RequiresManualOptIn, StagingError
from data_pipeline.staging import backends, core

_QUIET = lambda *a, **k: None  # noqa: E731 (silence log_fn in tests)


# --- helpers ------------------------------------------------------------------------
def _make_corpus(root: Path) -> dict[str, bytes]:
    """Write a tiny nested artifact tree; return {relpath: bytes} for round-trip assertions."""
    files = {
        "luts/raw/fam/sub/a.jpg": b"jpeg-a-" + b"x" * 20,
        "luts/raw/fam/b.jpg": b"jpeg-b-" + b"y" * 5,
        "luts/canonical_residual/r0.npy": b"npy-r0-" + b"z" * 12,
        "data/raw_registry/rows.jsonl": b'{"id":1}\n{"id":2}\n',
        "data/splits/frozen/split.json": b'{"split_id":"abc"}',
        "tokenizer/final/manifest.json": b'{"tokenizer_version":"vq_test"}',
        # excluded cruft — must NOT be packed:
        "luts/raw/fam/junk.part": b"partial",
    }
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return {k: v for k, v in files.items() if not k.endswith(".part")}


def _pack_cfg(shard_max_bytes: int = 30) -> dict:
    return {"pack": {"include": core.DEFAULT_INCLUDE, "exclude": core.DEFAULT_EXCLUDE,
                     "shard_max_bytes": shard_max_bytes, "compression": "none"}}


class _FakeGcloud:
    """Emulates `gcloud storage cp|ls|cat` against a local bucket dir. Records call count."""

    def __init__(self, bucket_dir: Path):
        self.bucket_dir = bucket_dir
        self.calls = 0

    def _local(self, uri: str) -> Path:
        assert uri.startswith("gs://")
        return self.bucket_dir / uri[len("gs://"):]

    def __call__(self, cmd, capture_output=False, text=False, **kw):
        self.calls += 1
        assert cmd[:2] == ["gcloud", "storage"]
        verb, rest = cmd[2], cmd[3:]
        if verb == "cp":
            src, dst = rest
            if dst.startswith("gs://"):
                out = self._local(dst)
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, out)
            else:
                src_p = self._local(src)
                if not src_p.exists():
                    return SimpleNamespace(returncode=1, stdout="", stderr="not found")
                Path(dst).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src_p, dst)
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if verb == "ls":
            target = self._local(rest[-1])
            if not target.exists():
                return SimpleNamespace(returncode=1, stdout="", stderr="not found")
            if target.is_dir():
                lines = [rest[-1].rstrip("/") + "/" + c.name + ("/" if c.is_dir() else "")
                         for c in sorted(target.iterdir())]
                return SimpleNamespace(returncode=0, stdout="\n".join(lines) + "\n", stderr="")
            return SimpleNamespace(returncode=0, stdout=rest[-1] + "\n", stderr="")
        if verb == "cat":
            p = self._local(rest[-1])
            if not p.exists():
                return SimpleNamespace(returncode=1, stdout="", stderr="not found")
            return SimpleNamespace(returncode=0, stdout=p.read_text(encoding="utf-8"), stderr="")
        raise AssertionError(f"unexpected gcloud verb: {verb}")


# --- pack -----------------------------------------------------------------------------
def test_pack_local_creates_shards_and_manifest(tmp_path):
    root, durable = tmp_path / "art", tmp_path / "dur"
    expected = _make_corpus(root)
    rep = core.run_pack(str(root), str(durable), _pack_cfg(), log_fn=_QUIET)

    assert rep["status"] == "ok"
    assert rep["transferred"] == len(rep["shards"]) >= 1
    manifest = json.loads((durable / core.MANIFEST_NAME).read_text())
    assert manifest["staging_version"] == core.STAGING_VERSION
    total_members = sum(s["member_count"] for s in manifest["shards"])
    assert total_members == len(expected)  # the .part file was excluded
    for s in manifest["shards"]:
        shard = durable / s["name"]
        assert shard.exists()
        from data_pipeline.acquire.downloaders import sha256_file
        assert sha256_file(shard) == s["sha256"]


def test_pack_dry_run_writes_nothing(tmp_path):
    root, durable = tmp_path / "art", tmp_path / "dur"
    _make_corpus(root)
    rep = core.run_pack(str(root), str(durable), _pack_cfg(), dry_run=True, log_fn=_QUIET)
    assert rep["status"] == "ok"
    assert rep["note"].startswith("DRY-RUN")
    assert not durable.exists() or not any(durable.iterdir())


def test_pack_idempotent_skip(tmp_path):
    root, durable = tmp_path / "art", tmp_path / "dur"
    _make_corpus(root)
    core.run_pack(str(root), str(durable), _pack_cfg(), log_fn=_QUIET)
    rep2 = core.run_pack(str(root), str(durable), _pack_cfg(), log_fn=_QUIET)
    assert rep2["status"] == "skipped"
    assert "unchanged" in rep2["note"]


# --- stage ----------------------------------------------------------------------------
def test_pack_stage_round_trip(tmp_path):
    root, durable, local = tmp_path / "art", tmp_path / "dur", tmp_path / "local"
    expected = _make_corpus(root)
    core.run_pack(str(root), str(durable), _pack_cfg(), log_fn=_QUIET)

    cfg = {"stage": {"min_free_bytes": 0, "verify": "sha256"}}
    rep = core.run_stage(str(durable), str(local), cfg, log_fn=_QUIET)
    assert rep["status"] == "ok"
    assert rep["verified"] == rep["transferred"] >= 1
    for rel, data in expected.items():
        assert (local / rel).read_bytes() == data          # structure + bytes preserved

    # second stage is a resumable no-op
    rep2 = core.run_stage(str(durable), str(local), cfg, log_fn=_QUIET)
    assert rep2["transferred"] == 0
    assert rep2["skipped"] == rep2["attempted"] >= 1


def test_stage_disk_aware_skip(tmp_path):
    root, durable, local = tmp_path / "art", tmp_path / "dur", tmp_path / "local"
    _make_corpus(root)
    core.run_pack(str(root), str(durable), _pack_cfg(), log_fn=_QUIET)
    cfg = {"stage": {"min_free_bytes": 10**18}}             # absurd headroom -> skip
    rep = core.run_stage(str(durable), str(local), cfg, log_fn=_QUIET)
    assert rep["status"] == "skipped"
    assert "insufficient disk" in rep["note"]


def test_stage_missing_manifest_raises(tmp_path):
    durable, local = tmp_path / "empty", tmp_path / "local"
    durable.mkdir()
    with pytest.raises(StagingError):
        core.run_stage(str(durable), str(local), {}, log_fn=_QUIET)


# --- push -----------------------------------------------------------------------------
def test_push_local(tmp_path):
    durable, local = tmp_path / "dur", tmp_path / "local"
    (local / "tokenizer" / "final").mkdir(parents=True)
    (local / "tokenizer" / "final" / "manifest.json").write_bytes(b"{}")
    (local / "models").mkdir()
    (local / "models" / "ckpt.pt").write_bytes(b"weights")
    rep = core.run_push(str(local), str(durable), {}, log_fn=_QUIET)
    assert rep["status"] == "ok"
    assert rep["transferred"] == 2
    assert (durable / "outputs" / "models" / "ckpt.pt").read_bytes() == b"weights"
    assert (durable / core.PUSH_MANIFEST_NAME).exists()


# --- backend selection + GCS ----------------------------------------------------------
def test_backend_for_selects_type(tmp_path):
    assert isinstance(backends.backend_for("gs://bkt/prefix"), backends.GcsBackend)
    assert isinstance(backends.backend_for(str(tmp_path)), backends.LocalBackend)


def test_gcs_backend_round_trip_with_fake_cli(tmp_path):
    """pack -> stage over a gs:// durable root, driven by a fake gcloud CLI (no network)."""
    root, local = tmp_path / "art", tmp_path / "local"
    expected = _make_corpus(root)
    bucket_dir = tmp_path / "bucket"
    (bucket_dir / "bkt").mkdir(parents=True)                # so `ls gs://bkt` (verify) succeeds
    fake = _FakeGcloud(bucket_dir)

    rep = core.run_pack(str(root), "gs://bkt/prompt_to_lut", _pack_cfg(), run_fn=fake, log_fn=_QUIET)
    assert rep["status"] == "ok"
    assert (bucket_dir / "bkt" / "prompt_to_lut" / core.MANIFEST_NAME).exists()

    cfg = {"stage": {"min_free_bytes": 0}}
    core.run_stage("gs://bkt/prompt_to_lut", str(local), cfg, run_fn=fake, log_fn=_QUIET)
    for rel, data in expected.items():
        assert (local / rel).read_bytes() == data
    assert fake.calls > 0


def test_gcs_credentials_absent_gates(tmp_path, monkeypatch):
    root = tmp_path / "art"
    _make_corpus(root)
    monkeypatch.setattr(backends, "_gcs_available", lambda *a, **k: (False, "no creds"))

    # direct backend op raises the opt-in gate
    b = backends.GcsBackend("gs://bkt/p", run_fn=lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""))
    ok, note = b.verify()
    assert ok is False and "no creds" in note
    with pytest.raises(RequiresManualOptIn):
        b.put(root / "luts" / "raw" / "fam" / "b.jpg", "x.bin")

    # and the pack orchestration surfaces it too
    with pytest.raises(RequiresManualOptIn):
        core.run_pack(str(root), "gs://bkt/p", _pack_cfg(), log_fn=_QUIET)


# --- HuggingFace Hub backend ----------------------------------------------------------
class _FakeHfClient:
    """Emulates the huggingface_hub calls HfBackend needs, against a local 'hub' dir."""

    def __init__(self, hub: Path, has_token: bool = True):
        self.hub = hub
        self._has_token = has_token
        self.uploads = 0

    def _base(self, repo_id: str, repo_type: str) -> Path:
        return self.hub / repo_type / repo_id

    def token(self):
        return "fake-token" if self._has_token else None

    def repo_exists(self, repo_id, repo_type):
        return self._base(repo_id, repo_type).exists()

    def upload(self, local, rel, repo_id, repo_type):
        dst = self._base(repo_id, repo_type) / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(local, dst)
        self.uploads += 1

    def download(self, rel, dest, repo_id, repo_type):
        src = self._base(repo_id, repo_type) / rel
        if not src.exists():
            raise FileNotFoundError(rel)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dest)

    def exists(self, rel, repo_id, repo_type):
        return (self._base(repo_id, repo_type) / rel).exists()

    def list(self, repo_id, repo_type):
        base = self._base(repo_id, repo_type)
        return [p.relative_to(base).as_posix() for p in base.rglob("*") if p.is_file()] if base.exists() else []


def test_backend_for_selects_hf():
    b = backends.backend_for("hf://datasets/user/repo/prompt_to_lut")
    assert isinstance(b, backends.HfBackend)
    assert b.repo_type == "dataset" and b.repo_id == "user/repo" and b._prefix == "prompt_to_lut"


def test_hf_backend_round_trip_with_fake_client(tmp_path):
    root, local = tmp_path / "art", tmp_path / "local"
    expected = _make_corpus(root)
    hub = tmp_path / "hub"
    (hub / "dataset" / "user" / "repo").mkdir(parents=True)      # repo "exists"
    fake = _FakeHfClient(hub)

    uri = "hf://datasets/user/repo/prompt_to_lut"
    rep = core.run_pack(str(root), uri, _pack_cfg(), hf_client=fake, log_fn=_QUIET)
    assert rep["status"] == "ok" and fake.uploads > 0
    assert (hub / "dataset" / "user" / "repo" / "prompt_to_lut" / core.MANIFEST_NAME).exists()

    core.run_stage(uri, str(local), {"stage": {"min_free_bytes": 0}}, hf_client=fake, log_fn=_QUIET)
    for rel, data in expected.items():
        assert (local / rel).read_bytes() == data


def test_hf_credentials_absent_gates(tmp_path, monkeypatch):
    root = tmp_path / "art"
    _make_corpus(root)
    monkeypatch.setattr(backends, "_hf_available", lambda *a, **k: (False, "no HuggingFace token"))

    b = backends.HfBackend("hf://datasets/user/repo", client=_FakeHfClient(tmp_path / "hub", has_token=False))
    ok, note = b.verify()
    assert ok is False and "token" in note
    with pytest.raises(RequiresManualOptIn):
        b.put(root / "luts" / "raw" / "fam" / "b.jpg", "x.bin")
    with pytest.raises(RequiresManualOptIn):
        core.run_pack(str(root), "hf://datasets/user/repo", _pack_cfg(),
                      hf_client=_FakeHfClient(tmp_path / "hub2", has_token=False), log_fn=_QUIET)
