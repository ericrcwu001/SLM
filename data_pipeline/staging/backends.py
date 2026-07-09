"""Durable-root backends for ``slm_stage`` (ADR-0018 durable root; ADR-0019 GCS).

A durable root is where packed shards + ``stage_manifest.json`` live between sessions. It is
one of:

* a **local path** — a plain directory, or a mounted Google Drive path such as
  ``/content/drive/MyDrive/prompt_to_lut`` (Drive is just a FUSE filesystem, so the same
  ``LocalBackend`` handles it);
* a **GCS bucket** — ``gs://<bucket>/<prefix>`` accessed through the ``gcloud storage`` CLI
  (``GcsBackend``). The CLI is shelled via an injectable ``run_fn`` so tests never touch the
  network, and missing credentials raise :class:`RequiresManualOptIn` (never a silent no-op).

All paths handed to a backend are POSIX-style **relative** keys under the durable root.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path

from ..errors import RequiresManualOptIn, StagingError

# `gcloud storage` and `gsutil` share the cp/ls/cat verbs; gcloud is preferred (newer, faster).
_GCLOUD = ("gcloud", "storage")


class DurableBackend(ABC):
    """Minimal object-store interface used by pack/stage/push. Keys are relative POSIX paths."""

    #: human-readable location, for logs/manifests
    uri: str

    @abstractmethod
    def put(self, local: Path, rel: str) -> None:
        """Upload a local file to ``rel`` under the durable root (create parents, atomic)."""

    @abstractmethod
    def get(self, rel: str, local: Path) -> None:
        """Download ``rel`` to the local file ``local`` (create parents, atomic)."""

    @abstractmethod
    def exists(self, rel: str) -> bool:
        ...

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """List relative keys directly under ``prefix`` (non-recursive)."""

    @abstractmethod
    def read_text(self, rel: str) -> str:
        ...

    @abstractmethod
    def write_text(self, rel: str, text: str) -> None:
        ...

    def verify(self) -> tuple[bool, str]:
        """Cheap reachability/credential check. Returns ``(ok, note)`` — never raises."""
        return True, f"{self.uri} (assumed reachable)"


class LocalBackend(DurableBackend):
    """Durable root on a local filesystem (plain dir or a mounted Google Drive path)."""

    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.uri = str(self.root)

    def _abs(self, rel: str) -> Path:
        return self.root / rel

    @staticmethod
    def _atomic_copy(src: Path, dst: Path) -> None:
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        shutil.copy2(src, tmp)
        tmp.replace(dst)  # atomic within a filesystem

    def put(self, local: Path, rel: str) -> None:
        self._atomic_copy(Path(local), self._abs(rel))

    def get(self, rel: str, local: Path) -> None:
        src = self._abs(rel)
        if not src.exists():
            raise StagingError(f"durable object missing: {src}")
        self._atomic_copy(src, Path(local))

    def exists(self, rel: str) -> bool:
        return self._abs(rel).exists()

    def list(self, prefix: str = "") -> list[str]:
        base = self._abs(prefix) if prefix else self.root
        if not base.exists():
            return []
        return sorted(
            p.relative_to(self.root).as_posix()
            for p in base.iterdir()
            if p.is_file()
        )

    def read_text(self, rel: str) -> str:
        p = self._abs(rel)
        if not p.exists():
            raise StagingError(f"durable object missing: {p}")
        return p.read_text(encoding="utf-8")

    def write_text(self, rel: str, text: str) -> None:
        dst = self._abs(rel)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(dst)

    def verify(self) -> tuple[bool, str]:
        try:
            self.root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:  # unwritable mount / permission
            return False, f"local durable root not writable: {self.root} ({exc})"
        return True, f"local durable root {self.root}"


def _gcs_available(bucket_uri: str, run_fn=subprocess.run) -> tuple[bool, str]:
    """Return ``(ok, note)`` for a GCS bucket URI via ``gcloud storage ls`` (module-level so
    tests can monkeypatch it to force the credentials-absent branch deterministically)."""
    bucket = _bucket_root(bucket_uri)
    try:
        proc = run_fn([*_GCLOUD, "ls", bucket], capture_output=True, text=True)
    except FileNotFoundError:
        return False, "gcloud CLI not found on PATH (install the Google Cloud SDK)"
    if getattr(proc, "returncode", 1) == 0:
        return True, f"gcloud storage reachable: {bucket}"
    err = (getattr(proc, "stderr", "") or "").strip().splitlines()
    return False, f"gcloud storage unavailable for {bucket}: {err[-1] if err else 'auth/bucket error'}"


def _bucket_root(bucket_uri: str) -> str:
    """``gs://bucket/some/prefix`` -> ``gs://bucket`` (the reachability check target)."""
    rest = bucket_uri[len("gs://"):]
    return "gs://" + rest.split("/", 1)[0]


class GcsBackend(DurableBackend):
    """Durable root in a GCS bucket, driven by the ``gcloud storage`` CLI.

    ``run_fn`` defaults to :func:`subprocess.run` and is injected in tests. Any real transfer
    while credentials/bucket are unreachable raises :class:`RequiresManualOptIn` — matching the
    repo's gating convention (see ``acquire/fivek_kaggle`` credentials handling).
    """

    def __init__(self, bucket_uri: str, run_fn=subprocess.run):
        self.uri = bucket_uri.rstrip("/")
        if not self.uri.startswith("gs://"):
            raise StagingError(f"not a gs:// URI: {bucket_uri}")
        self.run_fn = run_fn
        self._available: bool | None = None

    def _key_uri(self, rel: str) -> str:
        return f"{self.uri}/{rel.lstrip('/')}"

    def _ensure(self) -> None:
        if self._available is None:
            ok, note = self.verify()
            self._available = ok
            if not ok:
                raise RequiresManualOptIn(
                    f"GCS durable root not usable ({note}). Run `gcloud auth login` and check the "
                    f"bucket exists/is writable, then retry."
                )

    def _run(self, *args: str, tolerant: bool = False):
        cmd = [*_GCLOUD, *args]
        try:
            proc = self.run_fn(cmd, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise StagingError(f"gcloud CLI not found: {exc}") from exc
        if getattr(proc, "returncode", 1) != 0 and not tolerant:
            err = (getattr(proc, "stderr", "") or "").strip()
            raise StagingError(f"`{' '.join(cmd)}` failed (rc={proc.returncode}): {err}")
        return proc

    def put(self, local: Path, rel: str) -> None:
        self._ensure()
        self._run("cp", str(local), self._key_uri(rel))

    def get(self, rel: str, local: Path) -> None:
        self._ensure()
        Path(local).parent.mkdir(parents=True, exist_ok=True)
        self._run("cp", self._key_uri(rel), str(local))

    def exists(self, rel: str) -> bool:
        self._ensure()
        proc = self._run("ls", self._key_uri(rel), tolerant=True)
        return getattr(proc, "returncode", 1) == 0

    def list(self, prefix: str = "") -> list[str]:
        self._ensure()
        target = self._key_uri(prefix) if prefix else self.uri + "/"
        proc = self._run("ls", target, tolerant=True)
        if getattr(proc, "returncode", 1) != 0:
            return []
        out = []
        for line in (getattr(proc, "stdout", "") or "").splitlines():
            line = line.strip()
            if not line or not line.startswith(self.uri) or line.endswith("/"):
                continue
            out.append(line[len(self.uri) + 1:])
        return sorted(out)

    def read_text(self, rel: str) -> str:
        self._ensure()
        proc = self._run("cat", self._key_uri(rel))
        return getattr(proc, "stdout", "") or ""

    def write_text(self, rel: str, text: str) -> None:
        self._ensure()
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".tmp", delete=False, encoding="utf-8") as fh:
            fh.write(text)
            tmp = fh.name
        try:
            self._run("cp", tmp, self._key_uri(rel))
        finally:
            os.unlink(tmp)

    def verify(self) -> tuple[bool, str]:
        return _gcs_available(self.uri, self.run_fn)


# --- HuggingFace Hub backend (ADR-0018 named HF as a durable root; sibling of GcsBackend) ----
def _parse_hf_uri(uri: str) -> tuple[str, str, str]:
    """``hf://[datasets/|models/]user/repo[/prefix]`` -> (repo_type, repo_id, in-repo prefix)."""
    rest = uri[len("hf://"):].strip("/")
    repo_type = "dataset"
    if rest.startswith("datasets/"):
        rest = rest[len("datasets/"):]
    elif rest.startswith("models/"):
        repo_type, rest = "model", rest[len("models/"):]
    parts = rest.split("/")
    if len(parts) < 2:
        raise StagingError(f"HF durable root needs 'user/repo': {uri}")
    return repo_type, "/".join(parts[:2]), "/".join(parts[2:])


class _HfHubClient:
    """Thin wrapper over ``huggingface_hub`` — the single seam tests replace with a local fake."""

    def __init__(self):
        from huggingface_hub import HfApi
        self._api = HfApi()

    def token(self):
        from huggingface_hub import get_token
        return get_token()

    def repo_exists(self, repo_id: str, repo_type: str) -> bool:
        return self._api.repo_exists(repo_id, repo_type=repo_type)

    def upload(self, local: Path, rel: str, repo_id: str, repo_type: str) -> None:
        self._api.upload_file(path_or_fileobj=str(local), path_in_repo=rel,
                              repo_id=repo_id, repo_type=repo_type)

    def download(self, rel: str, dest: Path, repo_id: str, repo_type: str) -> None:
        import tempfile

        from huggingface_hub import hf_hub_download
        with tempfile.TemporaryDirectory() as td:
            p = hf_hub_download(repo_id=repo_id, filename=rel, repo_type=repo_type, local_dir=td)
            Path(dest).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(p, dest)

    def exists(self, rel: str, repo_id: str, repo_type: str) -> bool:
        return self._api.file_exists(repo_id, rel, repo_type=repo_type)

    def list(self, repo_id: str, repo_type: str) -> list[str]:
        return list(self._api.list_repo_files(repo_id, repo_type=repo_type))


def _hf_available(repo_id: str, repo_type: str, client) -> tuple[bool, str]:
    """(ok, note) for an HF repo — module-level so tests can force the no-credentials branch."""
    try:
        if not client.token():
            return False, "no HuggingFace token (run `hf auth login` or set HF_TOKEN)"
        if not client.repo_exists(repo_id, repo_type):
            return False, f"HF {repo_type} repo '{repo_id}' not found (create it first)"
    except Exception as exc:  # noqa: BLE001 - hub/network resilience
        return False, f"HF hub unreachable for {repo_id}: {exc}"
    return True, f"HF {repo_type} repo {repo_id} reachable"


class HfBackend(DurableBackend):
    """Durable root in a HuggingFace Hub repo (default: a private dataset repo).

    Drives ``huggingface_hub`` through an injectable ``client`` (default :class:`_HfHubClient`);
    a missing token/repo raises :class:`RequiresManualOptIn` rather than silently no-op'ing.
    """

    def __init__(self, uri: str, client=None):
        self.uri = uri.rstrip("/")
        self.repo_type, self.repo_id, self._prefix = _parse_hf_uri(self.uri)
        self.client = client or _HfHubClient()
        self._available: bool | None = None

    def _key(self, rel: str) -> str:
        rel = rel.lstrip("/")
        return f"{self._prefix}/{rel}" if self._prefix else rel

    def _unkey(self, key: str) -> str:
        if self._prefix and key.startswith(self._prefix + "/"):
            return key[len(self._prefix) + 1:]
        return key

    def _ensure(self) -> None:
        if self._available is None:
            ok, note = self.verify()
            self._available = ok
            if not ok:
                raise RequiresManualOptIn(
                    f"HF durable root not usable ({note}). Authenticate with `hf auth login` and "
                    f"ensure the repo exists, then retry."
                )

    def put(self, local, rel):
        self._ensure()
        self.client.upload(Path(local), self._key(rel), self.repo_id, self.repo_type)

    def get(self, rel, local):
        self._ensure()
        self.client.download(self._key(rel), Path(local), self.repo_id, self.repo_type)

    def exists(self, rel):
        self._ensure()
        return self.client.exists(self._key(rel), self.repo_id, self.repo_type)

    def list(self, prefix: str = ""):
        self._ensure()
        want = self._key(prefix) if prefix else self._prefix
        return sorted(self._unkey(k) for k in self.client.list(self.repo_id, self.repo_type)
                      if not want or k.startswith(want))

    def read_text(self, rel):
        self._ensure()
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "obj"
            self.client.download(self._key(rel), dest, self.repo_id, self.repo_type)
            return dest.read_text(encoding="utf-8")

    def write_text(self, rel, text):
        self._ensure()
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            src = Path(td) / "obj"
            src.write_text(text, encoding="utf-8")
            self.client.upload(src, self._key(rel), self.repo_id, self.repo_type)

    def verify(self):
        return _hf_available(self.repo_id, self.repo_type, self.client)


def backend_for(durable_root: str | os.PathLike, run_fn=subprocess.run, hf_client=None) -> DurableBackend:
    """Pick a backend from the durable root: ``gs://…`` -> GCS, ``hf://…`` -> HF Hub, else local."""
    s = str(durable_root)
    if s.startswith("gs://"):
        return GcsBackend(s, run_fn=run_fn)
    if s.startswith("hf://"):
        return HfBackend(s, client=hf_client)
    return LocalBackend(s)
