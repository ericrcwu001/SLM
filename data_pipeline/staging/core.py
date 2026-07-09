"""``slm_stage`` core: pack / stage / push (ADR-0018).

* ``run_pack``  — corpus dirs -> bounded, structure-preserving ``.tar`` shards + a
  ``stage_manifest.json`` (per-shard member count / bytes / sha256) in the durable root.
  Idempotent: a corpus whose (path, size) set is unchanged and whose shards are all present is
  skipped.
* ``run_stage`` — shards -> ``local_root`` (default ``/content/slm``): disk-aware, sha256-verified,
  resumable (a ``.shards/<name>.done`` marker per extracted shard). Afterwards ``local_root`` is a
  drop-in ``SLM_ARTIFACT_ROOT``.
* ``run_push``  — output dirs (checkpoints/eval) -> durable root, rate-limited, with a manifest, so
  they survive Colab session teardown.

Reuses ``downloaders.sha256_file`` (shard hashing), ``base.RateLimiter``/``utcnow_iso`` (push
bounding / timestamps), and ``paths.artifact_paths`` (root resolution). A ``tarfile`` packer is used
instead of ``downloaders.extract_zip`` because the latter flattens nested layout.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

from ..acquire.base import RateLimiter, utcnow_iso
from ..acquire.downloaders import sha256_file
from ..errors import RequiresManualOptIn, StagingError
from ..paths import artifact_paths
from .backends import DurableBackend, GcsBackend, HfBackend, backend_for

STAGING_VERSION = "v1"
MANIFEST_NAME = "stage_manifest.json"
PUSH_MANIFEST_NAME = "push_manifest.json"
_MARKER_DIR = ".shards"

# Corpus dirs to pack (relative to the artifact root); overridable via config `pack.include`.
DEFAULT_INCLUDE = ["luts/raw", "luts/canonical_residual", "data/raw_registry", "data/splits", "tokenizer/final"]
DEFAULT_EXCLUDE = ["*.lock", "*.metadata", "*.part"]
DEFAULT_PUSH_INCLUDE = ["models/**/*", "tokenizer/**/*", "eval_runs/**/*"]
_DEFAULT_SHARD_MAX_BYTES = 2 * 1024**3          # ~2 GiB, per ADR-0018
_DEFAULT_MIN_FREE_BYTES = 16 * 1024**3          # stage headroom


@dataclass
class StageReport:
    """Result of a pack/stage/push run (mirrors acquire.AcquireReport.summary)."""
    command: str
    status: str = "ok"                          # ok | skipped | partial | failed
    attempted: int = 0
    transferred: int = 0
    skipped: int = 0
    verified: int = 0
    bytes: int = 0
    durable_root: str = ""
    note: str = ""
    shards: list = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "command": self.command,
            "status": self.status,
            "attempted": self.attempted,
            "transferred": self.transferred,
            "skipped": self.skipped,
            "verified": self.verified,
            "bytes": self.bytes,
            "durable_root": self.durable_root,
            "note": self.note,
            "shards": self.shards,
        }


# --- helpers ------------------------------------------------------------------------
def _iter_corpus_files(root: Path, include: list[str], exclude: list[str]) -> list[tuple[Path, str]]:
    """(abs_path, arcname-relative-to-root) for every included, non-excluded file, sorted."""
    seen: dict[str, Path] = {}
    for inc in include:
        base = root / inc
        if not base.exists():
            continue
        candidates = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
        for p in candidates:
            rel = p.relative_to(root).as_posix()
            if any(fnmatch(p.name, pat) or fnmatch(rel, pat) for pat in exclude):
                continue
            seen[rel] = p
    return [(seen[rel], rel) for rel in sorted(seen)]


def _plan_shards(files: list[tuple[Path, str]], shard_max_bytes: int) -> list[list[tuple[Path, str]]]:
    """Greedy bin-pack into shards of ~shard_max_bytes; never splits a single file across shards."""
    shards: list[list[tuple[Path, str]]] = []
    cur: list[tuple[Path, str]] = []
    cur_bytes = 0
    for p, rel in files:
        sz = p.stat().st_size
        if cur and cur_bytes + sz > shard_max_bytes:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append((p, rel))
        cur_bytes += sz
    if cur:
        shards.append(cur)
    return shards


def _content_key(files: list[tuple[Path, str]]) -> str:
    h = hashlib.sha256()
    for p, rel in files:
        h.update(f"{rel}:{p.stat().st_size}\n".encode("utf-8"))
    return h.hexdigest()


def _shard_name(i: int, compression: str) -> str:
    ext = "tar.gz" if compression in ("gz", "gzip") else "tar"
    return f"corpus-{i:04d}.{ext}"


def _write_tar(dst: Path, members: list[tuple[Path, str]], compression: str) -> None:
    mode = "w:gz" if compression in ("gz", "gzip") else "w"
    tmp = dst.with_suffix(dst.suffix + ".part")
    with tarfile.open(tmp, mode) as tar:
        for p, arc in members:
            tar.add(str(p), arcname=arc, recursive=False)
    tmp.replace(dst)


def _safe_extract(tar: tarfile.TarFile, dest: Path) -> None:
    # `filter="data"` (py3.12+ / 3.11.4+) blocks path-traversal/absolute members; our own tars are
    # relative, but keep the guard for defense-in-depth and forward-compat.
    try:
        tar.extractall(dest, filter="data")  # type: ignore[call-arg]
    except TypeError:
        tar.extractall(dest)


def _verify_or_raise(backend: DurableBackend) -> None:
    ok, note = backend.verify()
    if ok:
        return
    if isinstance(backend, (GcsBackend, HfBackend)):   # remote creds/opt-in gate, not a hard error
        raise RequiresManualOptIn(note)
    raise StagingError(note)


def _read_manifest(backend: DurableBackend) -> dict | None:
    if not backend.exists(MANIFEST_NAME):
        return None
    try:
        return json.loads(backend.read_text(MANIFEST_NAME))
    except (ValueError, StagingError):
        return None


# --- pack ---------------------------------------------------------------------------
def run_pack(root, durable_root, config: dict, *, run_fn=subprocess.run, hf_client=None,
             dry_run: bool = False, log_fn=print) -> dict:
    paths = artifact_paths(root)
    pcfg = config.get("pack", {}) or {}
    include = pcfg.get("include", DEFAULT_INCLUDE)
    exclude = pcfg.get("exclude", DEFAULT_EXCLUDE)
    shard_max = int(pcfg.get("shard_max_bytes", _DEFAULT_SHARD_MAX_BYTES))
    compression = str(pcfg.get("compression", "none"))

    files = _iter_corpus_files(paths.root, include, exclude)
    rep = StageReport("pack", durable_root=str(durable_root))
    if not files:
        rep.status = "skipped"
        rep.note = f"no files found under include={include} at root {paths.root}"
        log_fn(f"[pack] {rep.note}")
        return rep.summary()

    plan = _plan_shards(files, shard_max)
    total_bytes = sum(p.stat().st_size for p, _ in files)
    rep.attempted = len(plan)
    rep.bytes = total_bytes

    if dry_run:
        rep.shards = [
            {"name": _shard_name(i, compression), "member_count": len(s),
             "bytes": sum(p.stat().st_size for p, _ in s)}
            for i, s in enumerate(plan)
        ]
        rep.note = (f"DRY-RUN: {len(files)} files -> {len(plan)} shards "
                    f"(~{shard_max} B cap), {total_bytes} B total; nothing written")
        log_fn(f"[pack] {rep.note}")
        for s in rep.shards:
            log_fn(f"[pack]   {s['name']}: {s['member_count']} files, {s['bytes']} B")
        return rep.summary()

    backend = backend_for(durable_root, run_fn=run_fn, hf_client=hf_client)
    _verify_or_raise(backend)

    content_key = _content_key(files)
    existing = _read_manifest(backend)
    if (existing and existing.get("content_key") == content_key
            and all(backend.exists(s["name"]) for s in existing.get("shards", []))):
        rep.status = "skipped"
        rep.skipped = len(existing.get("shards", []))
        rep.shards = existing.get("shards", [])
        rep.note = "corpus unchanged; shards already present in durable root"
        log_fn(f"[pack] {rep.note}")
        return rep.summary()

    shard_records: list[dict] = []
    with tempfile.TemporaryDirectory() as td:
        for i, members in enumerate(plan):
            name = _shard_name(i, compression)
            local_tar = Path(td) / name
            _write_tar(local_tar, members, compression)
            sha = sha256_file(local_tar)
            nbytes = local_tar.stat().st_size
            backend.put(local_tar, name)
            local_tar.unlink(missing_ok=True)   # reclaim immediately; peak temp disk = 1 shard
            shard_records.append({"name": name, "member_count": len(members),
                                  "bytes": nbytes, "sha256": sha})
            rep.transferred += 1
            log_fn(f"[pack] {name}: {len(members)} files, {nbytes} B, sha256 {sha[:12]}…")

    manifest = {
        "staging_version": STAGING_VERSION,
        "created_at": utcnow_iso(),
        "content_key": content_key,
        "root_include": include,
        "compression": compression,
        "total_bytes": total_bytes,
        "shards": shard_records,
    }
    backend.write_text(MANIFEST_NAME, json.dumps(manifest, indent=2, sort_keys=True))
    rep.shards = shard_records
    rep.note = f"packed {len(files)} files -> {len(shard_records)} shards to {backend.uri}"
    log_fn(f"[pack] {rep.note}")
    return rep.summary()


# --- stage --------------------------------------------------------------------------
def run_stage(durable_root, local_root, config: dict, *, run_fn=subprocess.run, hf_client=None,
              log_fn=print) -> dict:
    scfg = config.get("stage", {}) or {}
    min_free = int(scfg.get("min_free_bytes", _DEFAULT_MIN_FREE_BYTES))
    local = Path(local_root)
    local.mkdir(parents=True, exist_ok=True)

    backend = backend_for(durable_root, run_fn=run_fn, hf_client=hf_client)
    _verify_or_raise(backend)
    rep = StageReport("stage", durable_root=str(durable_root))
    if not backend.exists(MANIFEST_NAME):
        raise StagingError(f"no {MANIFEST_NAME} in durable root {backend.uri}; run `slm_stage pack` first")

    manifest = json.loads(backend.read_text(MANIFEST_NAME))
    shards = manifest.get("shards", [])
    rep.attempted = len(shards)

    free = shutil.disk_usage(local).free
    if free < min_free:
        rep.status = "skipped"
        rep.note = (f"insufficient disk at {local}: {free // 2**30} GiB free "
                    f"< required {min_free // 2**30} GiB headroom")
        log_fn(f"[stage] {rep.note}")
        return rep.summary()

    marker_dir = local / _MARKER_DIR
    marker_dir.mkdir(parents=True, exist_ok=True)
    for s in shards:
        name = s["name"]
        done = marker_dir / (name + ".done")
        if done.exists():
            rep.skipped += 1
            continue
        tmp = marker_dir / name
        backend.get(name, tmp)
        got = sha256_file(tmp)
        if got != s["sha256"]:
            tmp.unlink(missing_ok=True)
            raise StagingError(f"sha256 mismatch for {name}: got {got[:12]}… expected {s['sha256'][:12]}…")
        rep.verified += 1
        with tarfile.open(tmp, "r:*") as tar:
            _safe_extract(tar, local)
        tmp.unlink(missing_ok=True)          # reclaim shard; keep the .done marker for resume
        done.write_text(got, encoding="utf-8")
        rep.transferred += 1
        rep.bytes += int(s.get("bytes", 0))
        log_fn(f"[stage] {name}: verified + extracted {s.get('member_count', '?')} files")

    rep.note = f"staged to {local}  (set SLM_ARTIFACT_ROOT={local})"
    log_fn(f"[stage] {rep.note}")
    return rep.summary()


# --- push ---------------------------------------------------------------------------
def _resolve_globs(base: Path, patterns: list[str]) -> list[Path]:
    out: set[Path] = set()
    for pat in patterns:
        for p in base.glob(pat):
            if p.is_file():
                out.add(p)
    return sorted(out)


def run_push(local_root, durable_root, config: dict, *, rate_limit_s: float | None = None,
             run_fn=subprocess.run, hf_client=None, log_fn=print) -> dict:
    pcfg = config.get("push", {}) or {}
    include = pcfg.get("include", DEFAULT_PUSH_INCLUDE)
    rate = float(rate_limit_s if rate_limit_s is not None else pcfg.get("rate_limit_s", 0.0) or 0.0)
    local = Path(local_root)

    backend = backend_for(durable_root, run_fn=run_fn, hf_client=hf_client)
    _verify_or_raise(backend)
    rep = StageReport("push", durable_root=str(durable_root))
    if not local.exists():
        rep.status = "skipped"
        rep.note = f"local root {local} does not exist; nothing to push"
        log_fn(f"[push] {rep.note}")
        return rep.summary()

    files = _resolve_globs(local, include)
    rep.attempted = len(files)
    limiter = RateLimiter(rate)
    records: list[dict] = []
    for p in files:
        rel_local = p.relative_to(local).as_posix()
        limiter.wait()
        backend.put(p, f"outputs/{rel_local}")
        sha = sha256_file(p)
        records.append({"path": rel_local, "sha256": sha, "bytes": p.stat().st_size})
        rep.transferred += 1
        rep.bytes += p.stat().st_size
        log_fn(f"[push] outputs/{rel_local} ({p.stat().st_size} B)")

    backend.write_text(PUSH_MANIFEST_NAME, json.dumps(
        {"staging_version": STAGING_VERSION, "created_at": utcnow_iso(), "files": records},
        indent=2, sort_keys=True))
    rep.note = f"pushed {len(files)} files to {backend.uri}/outputs"
    log_fn(f"[push] {rep.note}")
    return rep.summary()
