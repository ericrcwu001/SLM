"""Low-level fetch helpers: resumable HTTP, archive extraction, HuggingFace, robots.

All network use is optional/lazy-imported so the package imports without the extras and so
tests never touch the network. Callers wrap these and downgrade failures to report entries.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

USER_AGENT = "slm-datagen/0.1 (+research; contact via repo)"
_CHUNK = 1 << 16


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def polite_session():
    import requests

    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def http_download(url: str, dest: str | Path, *, session=None, resume: bool = True,
                  timeout: int = 60, retries: int = 3) -> Path:
    """Download ``url`` -> ``dest`` with streaming, resume, and retry/backoff."""
    import time

    import requests

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0 and resume:
        return dest  # resumable skip (already present)
    sess = session or polite_session()
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            with sess.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(_CHUNK):
                        if chunk:
                            fh.write(chunk)
                tmp.replace(dest)
            return dest
        except Exception as e:  # noqa: BLE001 - network resilience
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"download failed after {retries} attempts: {url}: {last_err}")


def http_head_ok(url: str, *, session=None, timeout: int = 30) -> tuple[bool, str]:
    import requests

    sess = session or polite_session()
    try:
        r = sess.head(url, timeout=timeout, allow_redirects=True)
        if r.status_code < 400:
            return True, f"HEAD {r.status_code}"
        # some servers reject HEAD; try a ranged GET
        r = sess.get(url, timeout=timeout, stream=True, headers={"Range": "bytes=0-0"})
        return (r.status_code < 400), f"GET {r.status_code}"
    except Exception as e:  # noqa: BLE001
        return False, f"unreachable: {e}"


def extract_zip(zip_path: str | Path, dest_dir: str | Path,
                suffixes: Optional[Iterable[str]] = None, max_items: Optional[int] = None) -> list[Path]:
    """Extract (a bounded set of) members from a zip. Returns extracted file paths."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if not n.endswith("/")]
        if suffixes:
            sfx = tuple(s.lower() for s in suffixes)
            names = [n for n in names if n.lower().endswith(sfx)]
        names.sort()
        if max_items is not None:
            names = names[:max_items]
        for name in names:
            target = dest_dir / Path(name).name
            with zf.open(name) as src, open(target, "wb") as fh:
                fh.write(src.read())
            out.append(target)
    return out


def robots_allowed(url: str, *, user_agent: str = USER_AGENT) -> bool:
    """Check robots.txt for ``url``. Fails open (True) if robots is unreachable/empty."""
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(user_agent, url)
    except Exception:  # noqa: BLE001
        return True


# --- HuggingFace ------------------------------------------------------------------
def hf_list_files(repo_id: str, repo_type: str = "dataset") -> list[str]:
    from huggingface_hub import list_repo_files

    return list(list_repo_files(repo_id, repo_type=repo_type))


def hf_download_file(repo_id: str, filename: str, local_dir: str | Path,
                     repo_type: str = "dataset") -> Path:
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type,
                        local_dir=str(local_dir))
    return Path(p)
