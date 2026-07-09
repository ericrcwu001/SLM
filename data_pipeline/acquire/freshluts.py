"""FreshLUTs authenticated connector (freshluts.com — Rails/Devise).

The user granted permission to sign up and download the full catalog. Flow:
  GET /users/sign_in -> CSRF -> POST /users/sign_in  (auto sign-up via /users/sign_up if login
  fails) -> crawl /browse for /luts/{id} -> parse each LUT page -> download its ``.cube`` to
  ``luts/raw/fresh_luts/{author}/{lut_id}.cube`` (license/title/author captured).

Credentials come from env (``SLM_FRESHLUTS_EMAIL`` / ``SLM_FRESHLUTS_PASSWORD``); never stored.
Resumable (skips ids already on disk), rate-limited. Network I/O goes through an injectable
``session`` so the Devise flow is unit-tested offline.
"""

from __future__ import annotations

import os
import random
import re
import time
from pathlib import Path
from urllib.parse import urljoin

from ..errors import RequiresManualOptIn
from . import downloaders as dl
from .base import AcquireLimits, AcquireReport, RawArtifact, utcnow_iso

_BASE = "https://freshluts.com"


class FreshLutsConnector:
    source_pack_id = "freshluts_public"
    family = "fresh_luts"

    def __init__(self, session=None, base_url: str = _BASE, email: str | None = None,
                 password: str | None = None, min_delay_s: float = 1.5, jitter_s: float = 2.0,
                 scan_max_id: int = 3000):
        self.base_url = base_url.rstrip("/")
        self.session = session
        self.email = email or os.environ.get("SLM_FRESHLUTS_EMAIL")
        self.password = password or os.environ.get("SLM_FRESHLUTS_PASSWORD")
        # jittered pacing so the request cadence isn't a fixed interval (Cloudflare-friendly)
        self.min_delay_s = min_delay_s
        self.jitter_s = jitter_s
        # the /luts?page catalog only surfaces a fraction of the library; valid LUTs are
        # reachable by direct /download/{id} well beyond it, so we scan the full id range.
        self.scan_max_id = scan_max_id

    def _sleep(self, base: float) -> None:
        d = base + random.uniform(0.0, self.jitter_s)
        if d > 0:
            time.sleep(d)

    # --- helpers ---
    def _url(self, path: str) -> str:
        return urljoin(self.base_url + "/", path.lstrip("/"))

    @staticmethod
    def _csrf(html: str) -> str | None:
        m = re.search(r'name="csrf-token"\s+content="([^"]+)"', html)
        if m:
            return m.group(1)
        m = re.search(r'name="authenticity_token"\s+value="([^"]+)"', html)
        return m.group(1) if m else None

    def verify(self) -> tuple[bool, str]:
        if not self.email or not self.password:
            return False, "missing SLM_FRESHLUTS_EMAIL / SLM_FRESHLUTS_PASSWORD"
        try:
            ok, note = dl.http_head_ok(self._url("/users/sign_in"), session=self.session)
            return ok, note
        except Exception as e:  # noqa: BLE001
            return False, f"unreachable: {e}"

    def _require_creds(self):
        if not self.email or not self.password:
            raise RequiresManualOptIn(
                "FreshLUTs needs SLM_FRESHLUTS_EMAIL / SLM_FRESHLUTS_PASSWORD (one-time; "
                "a Devise email-confirmation may need to be completed once)."
            )

    def sign_in(self, session) -> tuple[bool, str]:
        """Log in; auto sign-up on failure. Returns (signed_in, note)."""
        page = session.get(self._url("/users/sign_in")).text
        token = self._csrf(page)
        resp = session.post(self._url("/users/sign_in"), data={
            "user[email]": self.email, "user[password]": self.password,
            "authenticity_token": token or "", "commit": "Log in",
        })
        if self._looks_signed_in(getattr(resp, "text", "")):
            return True, "signed_in"
        # try sign up, then log in again
        up = session.get(self._url("/users/sign_up")).text
        up_token = self._csrf(up)
        reg = session.post(self._url("/users"), data={
            "user[email]": self.email, "user[password]": self.password,
            "user[password_confirmation]": self.password, "authenticity_token": up_token or "",
            "commit": "Sign up",
        })
        rtext = getattr(reg, "text", "")
        if re.search(r"confirm(ation)? (your )?email|confirmation link", rtext, re.I):
            return False, "email_confirmation_required"
        page2 = session.get(self._url("/users/sign_in")).text
        resp2 = session.post(self._url("/users/sign_in"), data={
            "user[email]": self.email, "user[password]": self.password,
            "authenticity_token": self._csrf(page2) or "", "commit": "Log in",
        })
        if self._looks_signed_in(getattr(resp2, "text", "")):
            return True, "signed_up_then_in"
        return False, "sign_in_failed"

    @staticmethod
    def _looks_signed_in(html: str) -> bool:
        # signed-in pages expose a sign-out control and no sign-in form action
        if re.search(r'href="/users/sign_out"|Sign out|Log out', html, re.I):
            return True
        return "user[password]" not in html and "sign_in" not in html.lower()

    def _discover_ids(self, session, limit: int | None, max_pages: int = 4000) -> list[int]:
        """Return the full 1..scan_max_id id range to scan via direct ``/download/{id}``.

        The ``/luts?page=N`` catalog listing is both incomplete and slow to crawl; the range
        scan is a strict superset of it (observed max id ~2745), so we skip the crawl and let
        gap ids return non-cube (skipped fast in ``acquire``).
        """
        ordered = list(range(1, self.scan_max_id + 1))
        return ordered[: limit] if limit is not None else ordered

    def _parse_lut_page(self, html: str) -> dict:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        download_url = None
        for a in soup.find_all("a"):
            href = a.get("href") or ""
            text = (a.get_text() or "").lower()
            if href.lower().endswith(".cube") or "/download" in href.lower() or "download" in text:
                download_url = href
                break
        lic = None
        m = re.search(r"(CC0[^<]*|Creative Commons[^<]*|Free for commercial use)", html, re.I)
        if m:
            lic = m.group(1).strip()
        title_el = soup.find("h1")
        title = title_el.get_text().strip() if title_el else None
        return {"download_url": download_url, "license": lic, "title": title}

    def acquire(self, raw_root, limits: AcquireLimits) -> AcquireReport:
        report = AcquireReport(source_pack_id=self.source_pack_id)
        self._require_creds()
        session = self.session or dl.polite_session()
        base_delay = max(self.min_delay_s, limits.rate_limit_s or 0.0)

        signed, note = self.sign_in(session)
        if not signed:
            report.status = "skipped" if note == "email_confirmation_required" else "failed"
            report.note = f"auth: {note}"
            return report

        ids = self._discover_ids(session, limits.max_items)
        if not ids:
            report.status = "partial"
            report.note = "no /luts ids discovered from /browse"
            return report

        dest_root = Path(raw_root) / "fresh_luts" / "freshluts"
        dest_root.mkdir(parents=True, exist_ok=True)
        print(f"[freshluts] scanning {len(ids)} ids for .cube downloads", flush=True)
        ts = utcnow_iso()
        for lut_id in ids:
            report.attempted += 1
            out = dest_root / f"{lut_id}.cube"
            if out.exists():
                report.skipped += 1
                report.artifacts.append(self._artifact(lut_id, out, None, ts))
            else:
                try:
                    info = self._download_cube(session, lut_id, out)
                    if info is None:
                        report.skipped += 1
                        self._sleep_gap()               # light pace for gap/404 probes
                    else:
                        report.artifacts.append(self._artifact(lut_id, out, info, ts))
                        report.acquired += 1
                        self._sleep(base_delay)         # full jittered pace after a real download
                except Exception:  # noqa: BLE001
                    report.failed += 1
            if report.attempted % 100 == 0:
                print(f"[freshluts] scanned={report.attempted}/{len(ids)} id={lut_id} "
                      f"downloaded={report.acquired}", flush=True)
        report.status = "ok" if (report.acquired or report.skipped) else "partial"
        return report

    def _sleep_gap(self) -> None:
        if self.min_delay_s <= 0:
            return  # test/fast mode
        time.sleep(0.2 + random.uniform(0.0, 0.3))  # gaps are cheap; keep requests polite but fast

    def _download_cube(self, session, lut_id: int, out: Path) -> dict | None:
        """Download LUT #id via the real per-LUT flow, or None if it isn't a real LUT.

        The direct ``/download/{id}`` endpoint returns rotating/wrong content; the authoritative
        path is the LUT page's form: GET ``/luts/{id}`` -> POST its ``/downloadlut?lutid=..&userid=..``
        action with the page ``authenticity_token`` (CSRF) + a Referer header. A page with no such
        form is a gap (not a real LUT) -> None.
        """
        from bs4 import BeautifulSoup

        page_url = self._url(f"/luts/{lut_id}")
        try:
            page = session.get(page_url, timeout=30).text
        except Exception:  # noqa: BLE001
            return None
        soup = BeautifulSoup(page, "html.parser")
        form = next((f for f in soup.find_all("form") if "downloadlut" in (f.get("action") or "")), None)
        if form is None:
            return None  # not a real LUT id
        token = next((i.get("value") for i in form.find_all("input")
                      if i.get("name") == "authenticity_token"), None)
        try:
            r = session.post(self._url(form["action"]),
                             data={"authenticity_token": token or "", "commit": "Download LUT"},
                             headers={"Referer": page_url}, allow_redirects=True, timeout=30)
        except Exception:  # noqa: BLE001
            return None
        content = r.content if isinstance(r.content, (bytes, bytearray)) else str(r.content or "").encode()
        ctype = r.headers.get("Content-Type", "") if hasattr(r, "headers") else ""
        if getattr(r, "status_code", 200) >= 400:
            return None
        if b"LUT_3D_SIZE" not in content[:4096] and "octet-stream" not in ctype:
            return None
        out.write_bytes(content)
        disp = r.headers.get("Content-Disposition", "") if hasattr(r, "headers") else ""
        m = re.search(r'filename="?([^\";]+)', disp)
        lic = re.search(r"(CC0[^<]*|Creative Commons[^<]*|Free for commercial use)", page, re.I)
        return {"title": (m.group(1) if m else None),
                "license": (lic.group(1).strip() if lic else "freshluts (per-LUT)")}

    def _artifact(self, lut_id: int, path: Path, info: dict | None, ts: str) -> RawArtifact:
        return RawArtifact(
            kind="lut_file", source_pack_id=self.source_pack_id, family=self.family,
            declared_domain="srgb", license=(info or {}).get("license") or "freshluts (per-LUT)",
            source_url=self._url(f"/luts/{lut_id}"),
            file_hash=(dl.sha256_file(path) if path.exists() else None), download_timestamp=ts,
            lut_id=f"fresh_{lut_id}", file_path=str(path), derivation_method="cube",
            author_uploader_pack_id="freshluts",
            extra={"title": (info or {}).get("title")},
        )
