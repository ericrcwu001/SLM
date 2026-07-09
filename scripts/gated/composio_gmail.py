#!/usr/bin/env python3
"""Composio Gmail helper for the gated-LUT scraper.

Reads eric.wu@alphaaiengineering.com via the local Composio CLI (~/.composio/composio) to fetch
signup-confirmation links and emailed download links. Composio writes large fetch results to a temp
file (storedInFile) — this helper parses that file ON DISK and returns only small extracted data
(sender, subject, links), so full email bodies never enter the agent's context.

Usage:
  python composio_gmail.py fetch  --query 'from:noreply@site.com newer_than:1h' [--max 5]
  python composio_gmail.py links  --query '...' [--max 5] [--host-filter site.com]
It prints compact JSON: [{id, from, subject, date, links:[...]}].
"""

from __future__ import annotations

import argparse
import base64
import html as _html
import json
import os
import re
import subprocess
import sys

_B64URL_RE = re.compile(r'^[A-Za-z0-9_\-]+={0,2}$')


def _maybe_b64_html(s: str):
    """If s is a base64url-encoded Gmail body part, decode it (Composio redacts token URLs in its
    cleaned `messageText`, but the raw base64 `payload.body.data` is un-redacted)."""
    s = s.strip()
    if len(s) < 200 or not _B64URL_RE.match(s):
        return None
    try:
        dec = base64.urlsafe_b64decode(s + "=" * (-len(s) % 4)).decode("utf-8", "ignore")
    except Exception:
        return None
    return dec if ("<a" in dec.lower() or "<html" in dec.lower()) else None

COMPOSIO = os.path.expanduser("~/.composio/composio")
_URL_RE = re.compile(r'https?://[^\s"\'<>)\]]+')
_ANCHOR_RE = re.compile(r'<a\b[^>]*?href\s*=\s*["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.I | re.S)
_TAG_RE = re.compile(r'<[^>]+>')
# link text worth flagging as an account activation / verification / download action
_ACTION_WORDS = ("activate", "verify", "verif", "confirm", "download", "get ", "access",
                 "complete", "reset", "sign in", "log in", "view", "click here")


def _all_strings(o, out, depth=0):
    """Recursively collect every string value in the message (the HTML part can live anywhere)."""
    if depth > 10:
        return
    if isinstance(o, str):
        out.append(o)
        dec = _maybe_b64_html(o)   # also scan the decoded raw body (un-redacted links live here)
        if dec:
            out.append(dec)
    elif isinstance(o, dict):
        for v in o.values():
            _all_strings(v, out, depth + 1)
    elif isinstance(o, list):
        for v in o:
            _all_strings(v, out, depth + 1)


def _extract_anchors(strings):
    """Extract {text, href} for every <a href> across the message's HTML strings, deduped by href."""
    anchors, seen = [], set()
    for s in strings:
        if "<a" not in s.lower() or "href" not in s.lower():
            continue
        for href, inner in _ANCHOR_RE.findall(s):
            href = _html.unescape(href).strip()
            if not href.lower().startswith(("http://", "https://")) or href in seen:
                continue
            seen.add(href)
            text = _html.unescape(_TAG_RE.sub(" ", inner)).strip()
            anchors.append({"text": text[:90], "href": href})
    return anchors


def _run_fetch(query: str, max_results: int) -> dict:
    # include_payload=true is REQUIRED to get the full HTML body (with <a href> buttons like
    # "Activate your account"); without it Composio returns optimized metadata that drops them.
    data = {"max_results": max_results, "include_payload": True, "verbose": True}
    if query:
        data["query"] = query
    proc = subprocess.run(
        [COMPOSIO, "execute", "GMAIL_FETCH_EMAILS", "-d", json.dumps(data)],
        capture_output=True, text=True, timeout=120,
    )
    out = proc.stdout.strip()
    try:
        return json.loads(out)
    except Exception:
        return {"successful": False, "error": f"non-JSON (rc={proc.returncode}): {out[:300]} {proc.stderr[:300]}"}


def _load_payload(resp: dict):
    """Return the actual response body: inline `data`, or the JSON at outputFilePath."""
    if resp.get("storedInFile") and resp.get("outputFilePath"):
        p = resp["outputFilePath"]
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as fh:
                return json.load(fh)
        except Exception:
            # some versions store the raw tool payload, others wrap it — fall back to text scan
            try:
                return {"_raw_text": open(p, encoding="utf-8", errors="ignore").read()}
            except Exception:
                return {}
    return resp.get("data") or resp


def _find_messages(o):
    if isinstance(o, dict):
        for key in ("messages", "emails", "items", "results"):
            if isinstance(o.get(key), list):
                return o[key]
        for v in o.values():
            r = _find_messages(v)
            if r:
                return r
    elif isinstance(o, list) and o and isinstance(o[0], dict):
        return o
    return []


def _msg_fields(m: dict) -> dict:
    subj = m.get("subject") or m.get("Subject")
    frm = m.get("sender") or m.get("from") or m.get("From")
    date = m.get("messageTimestamp") or m.get("date") or m.get("internalDate")
    mid = m.get("messageId") or m.get("id") or m.get("threadId")
    # recursively scan ALL string fields (the HTML part can be nested anywhere in the payload),
    # then pull anchors WITH their visible text so callers can pick the right button.
    strings: list[str] = []
    _all_strings(m, strings)
    blob = " ".join(strings)
    links = sorted(set(_html.unescape(u) for u in _URL_RE.findall(blob)))
    anchors = _extract_anchors(strings)
    # "actions" = anchors whose link text names an activate/verify/confirm/download action
    actions = [a for a in anchors if any(w in a["text"].lower() for w in _ACTION_WORDS)]
    return {"id": str(mid)[:24], "from": str(frm)[:80], "subject": str(subj)[:120],
            "date": str(date)[:32], "links": links, "anchors": anchors, "actions": actions}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["fetch", "links"])
    ap.add_argument("--query", default="")
    ap.add_argument("--max", type=int, default=5)
    ap.add_argument("--host-filter", default="", help="keep only links/anchors whose href contains this")
    ap.add_argument("--match", default="", help="keep only anchors/links whose text or href contains this (e.g. activate)")
    args = ap.parse_args()

    resp = _run_fetch(args.query, args.max)
    if not resp.get("successful"):
        print(json.dumps({"ok": False, "error": resp.get("error")}))
        return 1
    payload = _load_payload(resp)
    msgs = _find_messages(payload)
    rows = [_msg_fields(m) for m in msgs if isinstance(m, dict)]
    hf, mt = args.host_filter.lower(), args.match.lower()
    for r in rows:
        if hf:
            r["links"] = [u for u in r["links"] if hf in u.lower()]
            r["anchors"] = [a for a in r["anchors"] if hf in a["href"].lower()]
            r["actions"] = [a for a in r["actions"] if hf in a["href"].lower()]
        if mt:
            r["links"] = [u for u in r["links"] if mt in u.lower()]
            r["anchors"] = [a for a in r["anchors"] if mt in a["text"].lower() or mt in a["href"].lower()]
            r["actions"] = [a for a in r["anchors"]]  # when matching, anchors already narrowed
    print(json.dumps({"ok": True, "count": len(rows), "messages": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
