#!/usr/bin/env python3
"""CDP-driven headless-browser toolkit for the gated-LUT scraper (stealth + logging).

A long-lived Chromium (launched with --remote-debugging-port) persists cookies/state across
separate CLI invocations, so an agent can drive it step-by-step with its own vision. Includes
anti-automation stealth (mask navigator.webdriver, plugins, languages, WebGL, chrome runtime,
realistic UA/locale/flags + human-like mouse moves) so checkpoints (Vercel/DataDome/Cloudflare)
are less likely to flag it, and appends every action to ~/.slm_gated_browser/actions_<port>.log
for live visibility (tail -f).

Commands (global --port, default 9222; --settle ms after actions):
  start [--profile DIR] | goto <url> | shot <path> [--full] | text [sel] | html [sel]
  click <sel> | clickxy <x> <y> | fill <sel> <val> | press <Key> | eval <js>
  waitfor <sel> [--ms N] | download <sel> <destdir> | cookies | stop
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import time
from datetime import datetime
from pathlib import Path

STATE_DIR = Path(os.path.expanduser("~/.slm_gated_browser"))

# Injected into every new document before site JS runs — removes the common headless/automation
# tells that bot-management (Vercel, DataDome, Cloudflare, reCAPTCHA) fingerprints on.
STEALTH_JS = r"""
() => {
  Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
  Object.defineProperty(navigator, 'languages', {get: () => ['en-US','en']});
  Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5].map(i => ({name:'Plugin '+i}))});
  Object.defineProperty(navigator, 'platform', {get: () => 'MacIntel'});
  Object.defineProperty(navigator, 'hardwareConcurrency', {get: () => 8});
  Object.defineProperty(navigator, 'deviceMemory', {get: () => 8});
  window.chrome = window.chrome || {runtime: {}, app: {}, csi: () => {}, loadTimes: () => {}};
  const origQuery = window.navigator.permissions && window.navigator.permissions.query;
  if (origQuery) {
    window.navigator.permissions.query = (p) => (p && p.name === 'notifications'
      ? Promise.resolve({state: Notification.permission}) : origQuery(p));
  }
  try {
    const gp = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function(p) {
      if (p === 37445) return 'Intel Inc.';                 // UNMASKED_VENDOR_WEBGL
      if (p === 37446) return 'Intel Iris OpenGL Engine';    // UNMASKED_RENDERER_WEBGL
      return gp.apply(this, [p]);
    };
  } catch (e) {}
  Object.defineProperty(document, 'hidden', {get: () => false});
  Object.defineProperty(document, 'visibilityState', {get: () => 'visible'});
}
"""

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def _endpoint(port: int) -> str:
    return f"http://127.0.0.1:{port}"


def _log(port: int, action: str, detail) -> None:
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        line = f"{datetime.now().strftime('%H:%M:%S')} [{port}] {action}: {json.dumps(detail)[:400]}\n"
        with open(STATE_DIR / f"actions_{port}.log", "a", encoding="utf-8") as fh:
            fh.write(line)
    except Exception:
        pass


def _connect(port: int):
    from playwright.sync_api import sync_playwright
    p = sync_playwright().start()
    browser = p.chromium.connect_over_cdp(_endpoint(port), timeout=15000)
    ctx = browser.contexts[0] if browser.contexts else browser.new_context()
    try:
        ctx.add_init_script(STEALTH_JS)  # applies to subsequent navigations in this context
    except Exception:
        pass
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    return p, browser, ctx, page


def _with_page(port, fn):
    p, browser, ctx, page = _connect(port)
    try:
        return fn(page, ctx, browser)
    finally:
        browser.close()
        p.stop()


def _human_pause():
    time.sleep(0.3 + random.uniform(0.0, 0.7))


def cmd_start(args):
    from playwright.sync_api import sync_playwright
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    profile = Path(args.profile or (STATE_DIR / f"profile_{args.port}"))
    profile.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        chromium_path = p.chromium.executable_path
    cmd = [
        chromium_path, "--headless=new", f"--remote-debugging-port={args.port}",
        f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
        "--disable-blink-features=AutomationControlled", "--disable-features=IsolateOrigins,site-per-process",
        "--disable-dev-shm-usage", "--lang=en-US", "--window-size=1440,900",
        f"--user-agent={UA}", "about:blank",
    ]
    logf = open(STATE_DIR / f"chromium_{args.port}.log", "w")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=logf, start_new_session=True)
    import urllib.request
    for _ in range(40):
        try:
            urllib.request.urlopen(_endpoint(args.port) + "/json/version", timeout=1)
            res = {"ok": True, "port": args.port, "pid": proc.pid, "profile": str(profile)}
            _log(args.port, "start", res)
            return res
        except Exception:
            time.sleep(0.5)
    return {"ok": False, "error": "CDP endpoint did not come up"}


def cmd_goto(args):
    def f(page, ctx, b):
        try:
            ctx.add_init_script(STEALTH_JS)
        except Exception:
            pass
        r = page.goto(args.url, timeout=60000, wait_until="domcontentloaded")
        page.wait_for_timeout(args.settle)
        return {"ok": True, "status": (r.status if r else None), "url": page.url, "title": page.title()[:120]}
    return _with_page(args.port, f)


def cmd_shot(args):
    def f(page, ctx, b):
        page.screenshot(path=args.path, full_page=args.full)
        return {"ok": True, "path": args.path, "url": page.url}
    return _with_page(args.port, f)


def cmd_text(args):
    def f(page, ctx, b):
        t = page.inner_text(args.selector) if args.selector else page.inner_text("body")
        return {"ok": True, "text": t[:4000]}
    return _with_page(args.port, f)


def cmd_html(args):
    def f(page, ctx, b):
        if args.selector:
            el = page.query_selector(args.selector)
            h = el.evaluate("e=>e.outerHTML") if el else ""
        else:
            h = page.content()
        return {"ok": True, "html": h[:6000]}
    return _with_page(args.port, f)


def cmd_click(args):
    def f(page, ctx, b):
        el = page.query_selector(args.selector)
        if el:
            box = el.bounding_box()
            if box:
                page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, steps=8)
                _human_pause()
        page.click(args.selector, timeout=15000)
        page.wait_for_timeout(args.settle)
        return {"ok": True, "url": page.url}
    return _with_page(args.port, f)


def cmd_clickxy(args):
    def f(page, ctx, b):
        page.mouse.move(max(0, args.x - 40), max(0, args.y - 30), steps=6)
        _human_pause()
        page.mouse.move(args.x, args.y, steps=6)
        page.mouse.click(args.x, args.y)
        page.wait_for_timeout(args.settle)
        return {"ok": True, "at": [args.x, args.y]}
    return _with_page(args.port, f)


def cmd_fill(args):
    def f(page, ctx, b):
        page.click(args.selector, timeout=15000)
        _human_pause()
        page.fill(args.selector, "")
        page.type(args.selector, args.value, delay=random.randint(40, 110))
        return {"ok": True}
    return _with_page(args.port, f)


def cmd_press(args):
    def f(page, ctx, b):
        page.keyboard.press(args.key)
        page.wait_for_timeout(args.settle)
        return {"ok": True}
    return _with_page(args.port, f)


def cmd_eval(args):
    def f(page, ctx, b):
        return {"ok": True, "result": page.evaluate(args.js)}
    try:
        return _with_page(args.port, f)
    except Exception as e:
        return {"ok": False, "error": repr(e)[:300]}


def cmd_waitfor(args):
    def f(page, ctx, b):
        page.wait_for_selector(args.selector, timeout=args.ms)
        return {"ok": True}
    try:
        return _with_page(args.port, f)
    except Exception as e:
        return {"ok": False, "error": repr(e)[:200]}


def cmd_download(args):
    def f(page, ctx, b):
        Path(args.destdir).mkdir(parents=True, exist_ok=True)
        with page.expect_download(timeout=60000) as di:
            page.click(args.selector, timeout=15000)
        d = di.value
        dest = Path(args.destdir) / d.suggested_filename
        d.save_as(str(dest))
        return {"ok": True, "file": str(dest), "name": d.suggested_filename}
    try:
        return _with_page(args.port, f)
    except Exception as e:
        return {"ok": False, "error": repr(e)[:300]}


def cmd_cookies(args):
    return _with_page(args.port, lambda page, ctx, b: {"ok": True, "cookies": ctx.cookies()})


def cmd_stop(args):
    try:
        _with_page(args.port, lambda page, ctx, b: b.close())
    except Exception:
        pass
    return {"ok": True, "stopped": args.port}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9222)
    ap.add_argument("--settle", type=int, default=1500)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("start").add_argument("--profile", default=None)
    sub.add_parser("goto").add_argument("url")
    s = sub.add_parser("shot"); s.add_argument("path"); s.add_argument("--full", action="store_true")
    sub.add_parser("text").add_argument("selector", nargs="?")
    sub.add_parser("html").add_argument("selector", nargs="?")
    sub.add_parser("click").add_argument("selector")
    c = sub.add_parser("clickxy"); c.add_argument("x", type=int); c.add_argument("y", type=int)
    fl = sub.add_parser("fill"); fl.add_argument("selector"); fl.add_argument("value")
    sub.add_parser("press").add_argument("key")
    sub.add_parser("eval").add_argument("js")
    w = sub.add_parser("waitfor"); w.add_argument("selector"); w.add_argument("--ms", type=int, default=15000)
    dl = sub.add_parser("download"); dl.add_argument("selector"); dl.add_argument("destdir")
    sub.add_parser("cookies")
    sub.add_parser("stop")
    args = ap.parse_args()
    if not hasattr(args, "profile"):
        args.profile = None
    fn = {
        "start": cmd_start, "goto": cmd_goto, "shot": cmd_shot, "text": cmd_text, "html": cmd_html,
        "click": cmd_click, "clickxy": cmd_clickxy, "fill": cmd_fill, "press": cmd_press,
        "eval": cmd_eval, "waitfor": cmd_waitfor, "download": cmd_download, "cookies": cmd_cookies,
        "stop": cmd_stop,
    }[args.cmd]
    try:
        res = fn(args)
    except Exception as e:
        res = {"ok": False, "error": repr(e)[:300]}
    if args.cmd not in ("start",):
        # compact log line (omit big text/html/cookies payloads)
        detail = {k: v for k, v in res.items() if k not in ("text", "html", "cookies")}
        _log(args.port, args.cmd, detail)
    print(json.dumps(res))
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
