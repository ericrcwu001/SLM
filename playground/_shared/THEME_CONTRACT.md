# Shared Design-System Contract — light + dark, applied to all 3 sites

Goal: the Modal app (`webapp/static/index.html` + `best-of-n.html`), the dataset
explainer (`dataset_pipeline_explainer.html`), and the eval harness
(`evaluation-harness.html`) must read as **one product** in **both light and dark
modes**, with a **theme toggle** in a shared nav. Dark is the default; the toggle
persists to `localStorage` and there is a no-flash head snippet.

Portability rule: every site is a self-contained file (the webapp copies too). So
this block is **inlined verbatim into each file** — no shared external CSS/JS.
Duplication is intentional; do not introduce cross-file `<link>`/`<script>` deps.

Unify fonts on **Inter** (sans) + **JetBrains Mono** (mono). Drop Geist/Georgia/Avenir.

---

## 1. No-flash head snippet — paste as the FIRST thing in `<head>` (before any CSS)

```html
<script>
  // Set the theme before first paint to avoid a flash.
  (function () {
    try {
      var t = localStorage.getItem("chroma-theme");
      if (!t) t = matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", t);
    } catch (e) { document.documentElement.setAttribute("data-theme", "dark"); }
  })();
</script>
```

## 2. Canonical tokens — paste into each file's CSS, BEFORE that file's own `:root`

```css
:root{
  color-scheme: dark;
  /* Surfaces */
  --c-bg:#0A0B0D; --c-surface-1:#101216; --c-surface-2:#16181D; --c-surface-3:#1D2026; --c-inset:#07080A;
  --c-line:rgba(255,255,255,0.08); --c-line-strong:rgba(255,255,255,0.15);
  /* Text */
  --c-text:#F3F5F8; --c-text-2:#A9B1BD; --c-text-3:#7F8999;
  /* Brand accent (gold) */
  --c-accent:#E8A860; --c-accent-fill:#E8A860; --c-accent-2:#F3BE7C; --c-accent-soft:rgba(232,168,96,0.12); --c-on-accent:#1A130A;
  /* Data-encoding palette (charts, tiers) — consistent meaning everywhere */
  --c-gold:#E8A860;   /* positive / gold-tier / supported / OUR MODEL */
  --c-teal:#5CC7BE;   /* secondary series */
  --c-steel:#7FA0B7;  /* neutral / diagnostic */
  --c-coral:#E06B6B;  /* negative / rejected / refuse / fail */
  --c-mint:#86D6A8;   /* alt positive / pass */
  --c-amber:#E0B14E;  /* warn */
  --c-shadow:rgba(0,0,0,0.55); --c-backdrop:rgba(5,6,8,0.72);
  /* Type */
  --c-sans:"Inter",system-ui,-apple-system,"Segoe UI",sans-serif;
  --c-mono:"JetBrains Mono",ui-monospace,"SF Mono",Menlo,monospace;
}
:root[data-theme="light"]{
  color-scheme: light;
  --c-bg:#F3EFE7; --c-surface-1:#FBFAF5; --c-surface-2:#FFFFFF; --c-surface-3:#EFEBE1; --c-inset:#ECE7DB;
  --c-line:rgba(23,27,27,0.12); --c-line-strong:rgba(23,27,27,0.24);
  --c-text:#171B1B; --c-text-2:#4A524F; --c-text-3:#6E7572;
  --c-accent:#9A6512; --c-accent-fill:#C98A2E; --c-accent-2:#B4791F; --c-accent-soft:rgba(154,101,18,0.12); --c-on-accent:#1A130A;
  --c-gold:#9A6512; --c-teal:#2E8F86; --c-steel:#4E7288; --c-coral:#C0504A; --c-mint:#357D59; --c-amber:#B9832A;
  --c-shadow:rgba(32,36,32,0.16); --c-backdrop:rgba(40,40,35,0.35);
}
```

Rule for missing elevation steps: derive with `color-mix()` from the nearest canonical
surface, e.g. `color-mix(in srgb, var(--c-surface-3) 82%, var(--c-text) 6%)`. Never
hardcode a hex outside these tokens (except inside data-viz gradients that reference the
palette tokens).

## 3. Per-site alias maps — re-point each file's EXISTING `:root` vars to canonical

Keep each file's variable NAMES (so component CSS is untouched); only change their VALUES
to `var(--c-*)`. This is what makes every component theme-aware at once.

### 3a. `webapp/static/styles.css` (app + best-of-n)
```
--bg-0→var(--c-bg)  --bg-1→var(--c-surface-1)  --surface→var(--c-surface-2)
--surface-2→var(--c-surface-3)  --surface-3→color-mix(in srgb,var(--c-surface-3) 80%,var(--c-text) 7%)
--surface-inset→var(--c-inset)  --border→var(--c-line-strong)  --border-strong→var(--c-line-strong)
--border-subtle→var(--c-line)  --text-primary→var(--c-text)  --text-secondary→var(--c-text-2)
--text-tertiary→var(--c-text-3)  --accent→var(--c-accent)  --accent-hover→var(--c-accent-2)
--accent-press→var(--c-accent) [light: deepen]  --accent-contrast→var(--c-on-accent)
--accent-wash→var(--c-accent-soft)  --teal→var(--c-teal)  --success→var(--c-mint)
--warn→var(--c-amber)  --error→var(--c-coral)  --backdrop→var(--c-backdrop)
--image-well→var(--c-inset)  --font-sans→var(--c-sans)  --font-mono→var(--c-mono)
```
Also change `color-scheme: dark;` → remove (canonical sets it) and set body/shadow colors to tokens.

### 3b. `dataset_pipeline_explainer.html`
```
--bg→var(--c-bg)  --bg-1→var(--c-surface-1)  --bg-2→var(--c-surface-2)  --bg-3→var(--c-surface-3)
--line→var(--c-line)  --line-2→var(--c-line-strong)  --text→var(--c-text)  --text-2→var(--c-text-2)
--text-3→var(--c-text-3)  --accent→var(--c-accent)  --accent-2→var(--c-accent-2)
--accent-soft→var(--c-accent-soft)  --on-accent→var(--c-on-accent)
--gold→var(--c-gold)  --diag→var(--c-steel)  --reject→var(--c-coral)
```
Font-family "Geist"/"Geist Mono" → `var(--c-sans)`/`var(--c-mono)`; drop the Geist `<link>`, add Inter+JetBrains Mono link (see §5). Fix the hardcoded `#0A0B0D` in nav/theme-color and the `background:rgba(10,11,13,.62)` nav to token-based (use `color-mix`).

### 3c. `evaluation-harness.html`  (this one is currently LIGHT paper — the biggest change)
```
--paper→var(--c-bg)  --paper-bright→var(--c-surface-1)  --ink→var(--c-text)  --muted→var(--c-text-2)
--rule→var(--c-line)  --charcoal→var(--c-surface-2)  --charcoal-2→var(--c-surface-3)
--mint→var(--c-mint)  --mint-deep→var(--c-mint)  --coral→var(--c-coral)  --coral-soft→var(--c-accent-soft)
--amber→var(--c-amber)  --blue→var(--c-steel)  --serif→var(--c-sans)  --sans→var(--c-sans)  --mono→var(--c-mono)
```
IMPORTANT: sections using `.section-dark` (dark-on-charcoal) must still contrast — since
`--charcoal` now flips with theme, verify text on `.section-dark` uses `--c-text`, not a
hardcoded light color. Replace the paper grid-line background and noise overlay opacities so
they read in both modes (use `--c-line` for grid lines). Replace Georgia serif display type
with Inter; keep a heavier weight for display headings.

## 4. Shared nav + footer + toggle (paste into each file; keep each site's brand label)

Nav markup (adapt the brand label per site; keep the SAME links + toggle):
```html
<nav class="xnav" aria-label="Primary">
  <a class="xnav-brand" href="/"><span class="xnav-dot" aria-hidden="true"></span>Chroma <small>SITE_LABEL</small></a>
  <div class="xnav-links">
    <a href="/">App</a>
    <a href="/best-of-n.html">Best-of-N</a>
    <a href="/dataset.html">Dataset</a>
    <a href="/eval.html">Eval</a>
    <a href="https://github.com/ericrcwu001/SLM" target="_blank" rel="noopener">GitHub</a>
    <button class="xtheme" type="button" aria-label="Toggle light/dark theme" aria-pressed="false">
      <svg class="xtheme-sun" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>
      <svg class="xtheme-moon" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.8A9 9 0 1111.2 3a7 7 0 009.8 9.8z"/></svg>
    </button>
  </div>
</nav>
```
`SITE_LABEL`: "/ app" (index), "/ best of N" (best-of-n), "/ dataset" (explainer), "/ eval harness" (eval). Mark the current page's link with `aria-current="page"`.

Nav + toggle CSS (uses only canonical tokens):
```css
.xnav{position:sticky;top:0;z-index:100;display:flex;align-items:center;justify-content:space-between;
  gap:20px;height:60px;padding:0 clamp(16px,4vw,40px);background:color-mix(in srgb,var(--c-bg) 78%,transparent);
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border-bottom:1px solid var(--c-line);}
.xnav-brand{display:flex;align-items:center;gap:9px;font:600 15px/1 var(--c-sans);color:var(--c-text);letter-spacing:-0.01em;text-decoration:none;}
.xnav-brand small{color:var(--c-text-3);font-weight:400;font-size:12px;}
.xnav-dot{width:20px;height:20px;border-radius:6px;background:conic-gradient(from 210deg,var(--c-coral),var(--c-gold),var(--c-teal),var(--c-gold),var(--c-coral));box-shadow:inset 0 0 0 1px var(--c-line-strong);}
.xnav-links{display:flex;align-items:center;gap:22px;}
.xnav-links a{font:500 13.5px/1 var(--c-sans);color:var(--c-text-2);text-decoration:none;transition:color .18s;}
.xnav-links a:hover{color:var(--c-text);}
.xnav-links a[aria-current="page"]{color:var(--c-accent);}
.xtheme{display:inline-grid;place-items:center;width:34px;height:34px;border-radius:8px;border:1px solid var(--c-line-strong);
  background:var(--c-surface-2);color:var(--c-text-2);cursor:pointer;transition:color .18s,border-color .18s;}
.xtheme:hover{color:var(--c-accent);border-color:var(--c-accent);}
.xtheme-sun{display:none;} :root[data-theme="light"] .xtheme-moon{display:none;}
:root[data-theme="light"] .xtheme-sun{display:block;}
@media(max-width:720px){.xnav-links{gap:14px;} .xnav-links a{font-size:12.5px;}}
```

Footer markup + CSS:
```html
<footer class="xfooter">
  <div class="xfooter-in">
    <span>Chroma · prompt → LUT</span>
    <span class="xfooter-links"><a href="/dataset.html">Dataset methodology</a><a href="/eval.html">Eval harness</a><a href="https://github.com/ericrcwu001/SLM" target="_blank" rel="noopener">GitHub</a></span>
  </div>
</footer>
```
```css
.xfooter{border-top:1px solid var(--c-line);background:var(--c-surface-1);margin-top:64px;}
.xfooter-in{max-width:1180px;margin:0 auto;padding:28px clamp(16px,4vw,40px);display:flex;justify-content:space-between;gap:16px;flex-wrap:wrap;color:var(--c-text-3);font:400 13px/1.5 var(--c-sans);}
.xfooter-links{display:flex;gap:18px;} .xfooter-links a{color:var(--c-text-2);text-decoration:none;} .xfooter-links a:hover{color:var(--c-accent);}
```

Toggle JS (paste before `</body>`):
```html
<script>
  (function () {
    var root = document.documentElement;
    function sync(){ document.querySelectorAll(".xtheme").forEach(function(b){ b.setAttribute("aria-pressed", root.getAttribute("data-theme")==="light"); }); }
    document.addEventListener("click", function (e) {
      var btn = e.target.closest(".xtheme"); if (!btn) return;
      var next = root.getAttribute("data-theme")==="light" ? "dark" : "light";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("chroma-theme", next); } catch (e) {}
      sync();
    });
    sync();
  })();
</script>
```

## 5. Font link (replace each file's existing font `<link>`s)
```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```

## 5b. Accent has TWO roles — do not conflate (fixes a light-mode contrast bug)
`--c-accent` and `--c-accent-fill` diverge in light mode on purpose:
- `--c-accent` (light `#9A6512`, dark `#E8A860`) — for TEXT, icons, links, thin strokes/borders, `aria-current`. Readable ON the page background.
- `--c-accent-fill` (light `#C98A2E`, dark `#E8A860`) — for any SOLID accent BACKGROUND (primary buttons, filled badges/pills, skip-link, filled chart bars/segments, progress fills), always paired with `--c-on-accent` (`#1A130A`) text.

Rule: audit every site for elements that put text/icons on a SOLID accent background (`.btn-primary`, `.badge-pass`, `.skip-link`, filled segment/quota/progress bars, the play button, gate ticks on a fill, etc.). Their `background` must be `var(--c-accent-fill)` (NOT `var(--c-accent)`/`var(--accent)`), with `color:var(--c-on-accent)`. This keeps AA contrast in BOTH modes (light fill #C98A2E on #1A130A ≈ 6:1; dark fill #E8A860 on #1A130A ≈ 8:1). Accent used as text/stroke stays `--c-accent`. Because the per-site alias maps point `--accent → var(--c-accent)`, this means switching the FILL declarations specifically to `--c-accent-fill` — the alias stays for the text/stroke uses.

## 6. Acceptance checks (every site, both modes)
- Toggle flips the whole page; choice persists on reload; no flash on load.
- No hardcoded page/text/border hex remain outside the canonical `--c-*` tokens (data-viz gradients may reference palette tokens).
- Text/!bg contrast ≥ 4.5:1 for body in both modes; nav + footer identical across sites.
- `prefers-reduced-motion` still respected. Existing interactions still work.
