# 04 — Frontend: UI/UX + Design System

Status: Build spec (prescriptive). Target implementer: **ChatGPT Codex + computer-use, one
overnight pass**. Scope: the single-page static frontend of the LOCAL prompt→LUT demo, served by
the FastAPI backend. Companion docs: `01`–`03` (backend/pipeline/API), `05` (backend wiring),
`06_reference_images.md` (the 6 neutral demo photos). Grounding for the color vocabulary:
`eval/tag_vocabulary.py`, `docs/attribute_spec.md` §5 (routes) + §10 (tags), `docs/behavior_spec.md`.

> **North star.** This is a *pro color-grading tool*, not a landing page. It must read as
> something a colorist would trust: quiet, dark, precise, typographically confident. The failure
> mode to avoid is "generic AI-generated web app" — centered purple gradient, one giant card,
> emoji headings, rounded-everything, Bootstrap spacing. Every rule below exists to prevent that.
> If a choice is between "safe/default" and "intentional", pick intentional.

---

## 0. Non-negotiables (read first)

1. **No build step.** Three hand-written files only: `webapp/static/index.html`,
   `webapp/static/styles.css`, `webapp/static/app.js`. Vanilla JS (ES2020, modules OK via
   `<script type="module">`), hand-crafted CSS with custom properties. **No** React/Vue/Tailwind/
   Bootstrap/CSS framework, **no** bundler, **no** npm install. Fonts load from Google Fonts CDN;
   everything else is local.
2. **Dark theme only** (this is a cinema tool; a light theme is out of scope for the demo).
3. **Design tokens are law.** Every color, size, radius, shadow, and duration comes from a CSS
   custom property defined in `:root`. No raw hex or px literals scattered in component rules
   (spacing utilities may reference the scale). This is what makes it look designed, not assembled.
4. **Accessibility is not optional.** All body text ≥ 4.5:1 contrast on its background; large text
   and UI affordances ≥ 3:1. Every interactive element is keyboard-reachable with a visible focus
   ring. Respect `prefers-reduced-motion`.
5. **The image is the hero.** Chrome recedes; graded photographs are the brightest, largest,
   most saturated things on screen. Surfaces are near-neutral so they never fight the grade.
6. **Render by `route`.** The three backend routes (`grade` / `clarify` / `refuse`) are three
   distinct, deliberately-styled result states — not one state with an error banner bolted on.

---

## 1. Design system

### 1.1 Design language / mood

Reference points: DaVinci Resolve's color page, Frame.io, Linear's dark surfaces, Arri/Kodak
film-stock packaging. Characteristics to reproduce:

- **Near-black, layered background** with barely-there elevation steps (not flat #000, not gray).
- **A single warm accent** — amber/gold, evoking a warm color grade and film — used sparingly:
  the primary action, active states, focus. A **cool teal** is the *secondary* accent, reserved
  for "magnitude" affordances so the two prompt-feedback groups read as warm vs cool at a glance.
- **Hairline borders** (1px, low-contrast) do the structural work; shadows are subtle and low-key,
  never the puffy drop-shadows of default component libraries.
- **Confident type hierarchy**: a few sizes, strong weight contrast, generous line-height on body,
  tight tracking on large headings. Mono for anything machine-flavored (terms, the attribute spec,
  `.cube` filename, hue angles).

### 1.2 Color tokens

Define in `:root`. Values are tuned for contrast on the dark stack; do not "brighten" surfaces.

```css
:root{
  /* ---- Background layers (deepest → raised) ---- */
  --bg-0:        #0A0B0D;   /* app canvas, body background */
  --bg-1:        #101216;   /* panel base (input/results panels) */
  --surface:     #16181D;   /* cards (preview cards, popover, chips container) */
  --surface-2:   #1D2026;   /* raised: hover, inputs, chip default */
  --surface-3:   #24272F;   /* raised-more: chip hover, slider handle track */

  /* ---- Borders / dividers (hairlines) ---- */
  --border:        #262A31;      /* default 1px hairline */
  --border-strong: #343A44;      /* emphasized / hovered edge */
  --border-subtle: rgba(255,255,255,0.06);

  /* ---- Text ---- */
  --text-primary:   #F3F5F8;   /* headings, key values  (≥ 13:1 on bg-1) */
  --text-secondary: #A9B1BD;   /* body copy, labels      (≥ 6:1 on bg-1) */
  --text-tertiary:  #6B7482;   /* captions, placeholders (≥ 4.5:1 on surface) */
  --text-disabled:  #454B55;

  /* ---- Accent (warm / primary) ---- */
  --accent:        #E8A860;   /* primary action, active, focus */
  --accent-hover:  #F3BE7C;
  --accent-press:  #D2924A;
  --accent-contrast:#1A130A;  /* text/icon ON the accent fill */
  --accent-wash:   rgba(232,168,96,0.12);   /* subtle fills, active chip bg */
  --accent-glow:   rgba(232,168,96,0.28);   /* focus ring / button glow */

  /* ---- Secondary accent (cool / "magnitude") ---- */
  --teal:          #5CC7BE;
  --teal-wash:     rgba(92,199,190,0.12);

  /* ---- Status ---- */
  --success:       #57C08A;
  --success-wash:  rgba(87,192,138,0.12);
  --warn:          #E0B14E;
  --warn-wash:     rgba(224,177,78,0.12);
  --error:         #E06B6B;
  --error-wash:    rgba(224,107,107,0.12);

  /* ---- Focus ring (single source) ---- */
  --ring: 0 0 0 2px var(--bg-0), 0 0 0 4px var(--accent-glow);
}
```

Semantic aliases (use these in component rules, not the raw ones above, where meaning matters):
`--panel-bg: var(--bg-1)`, `--card-bg: var(--surface)`, `--input-bg: var(--surface-2)`.

### 1.3 Typography

Load from Google Fonts CDN in `<head>` (preconnect + one stylesheet link). Primary sans **Inter**;
mono **JetBrains Mono**. (Geist is an acceptable substitute for Inter if preferred; if so, self-note
it. Do not mix both sans.)

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
```

```css
:root{
  --font-sans: "Inter", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;

  /* Type scale (1.25 modular-ish, hand-tuned). font-size / line-height / tracking */
  --fs-display: 2.25rem;   /* 36px  — app name        | lh 1.1  | ls -0.02em */
  --fs-h1:      1.5rem;    /* 24px  — section titles   | lh 1.2  | ls -0.01em */
  --fs-h2:      1.125rem;  /* 18px  — card/panel heads | lh 1.3  | ls -0.005em */
  --fs-body:    0.9375rem; /* 15px  — body copy        | lh 1.6  */
  --fs-label:   0.8125rem; /* 13px  — labels, buttons  | lh 1.4  | ls  0.01em */
  --fs-small:   0.75rem;   /* 12px  — captions         | lh 1.4  */
  --fs-mono:    0.8125rem; /* 13px  — terms, spec text | lh 1.5  */

  --fw-regular: 400; --fw-medium: 500; --fw-semibold: 600; --fw-bold: 700;
}
```

Rules:
- Body copy uses `--text-secondary`; only headings, key numbers, and active labels use
  `--text-primary`. This two-tone approach is most of the "designed" feeling.
- Uppercase micro-labels (e.g. section eyebrows like `INPUT`, `RESULT`) use `--fs-small`,
  `--fw-semibold`, `letter-spacing: 0.08em`, `--text-tertiary`. Use sparingly (one per section).
- Mono is *only* for machine artifacts: suggested terms, the `attribute_spec_text`, the `.cube`
  filename, hue angles (`210°`), and numeric deltas. Never for prose.
- Never justify text. Max line length for any paragraph ~68ch (`max-width: 62ch`).

### 1.4 Spacing scale

4px base. Use tokens; do not invent one-off margins.

```css
:root{
  --space-1: 4px;  --space-2: 8px;  --space-3: 12px; --space-4: 16px;
  --space-5: 24px; --space-6: 32px; --space-7: 48px; --space-8: 64px;
  --page-pad: clamp(16px, 4vw, 48px);   /* horizontal page gutter */
  --panel-pad: var(--space-6);          /* interior padding of panels */
  --stack: var(--space-4);              /* default vertical rhythm */
}
```

Intentional spacing (anti-slop): panels breathe (32px interior), related items sit close
(8–12px), unrelated groups are separated generously (24–48px). Avoid uniform 16px everywhere —
uniform spacing is the tell of an unstyled layout.

### 1.5 Radii, borders, shadows, motion

```css
:root{
  --r-sm: 6px;   --r-md: 10px;  --r-lg: 14px;  --r-xl: 20px;  --r-full: 999px;

  /* Low-key elevation on dark — subtle, never puffy */
  --shadow-sm: 0 1px 2px rgba(0,0,0,0.40);
  --shadow-md: 0 4px 16px rgba(0,0,0,0.45);
  --shadow-lg: 0 16px 48px rgba(0,0,0,0.55);
  --shadow-pop: 0 12px 32px rgba(0,0,0,0.60);   /* popovers/menus */

  /* Motion */
  --dur-fast: 120ms; --dur-med: 180ms; --dur-slow: 260ms;
  --ease: cubic-bezier(0.2, 0.8, 0.2, 1);       /* standard */
  --ease-out: cubic-bezier(0.16, 1, 0.3, 1);    /* entrances */
}
@media (prefers-reduced-motion: reduce){
  *,*::before,*::after{ animation-duration:0.001ms!important; transition-duration:0.001ms!important; }
}
```

Component conventions:
- Cards: `background: var(--card-bg); border:1px solid var(--border); border-radius: var(--r-lg)`.
  Hover raises to `--surface-2` + `--border-strong`, transition `--dur-fast`.
- Radii by scale: chips/inputs `--r-md`, cards/panels `--r-lg`, buttons `--r-md`, pills `--r-full`.
  Do not round everything to the same big radius (another slop tell).
- Focus: every focusable element gets `outline: none; box-shadow: var(--ring)` on
  `:focus-visible`. Never remove focus without replacing it.
- Motion is functional only: fades/rises on mount (`translateY(6px)→0`, opacity), color/scale on
  hover. No parallax, no infinite loops, no bounce.

### 1.6 Iconography

Use inline SVG (stroked, 1.5px, `currentColor`, 20×20 viewBox) — upload/cloud, download, copy,
info (ⓘ), close (×), warning triangle. No emoji anywhere in the UI chrome. Keep a tiny set (~6);
define them once as `<symbol>` in an inline `<svg style="display:none">` sprite and `<use>` them.

---

## 2. Layout

### 2.1 Page frame

Single column, centered content, max width **1120px**, page gutter `--page-pad`. Two stacked
panels — **Input** then **Results** — with a header above. On wide screens the results grid uses
its width; the layout does **not** become a two-column app (keep the "compose → reveal" narrative
vertical, like submitting a render and watching it come back).

```
┌───────────────────────────────────────────────────────────────────────────┐
│  [◆] Chroma            prompt → LUT color grading                    ⓘ Terms │   header (sticky, hairline bottom)
├───────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT                                                                     │
│   ┌───────────────────────────────┐   ┌───────────────────────────────┐    │
│   │                               │   │  Describe the look             │    │
│   │     ⤒  Drop an image          │   │  ┌─────────────────────────┐   │    │
│   │        or click to browse     │   │  │ e.g. "warm cinematic     │   │    │
│   │     JPG / PNG · up to 12MB    │   │  │  teal-orange, lifted…"    │   │    │
│   │   (becomes a thumbnail once   │   │  └─────────────────────────┘   │    │
│   │    a file is chosen)          │   │                    [ Generate LUT →]│    │
│   └───────────────────────────────┘   └───────────────────────────────┘    │
│         upload zone (≈5/12)                 prompt column (≈7/12)           │
│                                                                             │
│   ─────────────────────────────────────────────────────────────────────    │
│                                                                             │
│   RESULT                                                                    │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │                    BEFORE / AFTER split slider                     │    │
│   │        (user's image; drag handle reveals graded vs original)      │    │
│   │                                                     [⇩ Download .cube]│    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│   Applied to reference shots                                                │
│   ┌────────┐ ┌────────┐ ┌────────┐   (responsive grid, 3×2 → 2 → 1)         │
│   │ City   │ │Landscape│ │Portrait│                                          │
│   └────────┘ └────────┘ └────────┘                                          │
│   ┌────────┐ ┌────────┐ ┌────────┐                                          │
│   │Close-up│ │ Food   │ │Interior│                                          │
│   └────────┘ └────────┘ └────────┘                                          │
│                                                                             │
│   ┌───────────────────────────────────────────────────────────────────┐    │
│   │  Sharpen your prompt                                               │    │
│   │  "Clear direction, but the strength is ambiguous."   ← assessment   │    │
│   │                                                                     │    │
│   │  Add magnitude   [subtle] [moderate] [heavy]  ← teal chips          │    │
│   │  Clarify direction [warmer] [lifted blacks] [teal-orange] ← amber   │    │
│   │  Refine style    [matte] [filmic] [bleach bypass]                   │    │
│   └───────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└───────────────────────────────────────────────────────────────────────────┘
```

Grid: use CSS Grid for the input row (`grid-template-columns: 5fr 7fr; gap: var(--space-5)`,
collapsing to one column below 860px) and for the reference grid (see §3.4). Everything else is
flow layout with the spacing scale.

### 2.2 Responsive breakpoints

- `≥ 1024px` (desktop): reference grid 3 columns; input row 5fr/7fr.
- `640–1023px` (tablet): reference grid 2 columns; input row stacks to 1 column.
- `< 640px` (mobile): everything 1 column; header tagline hidden, "Terms" stays; hero slider full
  width; touch targets ≥ 44px; term chips wrap freely.

---

## 3. Components

Each component below lists: purpose, wireframe/structure, styling intent, and states. Build them as
sections in `index.html` with BEM-ish class names (`.panel`, `.upload`, `.upload__hint`,
`.chip`, `.chip--magnitude`, `.popover`, …).

### 3.1 Header

```
[◆ Chroma]   prompt → LUT color grading                              [ ⓘ Terms ]
```

- Sticky top, `background: color-mix(in srgb, var(--bg-0) 82%, transparent)` + `backdrop-filter:
  blur(10px)`, `border-bottom: 1px solid var(--border)`, height ~64px, page-gutter padding.
- Left: a small mark (an inline SVG diamond/aperture glyph filled `--accent`) + wordmark
  **"Chroma"** (`--fs-h2`, `--fw-semibold`, `--text-primary`) + a `·`-separated tagline
  `prompt → LUT color grading` (`--fs-label`, `--text-tertiary`, hidden < 640px).
- Right: a ghost **"Terms"** button (opens the full glossary from `/api/terms` in a drawer/modal —
  §3.6). This is the app's name/tagline requirement; keep it understated.

### 3.2 Input panel — upload zone

Structure: a `<label>`-wrapped drop zone containing a hidden `<input type="file" accept="image/*">`,
so click and keyboard both work for free.

```
┌───────────────────────────────┐        (empty)                 ┌───────────────────────────────┐
│                               │                                 │  ┌─────┐  street_02.jpg        │  (filled)
│        ⤒                      │                                 │  │thumb│  1.8 MB · 3024×4032   │
│   Drop an image here          │                                 │  └─────┘  [ Replace ]  [ × ]   │
│   or click to browse          │                                 │                                 │
│   JPG · PNG · up to 12 MB     │                                 └───────────────────────────────┘
└───────────────────────────────┘
```

- Empty: dashed 1.5px `--border-strong` border, `--r-lg`, min-height ~220px, centered stack (upload
  icon `--text-tertiary`, primary line `--text-secondary`, hint `--text-tertiary` `--fs-small`).
- Hover / keyboard focus: border → `--accent`, faint `--accent-wash` fill, icon → `--accent`.
- **Dragover**: border solid `--accent`, `--accent-wash` fill, subtle scale(1.005); toggled via a
  `.is-dragover` class on `dragenter/dragover`, removed on `dragleave/drop`. `preventDefault` on
  both `dragover` and `drop`.
- Filled: replace the prompt with a compact row — a rounded thumbnail (object-fit: cover, ~64px,
  `--r-md`), filename (mono, truncated with ellipsis), size + dimensions (`--fs-small`,
  `--text-tertiary`), a **Replace** ghost button and a **×** icon button to clear.
- Client-side validation: reject non-images and > 12 MB with an inline `--error` helper line under
  the zone (do not use `alert()`). Read dimensions via an `Image()`/`createObjectURL` for the meta.

### 3.3 Input panel — prompt + generate

```
Describe the look
┌───────────────────────────────────────────────┐
│ warm cinematic teal-orange, lifted matte blacks│   ← textarea, auto-grow 3→8 rows
│                                                 │
└───────────────────────────────────────────────┘
Be specific about direction and strength.        0/280      [ Generate LUT → ]
```

- Label (`--fs-label`, `--text-secondary`) above a `<textarea>`: `--input-bg`, 1px `--border`,
  `--r-md`, `--panel-pad/2` padding, `--font-sans`, `--fs-body`, `--text-primary`, placeholder
  `--text-tertiary`. Focus → border `--accent` + `--ring`. Auto-grow height with JS (min 3 rows,
  max ~8 then scroll). `maxlength ≈ 280` with a live count in `--text-tertiary`.
- Helper microcopy under it (`--fs-small`, `--text-tertiary`): *"Be specific about direction and
  strength — the prompt panel below will suggest terms after you generate."*
- **Generate LUT** button (primary): solid `--accent` fill, `--accent-contrast` text,
  `--fw-semibold`, `--r-md`, ~44px tall, right-aligned. Hover → `--accent-hover` + soft
  `box-shadow: 0 0 0 3px var(--accent-glow)`. Active → `--accent-press`. Trailing arrow icon.
  - **Disabled** until both an image and a non-empty prompt exist (reduced opacity, `not-allowed`).
  - **Loading**: label → "Generating…", a small inline spinner replaces the arrow, button
    `aria-busy="true"`, whole input panel `inert`/pointer-events-none. Do not collapse width.

### 3.4 Results — hero before/after slider

The user's own image graded, presented as a **before/after split slider** (previews[0]:
`original_url` = before, `graded_url` = after).

```
┌──────────────────────────────────────────────────────────────┐
│  before  ◀   ┃   ▶  after                       [ ⇩ Download .cube ]│
│               ┃                                                │
│   original    ┃            graded (LUT applied)                │
│               ┃                                                │
│              [⇕] draggable handle                              │
└──────────────────────────────────────────────────────────────┘
```

- Implementation: a positioned container with the **graded** image as the base layer and the
  **original** image in an absolutely-positioned overlay whose `width` (or `clip-path: inset(0 X 0
  0)`) is driven by an `<input type="range" min="0" max="100">` overlaid full-bleed (transparent,
  custom handle). A vertical divider line (`--accent`, 2px) with a round grabber sits at the split.
  Tiny `before`/`after` pills top-left/right (`--fs-small`, mono, `--surface`/blur bg).
- Keyboard: the range input is focusable; arrow keys move the split. Default position 50%.
- The container keeps the user image's aspect ratio (`aspect-ratio` from natural dimensions), max
  height ~min(70vh, 620px), `object-fit: contain`, `--r-lg`, `--shadow-md`, `--border`.
- **Download .cube**: secondary button (outline: 1px `--border-strong`, `--text-primary`, download
  icon) top-right of the hero. Fetches `lut.cube_url`, triggers a download with a sensible filename
  (see app.js §5). Show the filename in mono beneath on hover/focus.

### 3.5 Results — reference preview grid

The 6 neutral reference photos (previews[1..6]), each graded with the same LUT — the actual "demo
of the LUT". See `06_reference_images.md` for the images themselves.

```
Applied to reference shots                                  (eyebrow + one-line caption)
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  [graded img]│  │  [graded img]│  │  [graded img]│
│  City        │  │  Landscape   │  │  Portrait    │   ← label bottom-left, mono/caption
└──────────────┘  └──────────────┘  └──────────────┘
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Close-up    │  │  Food        │  │  Interior    │
└──────────────┘  └──────────────┘  └──────────────┘
```

- Grid: `display:grid; grid-template-columns: repeat(3, 1fr); gap: var(--space-4)` →
  `repeat(2,1fr)` at tablet → `1fr` at mobile. Use `previews[1..]` order but render the label from
  each preview's `name` (fallback to the fixed category list City / Landscape / Portrait /
  Close-up / Food / Interior).
- Card: `--card-bg`, `--border`, `--r-lg`, overflow hidden. Image fills the card top with a fixed
  `aspect-ratio: 4/3`, `object-fit: cover`. A slim footer bar holds the category label
  (`--fs-label`, `--text-secondary`) left and an optional tiny **hover reveal**: on card hover,
  cross-fade to the *original* (ungraded) for ~an instant so the LUT effect is legible (store both
  `graded_url` and `original_url`; swap `src`/opacity on `mouseenter`/`mouseleave`). Keep this
  subtle and optional — it must not flicker.
- Card hover: border → `--border-strong`, translateY(-2px), `--shadow-md`, `--dur-fast`.
- Loading placeholder: skeleton shimmer block at the right aspect ratio (see §4.2).

### 3.6 Prompt-improvement panel (first-class)

This is the research-driven centerpiece: the model routes well but grades *magnitude* weakly for
vague prompts, so we **coach the user toward explicit terms** — never rewriting their prompt for
them. Renders from `prompt_feedback` on a `grade` result, and is the *primary* content on
`clarify`.

Structure:

```
┌───────────────────────────────────────────────────────────────────────────┐
│  Sharpen your prompt                                                        │  h2
│  Clear direction, but the strength is ambiguous.                            │  assessment (secondary)
│                                                                             │
│  ┌ Add magnitude ─────────────────────────────────────────────────┐        │  group (teal accent)
│  │  [ subtle ]  [ moderate ]  [ heavy ]  [ +2 stops ]               │        │  chips
│  └─────────────────────────────────────────────────────────────────┘        │
│  ┌ Clarify direction ─────────────────────────────────────────────┐         │  group (amber accent)
│  │  [ warmer ]  [ lifted blacks ]  [ teal-orange ]                 │         │
│  └─────────────────────────────────────────────────────────────────┘         │
│  ┌ Refine style ──────────────────────────────────────────────────┐         │  group (neutral)
│  │  [ matte ]  [ filmic ]  [ bleach bypass ]                       │         │
│  └─────────────────────────────────────────────────────────────────┘         │
│                                          hover a term for its definition ⓘ   │  hint (tertiary)
└───────────────────────────────────────────────────────────────────────────┘
```

- **Assessment line**: `prompt_feedback.assessment`, `--fs-body`, `--text-secondary`, one sentence.
- **Grouping**: bucket `suggested_terms` by their `axis`/intent into three groups with fixed
  headings and accent coding:
  - **Add magnitude** — teal (`--teal`, `--teal-wash`): intensity/strength words (subtle, moderate,
    heavy, "+2 stops", slight, strong). Group appears only if it has terms.
  - **Clarify direction** — amber (`--accent`, `--accent-wash`): directional color terms (warmer,
    cooler, lifted blacks, teal-orange, greener shadows…).
  - **Refine style** — neutral (`--surface-2`, `--border-strong`): style bundles (matte, filmic,
    cinematic, sepia, bleach bypass). NOTE (per `05_terms.md`): the style *composites* are
    `grounded:false` — the glossary/hover documents them, but the suggestion box surfaces their
    **grounded equivalents** instead (e.g. `lifted_blacks` + `muted` for "matte"). So this group is
    populated only when the backend actually returns a grounded term the bucketing routes here; if
    empty it is omitted (below). Do not hard-code style chips as if they were suggested.
  - Map by the term's `axis` field (the source of truth — each term carries `axis` per `03`/`05`;
    the magnitude bucket has `axis:"(magnitude bucket)"`, Direction/Saturation/Hue axes → the
    direction group, Style composites → style). Fall back to the vocab families in
    `eval/tag_vocabulary.py` only if `axis` is missing. Empty groups are omitted, not shown empty.
- **Only grounded terms render.** Filter to `term.grounded === true`. This is a hard rule: the
  panel must never surface a term the pipeline can't back with a measurable axis
  (`docs/attribute_spec.md` §6 backing rule). If the backend already filters, still assert it
  client-side.
- **Chips**: pill (`--r-full`), mono label (`--font-mono`, `--fs-mono`), `--surface-2` bg, 1px
  border tinted by the group accent, `--text-primary`. Hover: bg → the group's wash, border → the
  group accent, cursor pointer. Focusable (`role="button"`, `tabindex="0"`), Enter/Space activate.
- **Click = copy the term** to the clipboard (`navigator.clipboard.writeText(term.term)`), then a
  brief inline "copied" affordance (chip flashes `--success` border + a check for ~1s, or a small
  toast). Rationale: the user writes their *own* prompt; we hand them precise words, we don't
  compose for them. (Optional nicety: also append the term to the textarea if empty focus — but
  copy is the contract.)
- **Definition popover on hover/focus** (§3.7).

### 3.7 Definition popover (hover card)

On chip hover (and keyboard focus, and long-press on touch), show a popover anchored to the chip
with the term's definition. Content comes from the term object (`definition`, `example_usage`) and,
for the full glossary, from `/api/terms` (`category`, plain + technical `definition`).

```
        ┌────────────────────────────────────────────┐
        │  lifted blacks                     DIRECTION │  term (mono) + axis tag
        │  ───────────────────────────────────────────│
        │  Raise the darkest tones so the image looks  │  plain definition (--text-secondary)
        │  softer and less deep — a faded, filmic feel.│
        │                                              │
        │  Technical: increases black_point_l_delta;   │  technical line (mono, --text-tertiary)
        │  the shadow floor moves up in L*.            │
        │                                              │
        │  e.g. "lifted blacks, matte finish"          │  example_usage (mono, --accent)
        └───────────────────────────────────────────┘▾
```

- Styling: `--surface`, 1px `--border-strong`, `--r-md`, `--shadow-pop`, max-width ~300px, padding
  `--space-4`. A small caret pointing to the chip. Appears with a `--dur-fast` fade+rise; ~120ms
  open delay, closes on mouseleave/blur/Escape.
- Header row: term in mono `--text-primary` + a right-aligned axis/category micro-tag
  (`--fs-small`, uppercase, `--text-tertiary`, wash-tinted by group).
- Body: **plain** definition first (`--text-secondary`, the human explanation), then a
  **Technical** line (mono, `--text-tertiary`) for cinematic/technical terms (black point, matte,
  split-tone, lifted blacks), then **example_usage** (mono, `--accent`) prefixed `e.g.`.
- Positioning: prefer above the chip; flip below if it would clip the viewport top; clamp
  horizontally into the viewport. Implement with a single reusable popover element positioned via
  `getBoundingClientRect()` (do not create one per chip). `role="tooltip"`, linked via
  `aria-describedby`.
- Reduced motion: no rise, instant.

### 3.8 Terms glossary (drawer)

The header **Terms** button opens the complete glossary fetched from `/api/terms`: a right-side
drawer (or centered modal) listing every term grouped by `category`, each with its plain +
technical definition and example. Search/filter box at top (`--input-bg`). This is reference
material; reuse the popover's typographic treatment per row. Close on × / Escape / backdrop click;
trap focus while open; restore focus to the trigger on close.

---

## 4. States

The app is a small state machine. Exactly one **result region** state is visible at a time; the
input panel is always present (disabled while loading). States: `idle → loading → (grade | clarify
| refuse | error)`; re-submitting returns to `loading`.

### 4.1 Idle / empty

- No result region yet. Instead, a quiet **placeholder** below the input: a faint framed area with
  a one-line invitation — *"Upload an image and describe a look to generate a LUT."* — in
  `--text-tertiary`, plus 6 empty reference-card outlines (dashed `--border`) hinting at what will
  fill in. This teaches the layout without shouting. No spinner.

### 4.2 Loading (skeleton, not spinner-only)

- On submit, immediately render the **results scaffold** with skeletons so the layout doesn't jump:
  a hero-sized shimmer block + 6 card-shaped shimmer blocks + a shimmer assessment line.
- Skeleton = `--surface-2` block with a slow left→right highlight sweep
  (`background: linear-gradient(90deg, --surface-2, --surface-3, --surface-2)` animated, ~1.4s,
  disabled under reduced motion → static `--surface-2`).
- The Generate button shows its loading state (§3.3). Keep the user's prompt/image visible.
- Scroll the results region into view smoothly (respect reduced-motion) so the user sees progress.

### 4.3 `grade` (success)

Render: hero before/after (previews[0]) → download .cube → reference grid (previews[1..6]) →
prompt-improvement panel (from `prompt_feedback`). Optionally show the returned
`attribute_spec_text` in a small, collapsible mono "What the model understood" caption under the
hero (`route=grade | warmer=+2.0 matte=+2.5 …`) — a tasteful pro touch, collapsed by default.

### 4.4 `clarify` (not an error — an invitation)

The request has color intent but is under-specified. **Do not fabricate a grade or show previews.**

```
┌───────────────────────────────────────────────────────────────────────────┐
│  ◐  Let's pin down the direction                                            │  calm heading, accent-wash icon
│                                                                             │
│  “Make it pop” could go a few ways. Tell me which and how strong.           │  clarify_message (from API)
│                                                                             │
│  Try adding one of these, then generate again:                             │
│  ┌ Clarify direction ┐  [ warmer ] [ more contrast ] [ teal-orange ]        │  prominent grounded chips
│  ┌ Add magnitude ────┐  [ subtle ] [ moderate ] [ heavy ]                   │  (same chip system, larger)
└───────────────────────────────────────────────────────────────────────────┘
```

- Styling: **calm and inviting**, not red. Panel gets a soft `--accent-wash` top border/glow and a
  half-filled ◐ icon in `--accent`. The `clarify_message` is the hero text (`--fs-h2`,
  `--text-primary`).
- Reuse the prompt-improvement chip system (§3.6) but make chips **more prominent** here (they are
  the primary CTA) and pull them from `prompt_feedback.suggested_terms` (and/or the spec's
  clarify options). Clicking a chip copies it *and* — acceptable here — focuses the prompt textarea,
  because the explicit ask is "re-prompt". A subtle "Generate again ↑" hint points back at the button.
- No previews, no download.

### 4.5 `refuse` (graceful, never a stack-trace)

The look is out of scope (local/semantic/relighting/etc.) or out of gamut (infrared, pure-primary,
big hue rotation). Present it as a **clear, respectful explanation of scope**, not a failure.

```
┌───────────────────────────────────────────────────────────────────────────┐
│  ⃠  This one's outside what a single LUT can do                             │  neutral, not error-red
│                                                                             │
│  <refuse_reason, humanized>                                                 │  e.g. "Removing the background is a
│  A LUT re-maps color globally — it can't select or edit objects, add        │       local edit; a LUT applies the
│  content, or relight a scene.                                               │       same mapping everywhere."
│                                                                             │
│  What it can do:  [ warmer ] [ matte ] [ teal-orange ] [ lifted blacks ]    │  redirect to supported grounded terms
│  Try a global color look instead.                                          │
└───────────────────────────────────────────────────────────────────────────┘
```

- Styling: neutral surface with a **warn**-tinted (not error-red) left accent and a ⃠/slash icon in
  `--warn`. Reserve `--error` strictly for actual failures (§4.6).
- Body: humanize `refuse_reason` (map `out_of_scope` / `out_of_gamut` to friendly copy; fall back to
  a generic scope explanation). Keep the tone "here's the boundary", per `docs/behavior_spec.md`.
- Redirect: a small row of supported grounded terms (from `/api/terms` or the feedback) so the user
  can pivot to something the tool *can* do. No previews, no download.

### 4.6 Error (real failure)

Network error, 5xx, malformed JSON, or timeout. A compact inline card in the results region: `--error`
icon + border, message *"Something went wrong generating the LUT."*, the technical detail in a
collapsible mono line, and a **Retry** button (re-submits the same image+prompt). Never surface a
raw stack trace as the primary message; never use `alert()`. Distinct from `refuse` (which is a
valid, expected outcome).

---

## 5. File structure & implementation notes

### 5.1 `index.html`

Semantic, minimal, ordered exactly as the DOM regions. Skeleton:

```html
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chroma — prompt → LUT</title>
  <!-- Google Fonts preconnect + Inter/JetBrains Mono (see §1.3) -->
  <link rel="stylesheet" href="./styles.css">
</head>
<body>
  <!-- inline SVG sprite (display:none) with <symbol> icons -->
  <header class="header"> … mark + wordmark + tagline + [Terms] button … </header>

  <main class="page">
    <section class="panel input" aria-label="Input">
      <p class="eyebrow">Input</p>
      <div class="input__row">
        <label class="upload" id="upload"> … drop zone + hidden file input + filled state … </label>
        <div class="prompt">
          <label for="prompt-text">Describe the look</label>
          <textarea id="prompt-text" maxlength="280"></textarea>
          <div class="prompt__footer"> helper · count · <button id="generate" class="btn btn--primary">Generate LUT</button></div>
        </div>
      </div>
      <p class="input__error" hidden></p>
    </section>

    <section class="panel results" aria-live="polite" aria-busy="false">
      <!-- JS swaps ONE of: .state-idle | .state-loading | .state-grade | .state-clarify | .state-refuse | .state-error -->
    </section>
  </main>

  <div class="popover" role="tooltip" hidden></div>            <!-- single reusable definition popover -->
  <aside class="drawer" id="terms-drawer" hidden> … glossary … </aside>
  <div class="toast" role="status" aria-live="polite" hidden></div>

  <script type="module" src="./app.js"></script>
</body>
</html>
```

Keep templates for result states as `<template>` elements (or build DOM in JS) — no innerHTML with
untrusted strings; always set text via `textContent`/`createElement` (the prompt and API strings
are user/model-derived — never inject as HTML).

### 5.2 `styles.css`

- Order: (1) `:root` tokens (§1.2–§1.5), (2) a small reset (`box-sizing:border-box`, margin 0,
  `body{background:var(--bg-0);color:var(--text-secondary);font-family:var(--font-sans)}`,
  `img{max-width:100%;display:block}`, `:focus-visible{box-shadow:var(--ring)}`), (3) layout
  (`.page`, `.header`, `.panel`, grids), (4) components (`.upload`, `.prompt`, `.btn`, `.hero`,
  `.ref-grid`, `.ref-card`, `.feedback`, `.chip`, `.popover`, `.drawer`, `.skeleton`, `.toast`),
  (5) state modifiers (`.is-dragover`, `.is-loading`, `.state-*`), (6) `@media` breakpoints,
  (7) `@media (prefers-reduced-motion)`.
- Use tokens exclusively. A quick self-audit: `grep -E '#[0-9a-fA-F]{3,6}' styles.css` should match
  **only** the `:root` block.
- Prefer CSS Grid/flex + `gap`; avoid margin hacks. Use logical spacing from the scale.

### 5.3 `app.js` — flow

```
init():
  1. Fetch GET /api/terms → cache glossary (Map term→meta). Populate the Terms drawer.
     On failure: keep chips working from per-result term objects; log, don't block.
  2. Wire upload: file input change + drag/drop (dragenter/over/leave/drop, preventDefault),
     validate (image, ≤12MB), render thumbnail + meta, keep the File in state, enable-check.
  3. Wire textarea: auto-grow, char count, enable-check (image && prompt.trim()).
  4. Wire the single popover: delegate mouseover/focusin on [data-term]; position via
     getBoundingClientRect; fill from cached glossary or the chip's dataset; hide on out/blur/Esc.
  5. Wire the Terms drawer open/close (focus trap, Esc, backdrop).

onSubmit():
  - guard (image + prompt); set results → loading skeleton; button loading; panel inert.
  - const fd = new FormData(); fd.append('image', file); fd.append('prompt', text);
  - fetch('/api/generate', { method:'POST', body: fd })  // do NOT set Content-Type; browser adds boundary
  - on !res.ok → render error state (with res.status / message); on network throw → error state.
  - const data = await res.json(); switch (data.route) { grade | clarify | refuse }.

render(data):
  grade:
    - hero: previews[0].original_url (before) + graded_url (after) → build split slider.
    - download: bind lut.cube_url → download as `chroma_<slug(prompt)>_<yyyymmdd>.cube`.
    - grid: previews.slice(1) → ref cards (label from name or fixed list; graded_url base,
      original_url for hover reveal). Use loading="lazy".
    - feedback: renderFeedback(prompt_feedback): assessment line + grouped grounded chips.
    - optional: show attribute_spec_text in a collapsible mono caption.
  clarify:
    - show clarify_message as hero text + prominent grounded chips (from suggested_terms);
      chip click → copy + focus textarea. No previews.
  refuse:
    - humanize(refuse_reason) + scope explanation + supported-terms redirect row. No previews.

renderFeedback(fb):
  - filter suggested_terms to grounded===true.
  - bucket by axis/intent → {magnitude, direction, style} (see §3.6). Skip empty groups.
  - build chip per term: mono label, data-term=term, group class; click → copy + toast/flash.

download(url, filename):
  - fetch(url) → blob → object URL → <a download> click → revokeObjectURL.
    (Or, if the backend sets Content-Disposition, a plain anchor href is fine.)

Helpers: slug(str), humanRefuse(reason), enableGenerate(), setResultsState(name),
  showToast(msg), copyTerm(term).
```

Robustness: guard every optional field (`previews ?? []`, `prompt_feedback ?? {}`,
`suggested_terms ?? []`); tolerate a missing `attribute_spec_text`; treat unknown `route` as error.
All fetches are same-origin (served by FastAPI). No third-party JS.

---

## 6. Anti-slop checklist (self-review before you call it done)

- [ ] Not a single centered card on a purple gradient. Layered near-black surfaces, real panels.
- [ ] Type hierarchy is visible at a glance: display wordmark, section eyebrows, two-tone body.
- [ ] Exactly one warm accent (+ one cool secondary for magnitude); accent used sparingly, not on
      every element. No rainbow.
- [ ] Spacing is intentional (tight within groups, generous between) — not uniform 16px.
- [ ] Hairline borders + subtle shadows; nothing looks like a default component library.
- [ ] Radii vary by element scale; not everything is a 24px pill.
- [ ] No emoji in chrome; icons are consistent stroked SVGs.
- [ ] The graded photographs are the brightest, most saturated things on screen.
- [ ] Mono is used only for machine artifacts (terms, spec, filename, angles).
- [ ] `refuse` looks like a calm scope note (warn tint), `clarify` like an invitation (accent),
      `error` like a real failure (error) — three visually distinct treatments.

## 7. Acceptance criteria

Functional:
1. Loads with no console errors; `GET /api/terms` fetched once on load and the Terms drawer is
   populated; if it fails, chips still function from per-result data.
2. Drag-drop **and** click-to-browse both upload; invalid/oversize files rejected inline (no
   `alert`); a thumbnail + filename + size + dimensions render; image can be replaced/cleared.
3. Generate is disabled until image + non-empty prompt exist; on submit it shows a loading state
   and the results region shows a skeleton (not just a spinner).
4. `POST /api/generate` is sent as `multipart/form-data` with fields `image` and `prompt`, and the
   browser sets the boundary (Content-Type not manually set).
5. `route=grade` renders: before/after hero slider (previews[0]), a working **Download .cube**, a
   responsive 6-card reference grid (previews[1..6]) each visibly showing the LUT, and the
   prompt-improvement panel.
6. Prompt-improvement panel: assessment line + grounded terms **only** (`grounded===true`), grouped
   into Add magnitude / Clarify direction / Refine style (empty groups hidden); hovering a chip
   shows a definition popover with plain + technical + example; clicking a chip copies the term.
7. `route=clarify` shows the clarifying message + prominent grounded chips and **no** previews;
   `route=refuse` shows a graceful humanized scope note + supported-term redirect and **no**
   previews; a network/5xx failure shows the distinct error state with Retry.
8. Download produces a `.cube` file with a sensible filename.

Design/quality:
9. All colors/sizes/radii/shadows/durations come from `:root` tokens; `grep` for hex outside
   `:root` returns nothing.
10. Body text ≥ 4.5:1 contrast; every interactive element has a visible `:focus-visible` ring and
    is keyboard-operable (upload, textarea, generate, chips, popovers, drawer, slider).
11. Layout is responsive at ≥1024 / 640–1023 / <640 with the specified grid collapses; no
    horizontal scroll; touch targets ≥ 44px on mobile.
12. `prefers-reduced-motion` disables non-essential animation (skeleton shimmer, rises, hover-reveal).
13. Passes the §6 anti-slop checklist — it reads as a pro color tool, not a generic AI web app.
