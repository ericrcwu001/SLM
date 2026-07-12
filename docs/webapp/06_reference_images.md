# 06 — Reference Images (the 6 neutral demo photos)

Status: Build spec (prescriptive). Target implementer: **ChatGPT Codex + computer-use, one
overnight pass**. Scope: sourcing, preparing, and shipping the six neutral reference photographs
that the app grades with the user's generated LUT to *demonstrate* the look. Companion:
`04_frontend.md` §3.5 (the reference grid) and the API contract (`previews[1..6]`).

> **Why these images exist.** The user uploads their own photo, but one photo can't show what a LUT
> *does* — a warm grade on an already-warm sunset looks like nothing. So we apply the same LUT to a
> **fixed, diverse, roughly-neutral panel** of six shots. Seeing "the same grade" land on skin, sky,
> foliage, food, metal, and interior light at once is what makes the LUT legible. These images are a
> **controlled test chart made of real photographs**.

---

## 1. The six categories (and why this exact spread)

A LUT is a global RGB→RGB map. Its character shows only where there is something to move. The six
categories are chosen so that, collectively, they exercise **every region of color space a global
grade touches**:

| # | Category | File slug | What it must contain | Color-space coverage it proves |
|---|----------|-----------|----------------------|--------------------------------|
| 1 | City / street | `city` | Buildings, sky, some neon/mixed artificial light, asphalt, glass | **Neutrals + mixed white balance**; how the grade treats concrete grays and how casts read on man-made surfaces |
| 2 | Landscape / nature | `landscape` | Sky, foliage/greens, water or distant haze, natural horizon | **Skies (blues) + greens**; the two hues people judge a grade by most; highlight roll-off in the sky |
| 3 | Portrait / person | `portrait` | One person, **visible skin**, natural light, plain-ish background | **Skin tones** — the single most scrutinized region; whether a grade keeps skin plausible or turns it green/orange |
| 4 | Close-up / macro | `closeup` | A saturated subject filling frame (flower, fabric, object), shallow DOF | **Saturated mid-chroma + fine gradients**; chroma push/pull and smoothness (banding/foldover) |
| 5 | Food | `food` | A plated dish, warm and cool ingredients together, tabletop | **Reds/oranges/yellows + appetizing warmth**; how warmth and saturation shifts read on organic color |
| 6 | Interior / architecture | `interior` | A room with windows, wood/wall neutrals, indoor+daylight mix | **Shadows, highlights, and interior neutrals**; contrast shaping, lifted/crushed blacks, highlight clipping near windows |

**Diverse coverage matters because** a single grade must be judged simultaneously on: **skin tones**
(portrait, food), **skies** (landscape, city), **greens** (landscape, close-up), **neutrals/grays**
(city, interior — these reveal color casts most honestly), and the **highlight/shadow extremes**
(interior windows, sky highlights, street shadows). If all six were, say, sunny landscapes, a grade
that wrecks skin or crushes interiors would look fine here — the panel would lie. Spread is what
makes the demo trustworthy.

---

## 2. Neutrality requirement (critical)

Each reference must be **roughly neutral / lightly-graded / close to a straight camera output** —
*not* already stylized. If a source photo is itself a heavy teal-orange cinematic grade, applying
the user's LUT on top shows almost nothing (or compounds unpredictably) and the demo fails.

Selection rules:
- Prefer natural-light, balanced-exposure, **SOOC-looking** photos: full tonal range, no crushed
  blacks, no blown highlights, neutral-ish white balance, moderate (not punchy) saturation.
- Avoid: obvious filters/presets, heavy vignettes, film-emulation looks, extreme HDR, monochrome,
  strong single-color scenes (an all-red wall gives the grade nothing neutral to act on).
- Each image should contain a **usable neutral or near-neutral element** (gray road, white wall,
  concrete, a white plate) so casts are readable, plus at least one saturated element so chroma
  moves are visible.
- Well-exposed with headroom in both tails — so lifted-blacks / crushed-blacks / highlight moves
  have somewhere to go.

Think "clean reference plates," the photographic equivalent of a neutral gray card scene.

---

## 3. Sourcing (royalty-free, attributed)

Download from **Unsplash** and/or **Pexels** (both allow free commercial and non-commercial use, no
attribution legally required, but we record it anyway for provenance). Codex + computer-use can
browse and download directly.

Guidelines:
- **License**: use only Unsplash License or Pexels License images (the default for those sites).
  Do **not** pull from Google Images, stock sites with watermarks, or anything of uncertain rights.
- **Record the source**: for every file, capture the **canonical photo page URL** (not the CDN blob
  URL) and the license name. These go in `references.json` (§5). Also note the photographer if shown
  (nice-to-have).
- **Search terms** that tend to yield neutral results:
  - city → "city street daytime", "urban street overcast", "street neutral"
  - landscape → "landscape overcast", "green valley daylight", "mountain lake neutral"
  - portrait → "natural light portrait plain background", "headshot daylight" (varied skin tones
    across the set is a plus — do not make all people the same skin tone)
  - closeup → "flower macro", "colorful fabric macro", "object close up studio"
  - food → "plated food overhead natural light", "breakfast table daylight"
  - interior → "living room window daylight", "minimal interior architecture"
- Prefer **landscape orientation** (they render in a 4:3 grid card, `04_frontend.md` §3.5).
- Sanity-check each pick against §2 before downloading; re-pick if it's already stylized.

If direct download via the site is blocked, Unsplash/Pexels also expose direct image URLs on the
photo page ("Download" button) — use those. Save originals to a temp dir, then process (§4).

---

## 4. Processing & output format

Normalize every image to a consistent, lightweight, sRGB asset so the grid loads fast and the LUT
math behaves (LUTs here assume sRGB input; see `docs/behavior_spec.md`).

Target spec per file:
- **Color space**: sRGB (convert/assign sRGB; strip other ICC profiles). 8-bit.
- **Format**: JPEG, quality ~82–85.
- **Dimensions**: long edge **~1200px** (downscale only; never upscale). Landscape preferred.
- **File size**: **< 300 KB** each (re-compress/resize until under; quality floor ~78).
- **Strip EXIF** (orientation baked in, metadata removed) to avoid rotation surprises and shrink size.
- **Filenames** (exactly, in `webapp/assets/references/`):
  `city.jpg`, `landscape.jpg`, `portrait.jpg`, `closeup.jpg`, `food.jpg`, `interior.jpg`.

Suggested processing (ImageMagick, available in most environments):

```bash
# for each downloaded original → normalized reference
magick input.jpg \
  -auto-orient \
  -resize '1200x1200>' \
  -colorspace sRGB -strip \
  -quality 84 \
  webapp/assets/references/<slug>.jpg
# verify size; if ≥ 300KB, drop quality (e.g. -quality 78) or resize to 1000px long edge.
```

(If ImageMagick isn't present, Pillow does the same: `Image.open`, `ImageOps.exif_transpose`,
`thumbnail((1200,1200))`, `convert('RGB')`, `save(..., 'JPEG', quality=84, optimize=True)`.)

---

## 5. `references.json` manifest

Write `webapp/assets/references/references.json`. This is the source of truth the backend uses to
populate `previews[1..6]` (and to attribute sources). Keep the array **ordered** exactly as the grid
should render: City, Landscape, Portrait, Close-up, Food, Interior.

```json
[
  {
    "name": "City",
    "category": "city",
    "file": "city.jpg",
    "source_url": "https://unsplash.com/photos/XXXXXXXX",
    "license": "Unsplash License",
    "photographer": "Jane Doe"
  },
  {
    "name": "Landscape",
    "category": "landscape",
    "file": "landscape.jpg",
    "source_url": "https://www.pexels.com/photo/XXXXXXX/",
    "license": "Pexels License",
    "photographer": "John Roe"
  },
  { "name": "Portrait",  "category": "portrait",  "file": "portrait.jpg",  "source_url": "…", "license": "Unsplash License" },
  { "name": "Close-up",  "category": "closeup",   "file": "closeup.jpg",   "source_url": "…", "license": "Pexels License" },
  { "name": "Food",      "category": "food",      "file": "food.jpg",      "source_url": "…", "license": "Unsplash License" },
  { "name": "Interior",  "category": "interior",  "file": "interior.jpg",  "source_url": "…", "license": "Unsplash License" }
]
```

Schema per entry: `name` (grid label, title-case), `category` (slug, matches filename stem),
`file` (basename under `assets/references/`), `source_url` (canonical photo page — **required**,
for provenance), `license` (`"Unsplash License"` | `"Pexels License"`), `photographer` (optional).
The manifest MUST have exactly 6 entries in the render order above. `photographer` may be omitted
per-entry but include it when the source page shows it.

---

## 6. Procedural neutral fallback (if download fails)

Computer-use browsing can fail (network, rate limits, layout changes). The demo must **never ship
with a broken grid**. So generate a **procedural neutral test card** for any missing category so all
six cards always render and always show *some* LUT effect.

Fallback design — a synthetic "reference plate" that still exercises the grade:
- A neutral **grayscale step wedge** (black → white ramp, ~11 steps) across the bottom third — this
  makes tonal/contrast and black/highlight moves obvious.
- A row of **primary/secondary color swatches** (R, G, B, C, M, Y) plus a **skin-tone swatch**
  (~sRGB `#C89F86`) and a **neutral 50% gray** patch — so hue/temperature/saturation shifts are
  visible on the colors and casts show on the neutral.
- The category name printed small in a corner (mono), so the grid stays labeled.
- Same output spec as real images (sRGB JPEG, ~1200px long edge, < 300 KB), same filename slug.

Generate with Pillow (self-contained, no assets needed):

```python
# scripts/gen_reference_fallback.py  (illustrative)
from PIL import Image, ImageDraw
def make_card(slug, name, w=1200, h=900):
    img = Image.new("RGB", (w, h), (128,128,128))       # 50% neutral gray field
    d = ImageDraw.Draw(img)
    # color + skin swatches (top): R G B C M Y skin
    sw = [(220,60,60),(60,200,90),(70,110,220),(60,200,200),(210,70,200),(230,210,70),(200,159,134)]
    cw = w // len(sw)
    for i,c in enumerate(sw): d.rectangle([i*cw,0,(i+1)*cw,h//3], fill=c)
    # grayscale step wedge (bottom third), 11 steps
    steps = 11; sh = h//3
    for i in range(steps):
        v = round(255*i/(steps-1)); x = i*(w//steps)
        d.rectangle([x, h-sh, x+(w//steps), h], fill=(v,v,v))
    d.text((16,16), name, fill=(20,20,20))
    img.save(f"webapp/assets/references/{slug}.jpg", "JPEG", quality=84, optimize=True)
```

Manifest entries for fallbacks set `source_url: "procedural"`, `license: "Generated (CC0)"` so the
provenance stays honest. Only fall back per-missing-category; keep any real downloads that succeeded.

---

## 7. Acceptance criteria

1. Exactly **6 files** exist in `webapp/assets/references/`:
   `city.jpg landscape.jpg portrait.jpg closeup.jpg food.jpg interior.jpg`.
2. Each is **sRGB JPEG, long edge ≈ 1200px, < 300 KB, EXIF-stripped**.
3. Each real (non-fallback) image is **roughly neutral / lightly-graded** per §2 — not already
   stylized — and its category shows the right content (skin in portrait, sky+greens in landscape,
   neutrals in city/interior, saturated subject in close-up, warm food, etc.).
4. `webapp/assets/references/references.json` exists with **6 ordered entries** (City, Landscape,
   Portrait, Close-up, Food, Interior), each with `name`, `category`, `file`, `source_url`,
   `license` (photographer where available). Every `file` resolves on disk.
5. All six render in the frontend reference grid (`04_frontend.md` §3.5) as `previews[1..6]`, in the
   manifest order.
6. Applying a clearly non-identity LUT (e.g. a warm, contrasty, or teal-orange grade) **visibly
   changes** every one of the six — the LUT effect is obvious across skin, sky, greens, neutrals,
   and highlight/shadow extremes.
7. If any download fails, a procedural neutral test card is generated for that slug so the grid is
   always complete, and its manifest entry is marked `procedural` / `Generated (CC0)`.
