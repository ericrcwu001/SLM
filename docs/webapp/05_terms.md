# 05 — Grounded term vocabulary, glossary API, and the term-suggestion algorithm

Status: spec for the local prompt→LUT demo website. Governs the hover-glossary
(`GET /api/terms`) and the term-suggestion box (`suggest_terms`). Everything here is
**grounded in the pipeline's real vocabulary** — no term is invented. Source-of-truth files:

- `eval/tag_vocabulary.py` — `DIRECTIONAL_TAG_AXIS`, `STYLE_TAGS`, `RETIRED_ALIASES`,
  `HUE_SECTORS`, `HUE_CAST_TAGS`, `SAT_SECTOR_TAGS`, `canonicalize_tag`.
- `data_pipeline/attribute_spec.py` — `_BIPOLAR`, `_UNIPOLAR`, `_HUE`, `_MAG_BUCKETS`,
  `serialize`/`serialize_bucketed`/`parse`, `AttributeSpec`.
- `data_pipeline/behavior_vector.py` — what each `behavior_v2` axis actually measures.
- `docs/attribute_spec.md` §3/§7/§10, `docs/behavior_spec.md` — axis meanings.

## 0. Why this feature exists (the grounding rule)

`docs/interpreter_results.md` is the load-bearing finding: the interpreter maps **direction**
well (`attribute_direction_f1 ≈ 0.47`) but **magnitude poorly** for vague text (`bucket_f1 ≈ 0.16`,
`exact ≈ 0.11`, flat across 5× data). The one lever that helped was **explicit intensity wording**
(the `literal` caption style reached `attribute_f1 ≈ 0.22`). So the website never rewrites the
user's prompt — it **suggests single terms / short phrases** that sharpen the prompt on **direction**
and **magnitude**, and it only ever suggests terms that are **grounded**:

> **grounded** ≙ the term maps to a real `attribute_spec` axis (a `_BIPOLAR` / `_UNIPOLAR` / `_HUE`
> field, or a `per_hue_saturation` sector) **or** is a magnitude bucket (`_MAG_BUCKETS`), **and** the
> term is **not** a `RETIRED_ALIAS`.

Grounded terms are "proven to work": they round-trip through `parse`/`serialize`, they back to a
`behavior_v2` axis the pipeline can measure (the backing rule, `attribute_spec.py:is_backed`), and
the eval harness sign-checks them (`DIRECTIONAL_TAG_AXIS`). Composite style words (`cinematic`,
`teal-orange`, …) are **recognized** vocabulary but not single-axis-backed, so they are documented
for the hover glossary yet **never suggested**. Retired aliases are never surfaced at all.

---

## 1. The grounded glossary

47 grounded terms + 7 recognized-but-not-suggested style composites. Every row's `axis` is copied
from the real tables above; definitions are authored once here and joined to the derived structure
by the builder (§2) so the *structure* cannot drift.

`grounded: true` ⇒ eligible for suggestion. `grounded: false` ⇒ documented (hover glossary explains
it if the user types it) but the suggester never emits it. Retired aliases are in the blocklist
(§1.7) and appear in no payload.

### 1.1 Direction — bipolar tonal & color pairs (`_BIPOLAR`, sign-checked by `DIRECTIONAL_TAG_AXIS`)

Each pair is one `behavior_v2` axis with a `+`/`−` pole. Magnitude is Lab units (see §1.6). All
`grounded: true`, category **Direction**.

| term | axis (field, sign) | plain-English | technical | example_usage |
|---|---|---|---|---|
| `warmer` | `temperature_delta_b` (+) | shifts the whole image toward warm (amber/yellow) | raises mean Lab **b\*** across the neutral ramp (toward +b, yellow) | "make it warmer" |
| `cooler` | `temperature_delta_b` (−) | shifts toward cool (blue) | lowers mean Lab b\* (toward −b, blue) | "a cooler look" |
| `tint_magenta` | `tint_delta_a` (+) | adds a magenta/pink tint | raises mean Lab **a\*** (toward +a, magenta/red) | "slight magenta tint" |
| `tint_green` | `tint_delta_a` (−) | adds a green tint | lowers mean Lab a\* (toward −a, green) | "pull the green tint" |
| `brighter` | `mean_l_delta` (+) | raises overall exposure | raises mean Lab **L\*** over the ramp | "make it brighter" |
| `darker` | `mean_l_delta` (−) | lowers overall exposure | lowers mean Lab L\* | "darker overall" |
| `more_contrast` | `contrast_l_spread_delta` (+) | widens the gap between darks and lights | increases the L\* p95−p5 **spread** (steeper tone curve) | "more contrast" |
| `less_contrast` | `contrast_l_spread_delta` (−) | flattens the tones | narrows the L\* spread (flatter curve) | "less contrast" |
| `more_saturated` | `chroma_delta` (+) | more intense color | raises mean Lab **chroma** (√(a\*²+b\*²)) | "more saturated" |
| `muted` | `chroma_delta` (−) | drains color intensity | lowers mean chroma (desaturates) | "a muted palette" |
| `lifted_blacks` | `black_point_l_delta` (+) | darkest tones become dark-grey, not pure black (faded low end) | raises the **black point** — the output L\* mapped to input-black — lifting the shadow toe | "lifted blacks" |
| `crushed_blacks` | `black_point_l_delta` (−) | shadows go to deep, inky black and lose detail | lowers/clamps the black point (crushed toe) | "crush the blacks" |
| `lifted_shadows` | `shadow_l_delta` (+) | opens up the shadow region (more shadow detail) | raises L\* in the shadow tone region (input L\* ≤ 25) | "lift the shadows" |
| `crushed_shadows` | `shadow_l_delta` (−) | darkens the shadow region | lowers shadow-region L\* (negative pole of `shadow_l_delta`; serialized spec key, `_BIPOLAR`) | "deepen the shadows" |
| `brighter_highlights` | `highlight_l_delta` (+) | brightens only the highlights | raises L\* in the highlight region (input L\* ≥ 75) | "brighter highlights" |
| `softer_highlights` | `highlight_l_delta` (−) | rolls off / protects highlights | lowers/compresses highlight-region L\* (highlight shoulder) | "soften the highlights" |

> **Contrast toe & shoulder (technical note, no user term).** The pipeline also measures contrast
> *shape* — `contrast_toe_delta` (local tone-curve slope in the shadow **toe**, input L\* ≤
> `SHADOW_L_MAX`) and `contrast_shoulder_delta` (slope in the highlight **shoulder**, ≥
> `HIGHLIGHT_L_MIN`), each relative to the identity slope of 1.0. These are **measured axes only**:
> there is no serialized `attribute_spec` key and no `DIRECTIONAL_TAG_AXIS` tag for them, so there is
> **no grounded user term**. Contrast intent is expressed with `more_contrast`/`less_contrast`
> (global spread); toe/shoulder are surfaced only in the technical explanation of contrast.

### 1.2 Hue — region temperature, global cast, split-tone strength

| term | axis | category | plain-English | technical | example_usage | grounded |
|---|---|---|---|---|---|---|
| `warmer_shadows` | `split_tone_shadow_b` (+) | Hue | tints the **shadows** warmer | raises the shadow-region b\* shift (split-toning); in the spec grammar this surfaces as `shadow_hue` (angle) + `split_strength` | "warmer shadows" | true |
| `cooler_shadows` | `split_tone_shadow_b` (−) | Hue | tints the shadows cooler | lowers shadow-region b\* shift | "cool the shadows" | true |
| `warmer_highlights` | `split_tone_highlight_b` (+) | Hue | tints the **highlights** warmer | raises highlight-region b\* shift | "warm the highlights" | true |
| `cooler_highlights` | `split_tone_highlight_b` (−) | Hue | tints the highlights cooler | lowers highlight-region b\* shift | "cooler highlights" | true |
| `hue_cast_red` | `global_hue_deg` + `global_hue_magnitude` | Hue | pushes the overall cast toward red | sets the global cast's hue angle to the red sector (~25°) with magnitude; `HUE_CAST_TAGS` | "a red cast" | true |
| `hue_cast_orange` | `global_hue_deg` + mag | Hue | overall orange cast | global hue angle ≈ 55° | "orange cast" | true |
| `hue_cast_yellow` | `global_hue_deg` + mag | Hue | overall yellow cast | global hue angle ≈ 95° | "yellow cast" | true |
| `hue_cast_green` | `global_hue_deg` + mag | Hue | overall green cast | global hue angle ≈ 160° | "green cast" | true |
| `hue_cast_cyan` | `global_hue_deg` + mag | Hue | overall cyan cast | global hue angle ≈ 200° | "cyan cast" | true |
| `hue_cast_blue` | `global_hue_deg` + mag | Hue | overall blue cast | global hue angle ≈ 270° | "blue cast" | true |
| `hue_cast_magenta` | `global_hue_deg` + mag | Hue | overall magenta cast | global hue angle ≈ 330° | "magenta cast" | true |
| `split_strength` | `split_tone_strength` (≥0, `_UNIPOLAR`) | Hue | how strongly shadows and highlights are tinted toward **different** hues | intensity of split-toning (‖shadow a,b shift‖ + ‖highlight a,b shift‖); also gates whether `shadow_hue`/`highlight_hue` are emitted | "stronger split-tone" | true |

> **Split-toning (technical).** Applying one hue to the shadows and a *different* hue to the
> highlights. `split_strength` is the magnitude of that separation; `warmer/cooler_shadows` and
> `warmer/cooler_highlights` are the per-region temperature poles (the sign-checked backward-compat
> proxy for the region-hue axes, `tag_vocabulary.py` lines 20–21). The 7 `hue_cast_*` sectors are the
> **global** cast direction. Sector centers (Lab hue circle): red 25°, orange 55°, yellow 95°,
> green 160°, cyan 200°, blue 270°, magenta 330° (`behavior_vector._HUE_SECTOR_CENTERS`).

### 1.3 Saturation — per-hue sectors (`SAT_SECTOR_TAGS`, `per_hue_saturation[sector]`)

14 terms: for each of the 7 `HUE_SECTORS` a `_up` (boost) and `_down` (crush) pole. All
`grounded: true`, category **Saturation**. `axis = per_hue_saturation[sector]`, sign `+` for `_up`,
`−` for `_down`.

| term family | axis | plain-English | technical | example_usage |
|---|---|---|---|---|
| `sat_red_up` / `sat_red_down` | `per_hue_saturation[red]` (±) | boost / crush the saturation of **reds only** | changes chroma only for input pixels whose hue bins to the red sector (selective, per-input-hue saturation) | "boost the reds" / "mute the reds" |
| `sat_orange_up` / `sat_orange_down` | `per_hue_saturation[orange]` (±) | boost / crush **oranges** (skin-tone family) | per-sector chroma delta, orange bin | "richer oranges" / "tone down oranges" |
| `sat_yellow_up` / `sat_yellow_down` | `per_hue_saturation[yellow]` (±) | boost / crush **yellows** | per-sector chroma delta, yellow bin | "punchier yellows" / "mute yellows" |
| `sat_green_up` / `sat_green_down` | `per_hue_saturation[green]` (±) | boost / crush **greens** (foliage) | per-sector chroma delta, green bin | "crush the greens" / "greener greens" |
| `sat_cyan_up` / `sat_cyan_down` | `per_hue_saturation[cyan]` (±) | boost / crush **cyans** | per-sector chroma delta, cyan bin | "boost the cyans" |
| `sat_blue_up` / `sat_blue_down` | `per_hue_saturation[blue]` (±) | boost / crush **blues** (skies) | per-sector chroma delta, blue bin | "deeper blue skies" / "mute the blues" |
| `sat_magenta_up` / `sat_magenta_down` | `per_hue_saturation[magenta]` (±) | boost / crush **magentas** | per-sector chroma delta, magenta bin | "boost magentas" |

### 1.4 Tone-shape — matte

| term | axis | category | plain-English | technical | example_usage | grounded |
|---|---|---|---|---|---|---|
| `matte` | `matte_strength` (≥0, `_UNIPOLAR`) | Tone-shape | the matte / faded-film low-contrast look | first-class axis = lifted black point **+** reduced contrast **+** slight desaturation, combined (`behavior_vector.matte_strength`) | "a matte finish" | true |

### 1.5 Magnitude — intensity buckets (`_MAG_BUCKETS`)

Intensity modifiers, category **Magnitude**, `grounded: true`. `axis = "(magnitude bucket)"`. These
are the exact ordinal bands the generator consumes via `serialize_bucketed` / `_bucket_mag`, in Lab
units. (An axis below the emit threshold `_MAG_EPS = 0.5` Lab is dropped entirely, so `slight`
covers the smallest *emitted* movement, ~0.5–1.5.)

| term | band (Lab units) | plain-English | example_usage |
|---|---|---|---|
| `slight` | `abs(v) < 1.5` | a barely-there nudge | "slightly warmer" |
| `moderate` | `1.5 ≤ abs(v) < 3.0` | a clear but restrained change | "moderately more contrast" |
| `strong` | `3.0 ≤ abs(v) < 6.0` | a bold, obvious change | "strongly muted" |
| `extreme` | `abs(v) ≥ 6.0` | push it as far as it goes | "extreme teal cast" |

> These are the "how much" words the research says the model needs. They attach to any Direction /
> Hue / Saturation term ("`extreme` + `warmer`", "`slight` + `crush the greens`").

### 1.6 Sign & magnitude semantics (shared by all of §1.1–§1.5)

- The **tag encodes the direction**; the serialized magnitude is always a positive Lab number
  (`serialize`: `warmer=+2.3`). Buckets replace the number for generator input only
  (`serialize_bucketed`: `warmer=moderate`) — lossy, one-way, never parsed back.
- Hue angles are integer degrees (`shadow_hue=210`), not magnitudes.
- Per-hue-sat keeps its sign word-free of a decimal (`sat_green=-1.5`).

### 1.7 Retired aliases — NEVER suggest, NEVER document as usable (`RETIRED_ALIASES`)

Canonicalized on ingest but removed from the code vocabulary; the builder excludes them from every
payload. If the user types one, treat it as its canonical target for detection, but suggest the
canonical form.

| retired alias | → canonical (suggest this instead) |
|---|---|
| `more_magenta` | `tint_magenta` |
| `more_green` | `tint_green` |
| `higher_contrast` | `more_contrast` |
| `softer_contrast` | `less_contrast` |
| `desaturated` | `muted` |

### 1.8 Recognized-but-not-suggested style composites (`STYLE_TAGS`, minus `matte`)

`grounded: false`, category **Style**, `axis = "composite (calibration window;
eval/configs/calibration_manifest.json)"`. The teacher can emit these and eval matches them, but they
are **measured composites**, not a single backable `attribute_spec` axis — so the hover glossary
explains them, but `suggest_terms` never emits them (it emits the grounded equivalents in §3.2).

| term | plain-English | technical (grounded equivalents) |
|---|---|---|
| `teal-orange` | the "blockbuster" teal-shadow / orange-highlight look | ≈ `cooler_shadows` + `warmer_highlights` + `split_strength` |
| `cinematic` | filmic, slightly desaturated, split-toned | ≈ `less_contrast` + `muted` + `split_strength` |
| `filmic` | analog-film tonality | ≈ `lifted_blacks` + `muted` + soft highlights |
| `faded` | washed-out, low-contrast | ≈ `lifted_blacks` + `less_contrast` + `muted` |
| `sepia` | warm monochrome-brown | ≈ `warmer` + `muted` (+ magenta/red tint) |
| `bleach bypass` | high-contrast, desaturated, silvery | ≈ `more_contrast` + `muted` |
| `natural` | true-to-life, minimal grade | ≈ small magnitudes, no strong cast |

---

## 2. `GET /api/terms` payload (hover glossary)

The frontend hover glossary consumes a flat list. **It is generated at runtime from the vocabulary
tables**, not hand-maintained, so it can't drift from `eval/tag_vocabulary.py` +
`data_pipeline/attribute_spec.py`. Retired aliases are excluded; each entry carries `grounded` so the
UI can style suggestable terms differently from merely-recognized composites.

```json
[
  {"term": "warmer", "axis": "temperature_delta_b (+)", "category": "Direction",
   "definition": "shifts the whole image toward warm (amber/yellow); raises mean Lab b*",
   "example_usage": "make it warmer", "grounded": true},
  {"term": "slight", "axis": "(magnitude bucket)", "category": "Magnitude",
   "definition": "a barely-there nudge (abs magnitude < 1.5 Lab units)",
   "example_usage": "slightly warmer", "grounded": true},
  {"term": "teal-orange", "axis": "composite (calibration window)", "category": "Style",
   "definition": "teal-shadow / orange-highlight blockbuster look; a composite, not one axis",
   "example_usage": "teal-orange grade", "grounded": false}
]
```

> Shape note: the base API is `[{term, axis, category, definition, example_usage}]`; we add the
> `grounded` boolean (per §1's "mark each grounded") so the UI can gate suggestions, plus an optional
> `sign` integer (`+1`/`-1`) on directional and saturation entries so the UI can show which way the
> axis moves (`null`/absent on magnitude and style rows). The `suggested_terms` objects in §3 project
> `{term, axis, definition, example_usage, grounded}`.

### 2.1 `webapp/terms.py` builder sketch

```python
# webapp/terms.py  — glossary is DERIVED from the frozen vocabulary tables.
from eval.tag_vocabulary import (
    DIRECTIONAL_TAG_AXIS, RETIRED_ALIASES, STYLE_TAGS, HUE_SECTORS,
    HUE_CAST_TAGS, SAT_SECTOR_TAGS,
)
from data_pipeline.attribute_spec import _BIPOLAR, _UNIPOLAR, _MAG_BUCKETS

# --- authored prose only (the ONE hand-maintained thing) ------------------------------
# term -> {"plain": ..., "tech": ..., "example": ...}. Keys MUST be a subset of the derived
# grounded term set; build_glossary() asserts coverage so a new tag fails loudly until documented.
_DEFS: dict[str, dict[str, str]] = { ... }   # see §1 tables (verbatim)

_BUCKET_TERMS = [label for _hi, label in _MAG_BUCKETS] + ["extreme"]  # slight/moderate/strong/extreme

def _category(term: str, axis_field: str | None) -> str:
    if term in _BUCKET_TERMS:                                   return "Magnitude"
    if term == "matte":                                         return "Tone-shape"
    if term in HUE_CAST_TAGS or term == "split_strength":       return "Hue"
    if axis_field in ("split_tone_shadow_b", "split_tone_highlight_b"): return "Hue"
    if term in SAT_SECTOR_TAGS:                                 return "Saturation"
    if term in STYLE_TAGS:                                      return "Style"
    return "Direction"

def build_glossary() -> list[dict]:
    """Every glossary entry, DERIVED from the vocabulary tables. Structure cannot drift."""
    entries: list[dict] = []
    def add(term, axis, grounded):
        d = _DEFS[term]
        entries.append({
            "term": term, "axis": axis, "category": _category(term, axis.split()[0]),
            "definition": f'{d["plain"]}; {d["tech"]}', "example_usage": d["example"],
            "grounded": grounded,
        })

    # (a) bipolar directional + region-temp directional tags (sign-checked)
    for tag, (field, sign) in DIRECTIONAL_TAG_AXIS.items():
        add(tag, f"{field} ({'+' if sign > 0 else '-'})", True)
    # (b) crushed_shadows: negative pole of shadow_l_delta present in _BIPOLAR only
    for field, (pos, neg) in _BIPOLAR.items():
        if neg not in DIRECTIONAL_TAG_AXIS:
            add(neg, f"{field} (-)", True)
    # (c) unipolar: matte, split_strength
    for key, field in _UNIPOLAR.items():
        add(key, f"{field} (>=0)", True)
    # (d) global hue-cast sectors (already in HUE_CAST_TAGS)  -> global_hue_deg + magnitude
    for tag in HUE_CAST_TAGS:
        add(tag, "global_hue_deg + global_hue_magnitude", True)
    # (e) per-hue saturation sectors
    for tag in SAT_SECTOR_TAGS:
        sector = tag[len("sat_"):].rsplit("_", 1)[0]
        add(tag, f"per_hue_saturation[{sector}] ({'+' if tag.endswith('_up') else '-'})", True)
    # (f) magnitude buckets
    for label in _BUCKET_TERMS:
        add(label, "(magnitude bucket)", True)
    # (g) recognized style composites (documented, NOT suggested)
    for tag in STYLE_TAGS:
        if tag != "matte":
            add(tag, "composite (calibration window; eval/configs/calibration_manifest.json)", False)

    assert not (set(RETIRED_ALIASES) & {e["term"] for e in entries})  # no alias ever leaks
    return entries

_GLOSSARY = build_glossary()
_GROUNDED = {e["term"] for e in _GLOSSARY if e["grounded"]}
_BY_TERM  = {e["term"]: e for e in _GLOSSARY}

def api_terms() -> list[dict]:
    """GET /api/terms — the hover glossary. Retired aliases already excluded."""
    return _GLOSSARY
```

---

## 3. `suggest_terms` — the suggestion algorithm

```python
def suggest_terms(prompt: str,
                  parsed_spec: "AttributeSpec | None",
                  route: str) -> dict:
    """Return {"assessment": str, "suggested_terms": [ {term, axis, definition,
    example_usage, grounded}, ... ]}.  NEVER returns full prompts. Every suggested term
    is a grounded glossary entry (grounded == True). Deterministic (no randomness)."""
```

Inputs: the raw `prompt`, the interpreter's `parsed_spec` (from
`data_pipeline.attribute_spec.parse(interpreter_output)`, or `None` if the interpreter wasn't run),
and the interpreter `route` (`grade` / `clarify` / `refuse`).

### 3.1 Detection helpers (deterministic, keyword + axis based)

```python
import re
from data_pipeline.attribute_spec import _BIPOLAR, _UNIPOLAR
from eval.tag_vocabulary import canonicalize_tag

# intensity words that count as an explicit magnitude in the RAW prompt (the thing the research
# says is missing). Buckets themselves + common natural-language intensifiers/diminishers.
_MAG_WORDS = {
    "slight", "slightly", "moderate", "moderately", "strong", "strongly", "extreme", "extremely",
    "subtle", "subtly", "barely", "a touch", "a bit", "a little", "hint", "gentle", "faint",
    "very", "super", "really", "heavily", "way", "much", "tons", "a lot", "punch", "intense",
    "dramatically", "aggressively",
}

def _has_magnitude(prompt: str, spec) -> bool:
    p = prompt.lower()
    return any(w in p for w in _MAG_WORDS)   # raw-text intensity is what we nudge for

def _asserted_axes(spec) -> set[str]:
    """behavior_v2 fields (and 'sat:<sector>') the interpreter already asserted."""
    if spec is None:
        return set()
    axes = {f for f in spec.axes}                         # e.g. temperature_delta_b, matte_strength
    axes |= {f"sat:{s}" for s in spec.sat}                # per-hue sectors
    return axes

def _term_axis_field(term: str) -> str | None:
    """The behavior_v2 field a grounded term backs to (for redundancy checks)."""
    if term in DIRECTIONAL_TAG_AXIS:  return DIRECTIONAL_TAG_AXIS[term][0]
    if term in _UNIPOLAR:             return _UNIPOLAR[term]
    if term in HUE_CAST_TAGS:         return "global_hue_deg"
    if term.startswith("sat_"):       return "sat:" + term[len("sat_"):].rsplit("_", 1)[0]
    # crushed_shadows -> shadow_l_delta (from _BIPOLAR negative poles)
    for field, (_pos, neg) in _BIPOLAR.items():
        if term == neg: return field
    return None

def _is_redundant(term: str, prompt: str, asserted: set[str]) -> bool:
    if term in prompt.lower():                 return True   # user already used the word
    return _term_axis_field(term) in asserted                # axis already asserted by interpreter
```

### 3.2 Vague-word → grounded-term map (single terms only, all grounded)

```python
# common vague words the user might type -> the nearest GROUNDED equivalents (never composites).
_VAGUE_TO_GROUNDED: dict[str, list[str]] = {
    "pop":       ["more_saturated", "more_contrast"],
    "punchy":    ["more_contrast", "more_saturated"],
    "vibrant":   ["more_saturated"],
    "vivid":     ["more_saturated"],
    "colorful":  ["more_saturated"],
    "washed":    ["lifted_blacks", "muted", "matte"],      # "washed out"
    "faded":     ["lifted_blacks", "less_contrast", "muted"],
    "matte":     ["matte", "lifted_blacks"],
    "moody":     ["darker", "muted", "crushed_blacks"],
    "dramatic":  ["more_contrast", "crushed_blacks"],
    "soft":      ["less_contrast", "softer_highlights", "lifted_blacks"],
    "dreamy":    ["less_contrast", "lifted_blacks", "muted"],
    "cinematic": ["less_contrast", "muted", "split_strength"],
    "filmic":    ["lifted_blacks", "muted", "softer_highlights"],
    "film":      ["lifted_blacks", "muted"],
    "vintage":   ["warmer", "muted", "lifted_blacks", "matte"],
    "retro":     ["warmer", "muted", "lifted_blacks"],
    "warm":      ["warmer"],
    "cold":      ["cooler"],
    "cool":      ["cooler"],
    "teal-orange": ["cooler_shadows", "warmer_highlights", "split_strength"],
    "teal and orange": ["cooler_shadows", "warmer_highlights"],
    "sepia":     ["warmer", "muted", "tint_magenta"],
    "bleach":    ["more_contrast", "muted"],               # "bleach bypass"
    "bright":    ["brighter"],
    "dark":      ["darker"],
    "contrasty": ["more_contrast"],
    "flat":      ["less_contrast"],
    "clean":     ["muted", "less_contrast"],
    "rich":      ["more_saturated", "more_contrast"],
}
# a small curated "starter" set for clarify/empty prompts: strong on DIRECTION + MAGNITUDE.
_STARTER_DIRECTION = ["warmer", "cooler", "brighter", "darker", "more_contrast", "more_saturated"]
_STARTER_MAGNITUDE = ["slight", "moderate", "strong", "extreme"]
```

### 3.3 Algorithm

```python
def suggest_terms(prompt, parsed_spec, route):
    p = (prompt or "").lower()
    asserted = _asserted_axes(parsed_spec)
    picks: list[str] = []            # ordered, de-duped below
    notes: list[str] = []

    def want(*terms):
        for t in terms:
            t = canonicalize_tag(t)                        # map any alias to canonical
            if t in _GROUNDED and t not in picks and not _is_redundant(t, p, asserted):
                picks.append(t)

    # (d) clarify route: LEAD with direction + magnitude
    if route == "clarify":
        notes.append("Your request is under-specified — pick a direction and say how much.")
        want(*_STARTER_DIRECTION)
        want(*_STARTER_MAGNITUDE)

    # refuse route: nothing to grade; do not suggest grade terms
    elif route == "refuse":
        return {"assessment": "This request can't be graded as a single global LUT, "
                              "so there are no color terms to suggest.",
                "suggested_terms": []}

    # (b) vague style words -> grounded equivalents
    hit_vague = [w for w in _VAGUE_TO_GROUNDED if w in p]
    for w in hit_vague:
        want(*_VAGUE_TO_GROUNDED[w])
    if hit_vague:
        notes.append("Vague style words map to specific grounded terms — try these instead.")

    # (a) no magnitude word anywhere -> suggest intensity terms
    if not _has_magnitude(prompt, parsed_spec):
        want(*_STARTER_MAGNITUDE)
        notes.append("Your request doesn't say HOW MUCH — add an intensity word "
                     "(slight / moderate / strong / extreme).")

    # (c) direction present but sharpenable: if exactly one axis is asserted, offer to
    #     also pin a complementary tonal axis (kept conservative & grounded).
    if parsed_spec is not None and len(asserted) == 1 and not hit_vague:
        only = next(iter(asserted))
        if only in ("temperature_delta_b", "tint_delta_a"):        # a color-only request
            want("more_contrast", "brighter")                      # common complements
            notes.append("Direction is clear — you can also pin a tonal term to sharpen it.")

    # if the interpreter gave us NOTHING and no words matched, fall back to the starter set
    if not picks and route != "refuse":
        want(*_STARTER_DIRECTION); want(*_STARTER_MAGNITUDE)
        notes.append("Start by naming a direction and an intensity.")

    picks = picks[:6]                                              # cap the UI list
    suggested = [{
        "term": _BY_TERM[t]["term"], "axis": _BY_TERM[t]["axis"],
        "definition": _BY_TERM[t]["definition"], "example_usage": _BY_TERM[t]["example_usage"],
        "grounded": True,                                          # invariant: only grounded picked
    } for t in picks]
    return {"assessment": " ".join(notes) if notes else
            "Your request already names a clear direction and intensity.",
            "suggested_terms": suggested}
```

Key properties enforced by construction:

- **Only grounded terms.** `want()` filters through `_GROUNDED`, so composites and retired aliases
  can never be emitted; the emitted `grounded` field is always `True`.
- **Never a full prompt.** Only single vocabulary terms / short glossary phrases are returned.
- **No redundancy.** `_is_redundant` drops any term whose word is already in the prompt or whose axis
  the interpreter already asserted; magnitude buckets are skipped when `_has_magnitude` is true.
- **Deterministic.** No randomness; identical inputs → identical output (dict-insertion order over
  fixed tables).

### 3.4 Acceptance criteria

1. **Membership.** For every input, each `term` in `suggested_terms` satisfies `term in _GROUNDED`
   and appears in `api_terms()` with `grounded == True`. (No composite, no retired alias, ever.)
2. **Vague prompt gets magnitude help.** `suggest_terms("make it warmer", parse("route=grade |
   warmer=+1.0"), "grade")` — no intensity word → `suggested_terms` includes `slight`, `moderate`,
   `strong`, `extreme`, and the assessment contains "HOW MUCH". `warmer` is **not** re-suggested
   (word already present + axis asserted).
3. **Already specific → few/no suggestions.** `suggest_terms("make it much warmer", parse("route=grade
   | warmer=+4.0"), "grade")` — magnitude word "much" present and `temperature_delta_b` asserted →
   no bucket suggestions, no `warmer`; result is empty or only a conservative complementary tonal
   term (§3.3c). The redundant-suggestion count for the asserted axis is 0.
4. **Vague style word maps to grounded.** `suggest_terms("make it cinematic", None, "grade")` →
   suggestions come from `_VAGUE_TO_GROUNDED["cinematic"]` (`less_contrast`, `muted`,
   `split_strength`), plus magnitude buckets (no intensity word); `cinematic` itself is **not**
   suggested (it's a non-grounded composite).
5. **Clarify leads with direction + magnitude.** For `route == "clarify"`, the first suggestions are
   from `_STARTER_DIRECTION` then `_STARTER_MAGNITUDE`, and the assessment says the request is
   under-specified.
6. **Refuse yields nothing.** For `route == "refuse"`, `suggested_terms == []`.

---

## 4. Source-of-truth cross-references

| This doc | Ground truth |
|---|---|
| §1.1 direction pairs | `DIRECTIONAL_TAG_AXIS` + `_BIPOLAR` (`crushed_shadows`) |
| §1.2 region temp / hue-cast / split_strength | `DIRECTIONAL_TAG_AXIS` (split_tone_*_b), `HUE_CAST_TAGS`, `_UNIPOLAR`, `_HUE` |
| §1.3 per-hue saturation | `SAT_SECTOR_TAGS`, `HUE_SECTORS`, `behavior_vector.per_hue_saturation` |
| §1.4 matte | `_UNIPOLAR["matte"]` → `matte_strength` |
| §1.5 magnitude buckets | `_MAG_BUCKETS` + `_bucket_mag` (`extreme`) |
| §1.7 retired aliases | `RETIRED_ALIASES` |
| §1.8 style composites | `STYLE_TAGS` (minus `matte`) |
| §2 payload / builder | `build_glossary()` derives from all of the above |
| §3 parsed_spec | `data_pipeline.attribute_spec.parse` → `AttributeSpec{route, axes, sat}` |

If a vocabulary table changes, `build_glossary()` regenerates automatically and its coverage
`assert` fails until §1's `_DEFS` documents the new term — so the glossary and suggester can never
silently drift from the pipeline.
