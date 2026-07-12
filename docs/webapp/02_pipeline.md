# Webapp 02 — Pipeline spec (`webapp/pipeline.py`, `webapp/lut.py`, loaders, `webapp/terms.py`)

Companion to `docs/webapp/01_architecture.md`. This is the **exact** spec for the pipeline modules.
Every heavy operation REUSES an existing, unit-tested repo function — the signatures below were read
from source and are load-bearing. Do **not** reimplement generation, decoding, LUT-apply, or `.cube`
serialization.

Branch: `feat/two-stage`. Repo root: `/Users/ericwu/Developer/SLM`.

---

## 0. Reuse map (real signatures — copy these exactly)

| Need | Function (real signature) | Module |
|---|---|---|
| Interpreter prompt ids | `build_prompt_ids(tokenizer, text: str) -> list[int]` | `interpreter/example.py` |
| Interpreter model load pattern | `_load_model(cfg, adapter) -> (model, tokenizer, device)` | `interpreter/score.py` |
| Parse spec text → object | `parse(text: str) -> AttributeSpec` | `data_pipeline/attribute_spec.py` |
| Canonical serialize | `serialize(spec) -> str` ; `serialize_bucketed(spec) -> str` | `data_pipeline/attribute_spec.py` |
| Round to canonical precision | `canonicalize(spec) -> AttributeSpec` | `data_pipeline/attribute_spec.py` |
| Guarded parse (no silent grade) | `_safe_parse(text) -> AttributeSpec | None` | `interpreter/comparator.py` |
| Best-of-N generate + rerank | `best_of_n_codes(model, processor, *, image, cond_text, spec_text=None, n=16, sampling=None, chunk=16, device=None, fast=False) -> (best_codes | None, record)` | `eval/best_of_n.py` |
| Greedy single generate (N=1 fallback) | `generate_codes(model, processor, *, image, text, sampling=None, max_new_tokens=68, device=None) -> list[int] | None` | `sft/generate.py` |
| Grammar / code-token ids (used internally) | `SpecialIds(tokenizer)` ; `make_prefix_fn(prompt_len, ids)` | `sft/generate.py` |
| Decode codes → LUT | `decode_codes(codes, *, final_dir=None) -> np.ndarray  # [17,17,17,3]` | `eval/behavioral_fidelity.py` |
| Rerank key / agreement (diagnostics) | `rerank_key(rec) -> tuple` ; `behavioral_agreement(spec, mb, *, tol=1.0) -> dict` | `eval/behavioral_fidelity.py` |
| Score a decoded LUT vs spec | `score_from_lut(pred_lut, spec, *, codes=None, ...) -> dict` | `eval/behavioral_fidelity.py` |
| Apply LUT (trilinear) | `apply_lut_trilinear(lut_abs, rgb) -> np.ndarray` | `data_pipeline/lut_ops.py` |
| Write `.cube` | `write_cube(path, lut_abs) -> None` ; `serialize_cube(lut_abs) -> bytes` | `eval/cube_io.py` |
| Identity grid / residual↔absolute | `identity_grid(size=17)` ; `residual_to_absolute(res)` ; `absolute_to_residual(abs)` | `eval/cube_io.py` |
| Measure behavior of a LUT | `measure_behavior(lut_abs) -> dict` | `data_pipeline/behavior_vector.py` |
| CUDA 4-bit generator loader | `load_eval_model(cfg, resized_model, adapter) -> (model, processor)` | `sft/loader.py` |
| SFT bnb config source | `SFTConfig()` (min_pixels/max_pixels/load_in_4bit/bnb_*) | `sft/config.py` |
| Grounded vocabulary | `DIRECTIONAL_TAG_AXIS`, `KNOWN_TAGS`, `STYLE_TAGS`, `HUE_SECTORS`, `canonicalize_tag` | `eval/tag_vocabulary.py` |
| Route/refuse enums | `ROUTE_GRADE/ROUTE_CLARIFY/ROUTE_REFUSE`, `REFUSE_OUT_OF_SCOPE/GAMUT` | `eval/refuse_taxonomy.py` |

**Grammar facts** (from `sft/generate.py` / `eval/behavioral_fidelity.py`): a grade output is exactly
64 codebook indices in `[0,255]`; `decode_codes` requires `len(codes)==64`; a refusal yields
`<unsupported>` and `best_codes is None`.

---

## 1. `webapp/pipeline.py` — `PromptToLutPipeline`

### 1.1 Contract

```python
from dataclasses import dataclass

@dataclass
class RouteResult:
    route: str                 # "grade" | "clarify" | "refuse"
    refuse_reason: str | None  # "out_of_scope" | "out_of_gamut" | None
    spec_text: str | None      # canonical attribute_spec_text (grade only)
    clarify_message: str | None
    spec: "AttributeSpec | None"

@dataclass
class LutResult:
    codes: list[int]           # 64 codebook indices
    lut: "np.ndarray"          # absolute LUT [17,17,17,3], clipped [0,1]
    record: dict               # best_of_n record: behavioral_fidelity, collapsed, code_stats, ...
    fell_back_greedy: bool
```

```python
class PromptToLutPipeline:
    def __init__(self, cfg: WebappConfig): ...
    def self_check(self) -> dict: ...                      # assert all artifacts resolve
    def route_and_spec(self, prompt: str) -> RouteResult: ...
    def generate_lut(self, cond_text: str, spec_text: str, image) -> LutResult: ...
    def run(self, prompt: str, image, run_dir: str) -> dict: ...   # full flow → API payload
```

`image` is a **PIL.Image (RGB)**. `qwen_vl_utils.process_vision_info` (used inside
`best_of_n_codes`/`generate_codes`) accepts a PIL image or a path — pass the opened PIL image.

### 1.2 Load-once `__init__`

Models are loaded exactly once and held as attributes (see 01_architecture.md §5). Reuse the loaders
from `webapp/models_config.py` (§5 below):

```python
def __init__(self, cfg):
    self.cfg = cfg
    self.device = cfg.device
    # interpreter (text-only 0.5B)
    self.interp_model, self.interp_tok, self.interp_device = load_interpreter(cfg)
    # generator (Qwen2.5-VL-3B + LoRA), device-branched (CUDA 4-bit vs mps/cpu fp16/fp32)
    self.gen_model, self.gen_processor = load_generator(cfg)
    # VQ-VAE decoder is lazy + lru_cached inside decode_codes; just remember the dir
    self.vq_final_dir = cfg.vq_decoder.final_dir   # None → auto-resolve
```

### 1.3 `route_and_spec(prompt) -> RouteResult`

Reuse `interpreter.example.build_prompt_ids` + greedy generate (mirror `interpreter/score.py::score`,
which is the ground truth for how this model is driven), then parse **defensively**.

```python
import torch
from data_pipeline.attribute_spec import canonicalize, serialize
from eval.refuse_taxonomy import ROUTE_GRADE, ROUTE_CLARIFY, ROUTE_REFUSE
from interpreter.comparator import _safe_parse   # returns None on ANY malformation (no silent grade)
from interpreter.example import build_prompt_ids

def route_and_spec(self, prompt: str) -> RouteResult:
    ids = build_prompt_ids(self.interp_tok, prompt)
    with torch.no_grad():
        out = self.interp_model.generate(
            torch.tensor([ids]).to(self.interp_device),
            max_new_tokens=self.cfg.interpreter.max_new_tokens, do_sample=False,
            eos_token_id=self.interp_tok.eos_token_id,
            pad_token_id=self.interp_tok.pad_token_id)
    pred_text = self.interp_tok.decode(out[0][len(ids):], skip_special_tokens=True)

    spec = _safe_parse(pred_text)
    if spec is None:                                   # unparseable → safest is CLARIFY, never grade
        return RouteResult(ROUTE_CLARIFY, None, None, self._clarify_message(prompt), None)
    spec = canonicalize(spec)

    if spec.route == ROUTE_REFUSE:
        return RouteResult(ROUTE_REFUSE, spec.refuse_reason, None, None, spec)
    if spec.route == ROUTE_CLARIFY:
        return RouteResult(ROUTE_CLARIFY, None, None, self._clarify_message(prompt), spec)
    # grade
    return RouteResult(ROUTE_GRADE, None, serialize(spec), None, spec)
```

Notes / rationale (grounded):
- `_safe_parse` (from `interpreter/comparator.py`) is used because `attribute_spec.parse` **defaults
  grammar-less gibberish to `route=grade`** and raises on bad floats. We must never silently grade —
  `None` → clarify.
- `pad_token_id` may be `None` on a fresh tokenizer; the interpreter loader (§5.1) points pad→eos, same
  as `interpreter/example.py::resolve_eos_and_pad`.
- `spec_text` returned for grade is the **canonical** `serialize(spec)`. If `generator.spec_bucketize`
  is true, the *conditioning* text will instead be `serialize_bucketed(spec)` (chosen in `generate_lut`),
  but the **scoring** spec stays canonical.

### 1.4 `generate_lut(cond_text, spec_text, image) -> LutResult` — best-of-N + decode

This is the deployable quality path. The generator collapses under greedy decoding (exposure bias,
`eval/behavioral_fidelity.py` docstring), so best-of-N reranked by behavioral fidelity is the fix.

```python
from eval.behavioral_fidelity import decode_codes
from eval.best_of_n import best_of_n_codes
from sft.generate import generate_codes

def generate_lut(self, cond_text: str, spec_text: str, image) -> LutResult:
    g = self.cfg.generator
    best_codes, record = best_of_n_codes(
        self.gen_model, self.gen_processor,
        image=image,
        cond_text=cond_text,            # what the model is CONDITIONED on (input_mode-selected)
        spec_text=spec_text,            # what candidates are SCORED against (canonical spec)
        n=g.best_of_n_N, sampling=g.sampling, chunk=g.chunk,
        device=self.device)

    fell_back = False
    if best_codes is None:              # every one of N candidates refused/malformed
        fell_back = True
        best_codes = generate_codes(self.gen_model, self.gen_processor,
                                    image=image, text=cond_text, sampling=None, device=self.device)
        if best_codes is None:          # still a refusal → let run() convert to a soft clarify
            raise GeneratorRefused(cond_text)
        from eval.behavioral_fidelity import score_from_lut
        record = score_from_lut(decode_codes(best_codes, final_dir=self.vq_final_dir),
                                spec_text, codes=best_codes)

    lut = decode_codes(best_codes, final_dir=self.vq_final_dir)   # [17,17,17,3], clipped [0,1]
    return LutResult(best_codes, lut, record, fell_back)
```

`best_of_n_codes` returns `(None, {"refused_all": True})` when all candidates refuse; handle that
(fallback → greedy → soft clarify). When it succeeds it returns the reranker-best valid candidate
(`max(..., key=rerank_key)`); `record` carries `behavioral_fidelity`, `collapsed`, `code_stats`,
which the UI can surface as a quality badge.

### 1.5 The router-only decision — one-stage vs two-stage conditioning (config-select)

The interpreter is a **router**; the *grade* path chooses what text the generator is conditioned on.
Both are valid and the research supports documenting both — select via `generator.input_mode`:

| `input_mode` | `cond_text` passed to the generator | When to use | Backing evidence |
|---|---|---|---|
| `attribute_spec_text` | interpreter's `serialize(spec)` (or `serialize_bucketed` if `spec_bucketize`) | **P6 default**; adapter trained on `attribute_spec_text` (`configs/candidate_two_stage.json`) | ADR 0021 two-stage seam |
| `instruction` | the **raw user prompt** | one-stage adapter (`input_field="instruction"`); the research recommendation for grade magnitude — *vague spec loses magnitude, raw text keeps it* | `docs/interpreter_results.md` §"What to do next" #1 |
| `instruction_and_spec` | `f"{prompt}\n{serialize(spec)}"` | hybrid: NL anchor + precise numbers (`sft.example.input_text_for`) | `SFTConfig.input_field` |

Crucial: **`spec_text` (the SCORING/rerank target) is ALWAYS the canonical interpreter spec**, even
when `cond_text` is the raw prompt. That is why the interpreter still runs on the grade path in
one-stage mode — it supplies the reranker's target so best-of-N can pick the on-request candidate.
`cond_text` selection:

```python
def _cond_text(self, prompt, spec) -> str:
    from data_pipeline.attribute_spec import serialize, serialize_bucketed
    mode = self.cfg.generator.input_mode
    spec_str = (serialize_bucketed if self.cfg.generator.spec_bucketize else serialize)(spec)
    if mode == "instruction":         return prompt
    if mode == "instruction_and_spec": return f"{prompt}\n{spec_str}"
    return spec_str                    # "attribute_spec_text" (default)
```

> Invariant (01_architecture §6.3): `input_mode`/`spec_bucketize` MUST match how the adapter was
> trained, or conditioning drifts from training. Ship the P6 adapter with `attribute_spec_text`.

### 1.6 `run(prompt, image, run_dir) -> dict` — full flow → API payload

```python
def run(self, prompt, image, run_dir) -> dict:
    r = self.route_and_spec(prompt)
    feedback = self.terms.prompt_feedback(prompt, r)      # webapp/terms.py (§6)

    if r.route == ROUTE_REFUSE:
        return {"route": "refuse", "refuse_reason": r.refuse_reason,
                "clarify_message": None, "attribute_spec_text": None,
                "lut": None, "previews": [], "prompt_feedback": feedback}
    if r.route == ROUTE_CLARIFY:
        return {"route": "clarify", "refuse_reason": None,
                "clarify_message": r.clarify_message, "attribute_spec_text": None,
                "lut": None, "previews": [], "prompt_feedback": feedback}

    # grade
    cond = self._cond_text(prompt, r.spec)
    try:
        out = self.generate_lut(cond, r.spec_text, image)
    except GeneratorRefused:
        return {"route": "clarify", "refuse_reason": None,
                "clarify_message": "The model could not produce a confident grade for this request. "
                                   "Try a more specific look.", "attribute_spec_text": r.spec_text,
                "lut": None, "previews": [], "prompt_feedback": feedback}

    from webapp.lut import apply_lut, export_cube, save_image, load_image
    cube_path = f"{run_dir}/output.cube"
    export_cube(out.lut, cube_path)

    previews = [self._preview(image, out.lut, run_dir, "user_image")]
    for name, ref in self._reference_images():            # 6 neutral refs from assets/references
        previews.append(self._preview(ref, out.lut, run_dir, name))

    return {"route": "grade", "refuse_reason": None, "clarify_message": None,
            "attribute_spec_text": r.spec_text,
            "lut": {"cube_url": f"/runs/{run_id}/output.cube"},
            "previews": previews,
            "prompt_feedback": feedback,
            "quality": {"behavioral_fidelity": out.record.get("behavioral_fidelity"),
                        "collapsed": out.record.get("collapsed"),
                        "fell_back_greedy": out.fell_back}}

def _preview(self, pil_img, lut, run_dir, name) -> dict:
    from webapp.lut import apply_lut, save_image
    import numpy as np
    orig = np.asarray(pil_img.convert("RGB"), dtype=np.float64) / 255.0
    graded = apply_lut(orig, lut)                          # trilinear, see webapp/lut.py
    save_image(pil_img, f"{run_dir}/{name}_original.png")
    save_image((graded * 255).round().astype("uint8"), f"{run_dir}/{name}_graded.png")
    return {"name": name,
            "original_url": f"/runs/{run_id}/{name}_original.png",
            "graded_url":   f"/runs/{run_id}/{name}_graded.png"}
```

(`run_id` = the last path segment of `run_dir`.)

---

## 2. `webapp/lut.py` — decode, apply, export

All three are thin wrappers over existing, tested repo functions. LUT nodes are **encoded-sRGB in
[0,1]** (`eval/cube_io.py` header); images are sRGB 8-bit → divide by 255 in, ×255 out.

### 2.1 `decode(codes) -> np.ndarray`  (REUSE)

```python
from eval.behavioral_fidelity import decode_codes

def decode(codes, *, final_dir=None):
    """64 codebook indices → absolute LUT [17,17,17,3], clipped to [0,1]."""
    return decode_codes(codes, final_dir=final_dir)
```

Do not build a decoder. `decode_codes` lazily loads the frozen VQ-VAE (`tokenizer.frozen.load_frozen_vqvae`,
`lru_cache`d), calls `VQVAE.decode(codes)` → residual `[17,17,17,3]`, adds the identity grid
(`residual_to_absolute`), and clips to [0,1] — matching the notebooks' apply path.

### 2.2 `apply_lut(image_rgb, lut) -> np.ndarray`  (REUSE `apply_lut_trilinear`)

**The canonical trilinear apply already exists** — `data_pipeline/lut_ops.py::apply_lut_trilinear`
(scipy `RegularGridInterpolator` on the canonical grid). Reuse it; do not reinvent.

```python
import numpy as np
from data_pipeline.lut_ops import apply_lut_trilinear

def apply_lut(image_rgb, lut):
    """image_rgb: float ndarray [...,3] in [0,1] (encoded sRGB). lut: absolute [17,17,17,3].
    Returns graded image, same shape, clipped to [0,1]."""
    out = apply_lut_trilinear(lut, image_rgb)     # trilinear on the 17-node grid, inputs clipped
    return np.clip(out, 0.0, 1.0)
```

**Reference sketch of what `apply_lut_trilinear` does** (for understanding — DO NOT paste a
reimplementation; call the repo function):

```python
# lut_ops.apply_lut_trilinear (already in the repo):
n = lut.shape[0]                                  # 17
axis = np.linspace(0.0, 1.0, n)                   # node i -> i/(n-1)
flat = np.clip(rgb, 0, 1).reshape(-1, 3)
out = np.empty_like(flat)
for ch in range(3):
    interp = RegularGridInterpolator((axis, axis, axis), lut[..., ch],
                                     method="linear", bounds_error=False, fill_value=None)
    out[:, ch] = interp(flat)                      # trilinear interpolation of channel ch
out = out.reshape(rgb.shape)
```

Optional acceleration: `colour-science` / `opencv` live in the `[color]` extra
(`pyproject.toml`: `colour-science>=0.4`, `opencv-python-headless>=4.8`). They are **not required** —
the scipy path is dependency-free (scipy is a core dep) and correct. Only reach for `[color]` if a
profiler shows the apply is a bottleneck; if you do, keep node semantics identical (encoded-sRGB [0,1],
`lut[r,g,b,ch]`, node `i→i/16`) and verify against `apply_lut_trilinear` on a random grid.

### 2.3 `export_cube(lut, path)`  (REUSE `write_cube`)

**The canonical `.cube` serializer already exists** — `eval/cube_io.py::write_cube` /
`serialize_cube`. It emits exactly the pinned format: `LUT_3D_SIZE 17`, `DOMAIN_MIN 0 0 0`,
`DOMAIN_MAX 1 1 1`, **R-fastest** table order (b outer, g mid, r inner), 10-decimal floats, LF endings.

```python
from eval.cube_io import write_cube

def export_cube(lut, path):
    """Write a valid 17^3 .cube. lut is absolute [17,17,17,3] in [0,1]."""
    write_cube(path, np.clip(np.asarray(lut, dtype=np.float64), 0.0, 1.0))
```

Do not hand-format `.cube` text. `serialize_cube` validates the shape (`[N,N,N,3]`, cube grid) and
raises on a malformed LUT. Roundtrip is guaranteed by `parse_cube` (used in the acceptance test).

### 2.4 Image IO helpers

```python
from PIL import Image
import numpy as np

def load_image(path) -> Image.Image:
    return Image.open(path).convert("RGB")

def save_image(img, path):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)                # expects uint8 [H,W,3]
    img.save(path)
```

---

## 3. Neutral reference images (`webapp/assets/references/`)

Ship **6 neutral references** so a grade is visible on varied content, not just the user's upload.
Suggested set (any royalty-free/neutral sRGB images are fine): a **portrait / skin tones**, a
**landscape**, a **neutral grey ramp / step wedge**, a **color chart (e.g. a synthetic 24-patch)**, an
**indoor/mixed-white scene**, and a **high-key/low-key** shot. The grey ramp + color chart make
temperature/tint/contrast moves legible; the ramp can be generated with `eval.cube_io.identity_grid`
sampling if no asset is handy. Store as `.png`; `pipeline._reference_images()` enumerates them.

---

## 4. FastAPI surface (`webapp/server.py`) — how the pipeline is wired

Detailed frontend/server design is in 01_architecture; the pipeline-relevant contract:

- One `PromptToLutPipeline` instance created at startup (module global or FastAPI lifespan), so models
  load once (01_architecture §5).
- `POST /api/generate` (multipart: `image`, `prompt`): create `run_dir = webapp/_runs/<uuid>`, save the
  upload, `pil = webapp.lut.load_image(...)`, `payload = pipeline.run(prompt, pil, run_dir)`, return
  `payload` as JSON. Mount `webapp/_runs` at `/runs` via `StaticFiles`; mount `webapp/static` at `/`.
- `GET /api/terms` → `pipeline.terms.all_terms()` (§6).
- `GET /api/health` → `pipeline.self_check()`.
- Response JSON exactly matches the shared contract:
  ```json
  {"route": "grade|clarify|refuse",
   "refuse_reason": "out_of_scope|out_of_gamut|null",
   "clarify_message": "…|null",
   "attribute_spec_text": "route=grade | warmer=+2.3 muted=+2.0 … | null",
   "lut": {"cube_url": "/runs/<id>/output.cube"} ,
   "previews": [{"name": "user_image", "original_url": "…", "graded_url": "…"}, …],
   "prompt_feedback": {"assessment": "…",
       "suggested_terms": [{"term":"warmer","axis":"temperature_delta_b",
                            "definition":"…","example_usage":"make it noticeably warmer",
                            "grounded": true}]}}
  ```

---

## 5. `webapp/models_config.py` — the two loaders (device-branched)

### 5.1 `load_interpreter(cfg) -> (model, tokenizer, device)`

Mirror `interpreter/score.py::_load_model`, extended for `mps`/`cpu`:

```python
def load_interpreter(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ic = cfg.interpreter
    device = cfg.device
    dtype = torch.float32 if device == "cpu" else torch.float16   # 0.5B; fp16 fine on mps/cuda
    tok = AutoTokenizer.from_pretrained(ic.model_path)
    if tok.pad_token_id is None:                    # interpreter/example.resolve_eos_and_pad
        tok.pad_token = tok.eos_token
    if ic.tuning_mode == "lora":
        from peft import PeftModel
        base = AutoModelForCausalLM.from_pretrained(ic.base_model_id, torch_dtype=dtype)
        model = PeftModel.from_pretrained(base, ic.model_path)
    else:                                           # "full" (default) — load the dir directly
        model = AutoModelForCausalLM.from_pretrained(ic.model_path, torch_dtype=dtype)
    return model.to(device).eval(), tok, device
```

### 5.2 `load_generator(cfg) -> (model, processor)`  — CUDA 4-bit vs mps/cpu fp16/fp32

This is the portability crux (01_architecture §7): **bitsandbytes 4-bit is CUDA-only**.

```python
def load_generator(cfg):
    g = cfg.generator
    if cfg.device == "cuda" and g.load_in_4bit:
        # REUSE the repo's canonical 4-bit loader. SFTConfig() supplies min/max_pixels + bnb knobs.
        from sft.config import SFTConfig
        from sft.loader import load_eval_model
        sft_cfg = SFTConfig(max_pixels=g.max_pixels, min_pixels=g.min_pixels)
        return load_eval_model(sft_cfg, g.resized_base_path, g.adapter_path)

    # mps / cpu: fp16 (mps) or fp32 (cpu); NO bitsandbytes.
    import torch
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
    except Exception:
        from transformers import AutoModelForVision2Seq as ModelCls
    dtype = torch.float32 if cfg.device == "cpu" else torch.float16
    processor = AutoProcessor.from_pretrained(g.resized_base_path, trust_remote_code=True,
                                              min_pixels=g.min_pixels, max_pixels=g.max_pixels)
    base = ModelCls.from_pretrained(g.resized_base_path, torch_dtype=dtype, trust_remote_code=True)
    if g.adapter_path:                              # None → a merged/distilled standalone model
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, g.adapter_path)
        model = model.merge_and_unload()            # fold LoRA in → faster local inference
    else:
        model = base
    return model.to(cfg.device).eval(), processor
```

Correctness notes:
- **Always load `resized_base_path` (`models/base_resized`), never the vanilla base** — the adapter
  references the 259 added token rows (`sft/config.py::num_new_tokens = 259`); the vanilla base lacks
  the `<lut_*>` embeddings and generation breaks.
- `min_pixels`/`max_pixels` must be passed to `AutoProcessor` (the SFT path sets them via
  `load_eval_model`); mismatched pixel caps change vision tokenization vs training.
- `device=None` may be passed down to `best_of_n_codes`/`generate_codes`; they fall back to
  `model.device`. Passing `cfg.device` explicitly is fine and clearer.

---

## 6. `webapp/terms.py` — grounded prompt-improvement feature

The research finding is that the generator acts best on **specific, grounded** direction+intensity
words. `terms.py` exposes the repo's canonical vocabulary and turns a route result into user guidance.
It is pure/stdlib (imports only `eval/tag_vocabulary.py` + `data_pipeline/attribute_spec.py`).

```python
from data_pipeline.attribute_spec import _MAG_BUCKETS   # ((1.5,"slight"),(3.0,"moderate"),(6.0,"strong")) → +extreme
from eval.tag_vocabulary import DIRECTIONAL_TAG_AXIS, STYLE_TAGS, HUE_SECTORS, KNOWN_TAGS

INTENSITY_WORDS = ["slight", "moderate", "strong", "extreme"]   # from _MAG_BUCKETS + extreme

def all_terms() -> list[dict]:
    """GET /api/terms — every grounded term the generator understands, grouped by axis."""
    out = []
    for tag, (axis, sign) in DIRECTIONAL_TAG_AXIS.items():
        out.append({"term": tag, "axis": axis, "sign": sign, "grounded": True,
                    "definition": _DEFN.get(tag, tag.replace('_',' ')),
                    "example_usage": f"make it {INTENSITY_WORDS[2]} {tag.replace('_',' ')}"})
    for s in STYLE_TAGS:
        out.append({"term": s, "axis": "style_bundle", "grounded": True, ...})
    return out

def prompt_feedback(prompt: str, route_result) -> dict:
    """assessment + suggested grounded terms, tuned to the route:
       - refuse  → explain it's not a single global color transform (or out of gamut);
                   suggest the closest supported global directions.
       - clarify → 'too vague to grade'; suggest a few grounded directions + an intensity word.
       - grade   → if the prompt lacked an intensity word, suggest adding one (why: magnitude is
                   under-determined by vague text — docs/interpreter_results.md); echo the parsed spec.
    """
    ...
```

- **`suggested_terms`** items carry `{term, axis, definition, example_usage, grounded}` and are drawn
  ONLY from `DIRECTIONAL_TAG_AXIS`/`STYLE_TAGS`/`HUE_SECTORS` (so `grounded:true` is truthful; never
  invent a term). Retired aliases are mapped via `canonicalize_tag` before display.
- The clarify/refuse `assessment` should name *why* (route-specific) and point at supported directions,
  turning a dead-end into a next step — the product's core UX per the research.

---

## 7. Acceptance criteria

**Pipeline (grade path):**
- [ ] For a specific grade prompt (e.g. *"make it strongly warmer and a bit more matte"*) with a test
      image, `pipeline.run` returns `route=="grade"`, a non-null `attribute_spec_text`, a `lut.cube_url`,
      and `len(previews) == 7` (user image + 6 references), each with distinct `original_url`/`graded_url`.
- [ ] `generate_lut` returns a LUT of shape `(17,17,17,3)` with values in `[0,1]`, and
      `record["behavioral_fidelity"]` is a float in `[0,1]` (or the greedy fallback triggers with
      `fell_back_greedy==true`).
- [ ] The graded preview is **visibly different** from the original (not an identity no-op) for a
      non-trivial grade — i.e. `record["collapsed"] is False` on a good candidate; if `collapsed`,
      best-of-N still returns the least-collapsed pick and the UI shows a low-fidelity badge.

**Router short-circuits:**
- [ ] A refuse prompt (e.g. *"remove the person on the left"*) → `route=="refuse"`,
      `refuse_reason ∈ {out_of_scope, out_of_gamut}`, `previews==[]`, `lut is None`, **generator not called**.
- [ ] A clarify prompt (e.g. *"make it better"*) → `route=="clarify"`, non-null `clarify_message`,
      `prompt_feedback.suggested_terms` non-empty and all `grounded:true`, **generator not called**.
- [ ] An unparseable interpreter output routes to `clarify` (never a silent grade) — via `_safe_parse`.

**LUT / `.cube`:**
- [ ] `export_cube(lut, path)` produces a file whose header is `LUT_3D_SIZE 17` / `DOMAIN_MIN 0 0 0` /
      `DOMAIN_MAX 1 1 1` with `17**3 = 4913` data rows in R-fastest order.
- [ ] Roundtrip: `parse_cube(open(path,'rb').read())[0]` reproduces the LUT within float tolerance
      (reuse `eval.cube_io.parse_cube`).
- [ ] The `.cube` **opens in a real LUT viewer** (e.g. DaVinci Resolve / an online `.cube` viewer /
      an OCIO tool) and applies a plausible grade — the manual end-to-end check.

**Reuse discipline:**
- [ ] `webapp/lut.py` imports and calls `apply_lut_trilinear`, `write_cube`, and `decode_codes` — it
      contains **no** hand-rolled interpolation, `.cube` string formatting, or VQ-VAE construction.
- [ ] `generate_lut` calls `eval.best_of_n.best_of_n_codes` (not a bespoke sampling loop).
- [ ] No bitsandbytes/4-bit code path is reachable when `cfg.device != "cuda"`.
