"""Prompted-frontier baseline client — the runnable form of the gated ``prompted_frontier``
baseline (#8 in docs/adr/0011-baseline-comparisons.md).

Calls a frontier model with the source image + the edit instruction and asks it to emit a
canonical 17^3 ``.cube`` LUT, or the literal token ``<unsupported>``. Two transports:

  * ``openai_compatible`` (default): an OpenAI Chat Completions gateway such as TrueFoundry
    (``/v1/chat/completions``, Bearer auth, ``group/model`` slugs). Uses the ``openai`` SDK
    pointed at ``base_url``. This is the correct client for a gateway that speaks OpenAI's
    schema — the Anthropic SDK cannot talk to ``/v1/chat/completions``.
  * ``anthropic``: the Anthropic Messages API directly (api.anthropic.com), via the
    ``anthropic`` SDK — adaptive thinking + ``output_config.effort``.

Design notes:
  * reasoning effort = ``medium``. openai_compatible sends ``reasoning_effort`` (gateway may
    ignore/reject it — toggle ``send_effort``); anthropic sends ``output_config.effort``.
  * A 17^3 .cube body is 17**3 = 4913 float-triple rows — a very large output. Both paths
    stream, and ``max_tokens`` is set near the 128K cap. A gateway that caps output lower
    will truncate (recorded as finish_reason=length -> invalid).
  * A provider safety decline (finish_reason ``content_filter`` / stop_reason ``refusal``)
    is recorded distinctly from the model choosing to emit ``<unsupported>``.

SDKs are imported lazily so the eval/scoring path imports without them; only generation
needs one (``pip install 'slm-eval[frontier]'`` + credentials).
"""

from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Optional

import yaml

GRID_SIZE_DEFAULT = 17


def build_system_prompt(grid_size: int = GRID_SIZE_DEFAULT) -> str:
    """The output contract + supported/unsupported policy, stated precisely.

    Grounded in docs/detailed_behavior_spec.md (supported attributes, style decomposition,
    the unsupported list) and eval/cube_io.py (canonical serialization contract).
    """
    n = grid_size
    rows = n * n * n
    return f"""You are a precise color-grading engine. Given a source image and a single \
natural-language editing instruction, you output ONE global 3D color lookup table (LUT) \
that applies the requested look, OR a refusal token.

A 3D LUT is *global*: it maps every input RGB value to one output RGB value, the same way \
in every pixel of the image. You cannot see or address regions, objects, or subjects — \
only remap colors globally.

OUTPUT CONTRACT — output EXACTLY ONE of the following, and NOTHING else (no prose, no \
explanation, no markdown, no ``` fences):

(A) A canonical .cube LUT in this exact format:
    - First line:  LUT_3D_SIZE {n}
    - Then:        DOMAIN_MIN 0 0 0
    - Then:        DOMAIN_MAX 1 1 1
    - Then EXACTLY {rows} data lines ({n}x{n}x{n}), one per line, each three floats in \
[0, 1] separated by single spaces, e.g. "0.5019608 0.4980392 0.4901961".
    - Table order: R varies FASTEST, then G, then B. Equivalently, loop B outermost, \
then G, then R innermost. Data row index i = b*{n}*{n} + g*{n} + r maps input color \
(r/{n-1}, g/{n-1}, b/{n-1}).
    - Values are display-encoded sRGB in [0, 1]. The identity LUT maps each input node \
(r,g,b)/{n-1} to itself; build your edit as a modification of that identity.
    - You MUST emit all {rows} rows. Do not summarize, truncate, or use "...".

(B) The single token:  <unsupported>
    with no other characters.

SUPPORTED (produce a LUT): global tone and color grades that one global LUT can represent \
— temperature (warmer/cooler), tint (magenta/green), exposure (brighter/darker), contrast, \
black point (lift/crush blacks), highlights, shadows, saturation, and named looks that \
decompose into those (e.g. matte, faded, filmic, cinematic, teal-orange, sepia, bleach \
bypass). Make the requested change clearly visible while keeping the LUT smooth and \
monotone, keeping neutrals neutral unless the edit is itself a global color cast, and \
avoiding heavy clipping.

UNSUPPORTED (output <unsupported>): anything a single global LUT cannot do — local or \
region edits, semantic object recoloring, subject-only or background-only changes, \
inpainting/removal/adding content, relighting or changed light direction, geometry/crop/\
perspective, texture/detail edits (sharpen, denoise, blur, skin smoothing), reference-image \
style transfer, or any mixed prompt that combines a supported global change with an \
unsupported component.

Refuse (output <unsupported>) ONLY when the edit is genuinely not globally representable — \
one of the local / regional / semantic / content changes listed above. Do NOT refuse \
because producing the full table is large, tedious, or impossible to compute exactly: for a \
global tone/color request, an approximate but smooth, correctly-directed LUT is the expected \
and correct answer. If the instruction is a global tone or color change (temperature, tint, \
exposure, contrast, black point, highlights, shadows, saturation, or a look that decomposes \
into those), you MUST output a LUT and MUST NOT output the refusal token."""


USER_INSTRUCTION_TEMPLATE = (
    "Edit instruction: {instruction}\n\n"
    "Output the {n}x{n}x{n} .cube LUT that applies this look to the image above, "
    "or <unsupported> if it is not globally representable. Output only the LUT text "
    "or the token."
)


@dataclass
class GenerationResult:
    text: Optional[str]
    provenance: dict


def _media_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".webp": "image/webp", ".gif": "image/gif"}.get(ext, "image/png")


def encode_image(path: str, min_edge: int = 256) -> tuple[str, str]:
    """Return (base64, media_type). Small images are upscaled (nearest-neighbor, integer
    factor) to at least ``min_edge`` on the short side so provider image validators don't
    reject them — the eval fixtures are 32x32. Nearest-neighbor preserves the exact pixel
    colors the model must grade (bilinear would blur them)."""
    import io

    from PIL import Image

    im = Image.open(path)
    w, h = im.size
    if min(w, h) < min_edge:
        import math

        scale = int(math.ceil(min_edge / min(w, h)))
        im = im.convert("RGB").resize((w * scale, h * scale), Image.NEAREST)
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        data = buf.getvalue()
        media = "image/png"
    else:
        with open(path, "rb") as fh:
            data = fh.read()
        media = _media_type(path)
    return base64.standard_b64encode(data).decode("utf-8"), media


def load_frontier_config(path: str = "configs/model_clients.yaml") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Create it from the template to enable the prompted-frontier "
            "baseline (see configs/model_clients.yaml)."
        )
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    pf = cfg.get("prompted_frontier")
    if not pf or not pf.get("models"):
        raise ValueError(f"{path}: prompted_frontier.models is empty")
    return pf


class FrontierClient:
    """OpenAI-compatible-gateway or Anthropic-native client for the raw-.cube baseline."""

    def __init__(self, config_path: str = "configs/model_clients.yaml"):
        self.cfg = load_frontier_config(config_path)
        self.provider = self.cfg.get("provider", "openai_compatible")
        self.base_url = self.cfg.get("base_url")
        self.api_key_env = self.cfg.get("api_key_env", "TFY_API_KEY")
        self.grid_size = int(self.cfg.get("grid_size", GRID_SIZE_DEFAULT))
        self.max_tokens = int(self.cfg.get("max_tokens", 120000))
        self.min_image_edge = int(self.cfg.get("min_image_edge", 256))
        self.send_effort = bool(self.cfg.get("send_effort", True))
        self.thinking = self.cfg.get("thinking", "adaptive")
        self.prompt_version = self.cfg.get("prompt_version", "frontier_raw_cube_v1")
        self.system_prompt = build_system_prompt(self.grid_size)
        self._client = None  # lazily constructed

    @property
    def models(self) -> list[dict]:
        return list(self.cfg["models"])

    def model_by_name(self, name: str) -> dict:
        for m in self.models:
            if m["name"] == name:
                return m
        raise KeyError(f"model '{name}' not in prompted_frontier.models")

    def generate(self, model_entry: dict, image_path: str, instruction: str) -> GenerationResult:
        if self.provider == "anthropic":
            return self._generate_anthropic(model_entry, image_path, instruction)
        return self._generate_openai(model_entry, image_path, instruction)

    # --- OpenAI-compatible gateway (TrueFoundry etc.) ---------------------------
    def _openai_client(self):
        if self._client is None:
            try:
                import openai
            except ImportError as exc:  # noqa: BLE001
                raise RuntimeError(
                    "openai SDK not installed. Run: pip install 'slm-eval[frontier]' "
                    "(or `pip install openai`)."
                ) from exc
            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"${self.api_key_env} is not set (Bearer token for {self.base_url}).")
            # A full 17^3 .cube is ~100K output tokens; a stream can run many minutes. Raise
            # the SDK's 10-min default so it doesn't abort a legitimately long generation.
            self._client = openai.OpenAI(base_url=self.base_url, api_key=key,
                                         timeout=float(self.cfg.get("request_timeout_s", 1800)),
                                         max_retries=1)
        return self._client

    def _generate_openai(self, model_entry, image_path, instruction) -> GenerationResult:  # noqa: ANN001
        client = self._openai_client()
        b64, media = encode_image(image_path, self.min_image_edge)
        data_uri = f"data:{media};base64,{b64}"
        user_text = USER_INSTRUCTION_TEMPLATE.format(instruction=instruction, n=self.grid_size)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": user_text},
            ]},
        ]
        kwargs = dict(model=model_entry["model_id"], messages=messages,
                      max_tokens=self.max_tokens, stream=True,
                      stream_options={"include_usage": True})
        # reasoning_effort is opt-in per model (global default via send_effort); routes that
        # reject the param — Gemini via the shim commonly does — set send_effort: false.
        send_effort = model_entry.get("send_effort", self.send_effort)
        effort = model_entry.get("effort")
        if send_effort and effort:
            # via extra_body so it passes through regardless of SDK-typed support
            kwargs["extra_body"] = {"reasoning_effort": effort}

        parts: list[str] = []
        finish: Optional[str] = None
        usage = None
        stream = client.chat.completions.create(**kwargs)
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage = chunk.usage
            if not chunk.choices:
                continue
            ch = chunk.choices[0]
            if ch.delta and ch.delta.content:
                parts.append(ch.delta.content)
            if ch.finish_reason:
                finish = ch.finish_reason

        text = "".join(parts) or None
        api_refusal = finish == "content_filter"
        det = getattr(usage, "completion_tokens_details", None) if usage else None
        provenance = {
            "provider": "openai_compatible", "base_url": self.base_url,
            "model_id": model_entry["model_id"], "effort": model_entry.get("effort", "medium"),
            "prompt_version": self.prompt_version, "finish_reason": finish,
            "input_tokens": getattr(usage, "prompt_tokens", None) if usage else None,
            "output_tokens": getattr(usage, "completion_tokens", None) if usage else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage else None,
            "reasoning_tokens": getattr(det, "reasoning_tokens", None) if det else None,
            "api_refusal": api_refusal,
        }
        if api_refusal:
            text = None
        return GenerationResult(text=text, provenance=provenance)

    # --- Anthropic Messages API (direct) ----------------------------------------
    def _anthropic_client(self):
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # noqa: BLE001
                raise RuntimeError(
                    "anthropic SDK not installed. Run: pip install 'slm-eval[frontier]' "
                    "(or `pip install anthropic`) and set ANTHROPIC_API_KEY / `ant auth login`."
                ) from exc
            self._client = anthropic.Anthropic()
        return self._client

    def _generate_anthropic(self, model_entry, image_path, instruction) -> GenerationResult:  # noqa: ANN001
        client = self._anthropic_client()
        user_text = USER_INSTRUCTION_TEMPLATE.format(instruction=instruction, n=self.grid_size)
        b64, media = encode_image(image_path, self.min_image_edge)
        messages = [{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
            {"type": "text", "text": user_text},
        ]}]
        thinking = {"type": "adaptive"} if self.thinking == "adaptive" else {"type": "disabled"}
        with client.messages.stream(
            model=model_entry["model_id"], max_tokens=self.max_tokens,
            system=self.system_prompt, thinking=thinking,
            output_config={"effort": model_entry.get("effort", "medium")},
            messages=messages,
        ) as stream:
            msg = stream.get_final_message()
        text = "".join(b.text for b in msg.content if b.type == "text") or None
        provenance = {
            "provider": "anthropic", "model_id": model_entry["model_id"],
            "effort": model_entry.get("effort", "medium"), "prompt_version": self.prompt_version,
            "stop_reason": msg.stop_reason, "output_tokens": getattr(msg.usage, "output_tokens", None),
            "api_refusal": msg.stop_reason == "refusal",
        }
        if msg.stop_reason == "refusal":
            text = None
        return GenerationResult(text=text, provenance=provenance)
