"""Shared OpenAI-compatible chat transport for the pinned teacher + judge profiles.

The prompted-frontier baseline (:mod:`eval.frontier_client`) needs a *streaming* client for
the ~100K-token ``.cube`` generation, so it keeps its own client. The teacher
(:mod:`data_pipeline.instruction_gen`) and the L8 judge (:mod:`eval.judge_client`) instead
issue small, NON-streaming JSON calls to the same OpenAI-compatible gateway (TrueFoundry).
This module owns the parts they share so neither copies transport code: credential
resolution from env-var NAMES, lazy client construction, one non-streaming
``chat.completions`` call, and tolerant JSON parsing.

Unlike ``prompted_frontier`` (which hardcodes ``base_url``), ``teacher_primary`` /
``judge_primary`` reference the gateway by env-var NAME (``endpoint_env=TFY_BASE_URL``,
``api_key_env=TFY_API_KEY``); the URL + secret live only in the environment (``.env``),
never in the yaml (docs/data_collection_plan.md "Instruction Generation").

The ``openai`` SDK is imported lazily so importing this module (and, transitively, the data
pipeline / eval spine) never requires the ``[frontier]`` extra — only an actual call does.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

# A small text/JSON call is fast; keep a generous but finite ceiling so a wedged gateway
# does not hang a batch run. (The frontier .cube path uses its own 1800s timeout.)
DEFAULT_TIMEOUT_S = 120.0


class OpenAICompatError(RuntimeError):
    """Transport / configuration error for the shared OpenAI-compatible client."""


@dataclass
class ChatResult:
    """One non-streaming chat completion, normalized across SDK versions."""

    text: Optional[str]
    finish_reason: Optional[str]
    usage: dict = field(default_factory=dict)
    api_refusal: bool = False


def resolve_endpoint(profile: dict) -> tuple[str, str]:
    """Return ``(base_url, api_key)`` from the env-var NAMES a teacher/judge profile records.

    ``profile['endpoint_env']`` / ``profile['api_key_env']`` are the *names* of environment
    variables; their values live in the environment. Raises :class:`OpenAICompatError` with an
    actionable message when a name is missing from the profile or the referenced env var is
    unset — i.e. the profile is pinned but not runnable in this environment.
    """
    ep_name = profile.get("endpoint_env")
    key_name = profile.get("api_key_env")
    if not ep_name or not key_name:
        raise OpenAICompatError(
            "profile is missing endpoint_env/api_key_env (the env-var NAMES for the gateway "
            "base URL + API key)"
        )
    base_url = os.environ.get(str(ep_name))
    api_key = os.environ.get(str(key_name))
    if not base_url:
        raise OpenAICompatError(
            f"${ep_name} is not set (base URL for the OpenAI-compatible gateway)"
        )
    if not api_key:
        raise OpenAICompatError(f"${key_name} is not set (Bearer token for the gateway)")
    return base_url, api_key


def build_client(base_url: str, api_key: str, *, timeout: float = DEFAULT_TIMEOUT_S,
                 max_retries: int = 1):
    """Construct an ``openai.OpenAI`` client (lazy import; actionable error if the SDK is absent)."""
    try:
        import openai
    except ImportError as exc:  # noqa: BLE001
        raise OpenAICompatError(
            "openai SDK not installed. Run: pip install 'slm-eval[frontier]' (or `pip install openai`)."
        ) from exc
    return openai.OpenAI(base_url=base_url, api_key=api_key,
                         timeout=float(timeout), max_retries=max_retries)


def effort_from_profile(profile: dict) -> Optional[str]:
    """The ``reasoning_effort`` to send for a teacher/judge profile.

    Returns ``profile['effort']`` unless ``profile['send_effort']`` is False (a route that 400s
    on the param). None -> omit ``reasoning_effort`` entirely.
    """
    if not profile.get("send_effort", True):
        return None
    eff = profile.get("effort")
    return str(eff) if eff else None


def text_part(text: str) -> dict:
    """A user-message text content part."""
    return {"type": "text", "text": text}


def image_part(data_uri: str) -> dict:
    """A user-message image content part from a ``data:`` URI."""
    return {"type": "image_url", "image_url": {"url": data_uri}}


def chat_completion(client, model: str, messages: list, *, max_tokens: int,
                    temperature: Optional[float] = None,
                    reasoning_effort: Optional[str] = None) -> ChatResult:
    """Issue one NON-streaming ``chat.completions`` call and normalize the response.

    ``temperature`` is sent only when provided (teacher 0.7, judge 0.0). ``reasoning_effort``
    is sent via ``extra_body`` only when provided (teacher/judge profiles set no effort, so it
    is omitted by default). Returns text / finish_reason / usage; ``api_refusal`` is True when
    the gateway reports ``finish_reason == "content_filter"``.
    """
    kwargs: dict = dict(model=model, messages=messages, max_tokens=int(max_tokens))
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if reasoning_effort:
        kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
    # Normalize every SDK failure (4xx/5xx/timeout/connection — all subclass openai.OpenAIError,
    # which is NOT a RuntimeError) into OpenAICompatError so callers' `except OpenAICompatError`
    # can downgrade one row (teacher -> rejected, judge -> not_evaluated) instead of a single
    # transient error aborting an entire batch / the Stage-9 pipeline.
    try:
        import openai
    except ImportError as exc:  # noqa: BLE001
        raise OpenAICompatError(
            "openai SDK not installed. Run: pip install 'slm-eval[frontier]' (or `pip install openai`)."
        ) from exc
    try:
        resp = client.chat.completions.create(**kwargs)
    except openai.OpenAIError as exc:  # noqa: BLE001
        raise OpenAICompatError(f"chat.completions.create failed: {exc}") from exc

    text: Optional[str] = None
    finish: Optional[str] = None
    choices = getattr(resp, "choices", None) or []
    if choices:
        choice = choices[0]
        finish = getattr(choice, "finish_reason", None)
        msg = getattr(choice, "message", None)
        text = getattr(msg, "content", None) if msg is not None else None

    usage: dict = {}
    u = getattr(resp, "usage", None)
    if u is not None:
        det = getattr(u, "completion_tokens_details", None)
        usage = {
            "input_tokens": getattr(u, "prompt_tokens", None),
            "output_tokens": getattr(u, "completion_tokens", None),
            "total_tokens": getattr(u, "total_tokens", None),
            "reasoning_tokens": getattr(det, "reasoning_tokens", None) if det else None,
        }
    return ChatResult(text=(text or None), finish_reason=finish, usage=usage,
                      api_refusal=(finish == "content_filter"))


def parse_json_object(text: Optional[str]) -> dict:
    """Parse a JSON object from a model response, tolerating ``` fences / surrounding prose.

    Raises :class:`OpenAICompatError` if no JSON object can be recovered.
    """
    if text is None:
        raise OpenAICompatError("empty model response")
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    try:
        obj = json.loads(s)
    except json.JSONDecodeError:
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise OpenAICompatError(f"no JSON object in response: {s[:200]!r}")
        try:
            obj = json.loads(s[start:end + 1])
        except json.JSONDecodeError as exc:
            raise OpenAICompatError(f"unparseable JSON in response: {s[:200]!r}") from exc
    if not isinstance(obj, dict):
        raise OpenAICompatError("response JSON is not an object")
    return obj
