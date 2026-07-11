"""Captioning helpers + captioner assembly (ADR 0026) — GPU/teacher-free."""

from __future__ import annotations

import importlib
import json

from data_pipeline import captioning as C
from data_pipeline.behavior_vector import measure_behavior
from data_pipeline.sources import procedural as proc

gc = importlib.import_module("scripts.generate_captions")


def _mb(name):
    lut = proc.generate_lut_tensor(next(s for s in proc.catalog() if s.lut_id == name))
    return measure_behavior(lut)


def test_caption_target_matches_attribute_spec():
    mb = _mb("proc_attr_warmer")
    text = C.caption_target_text(mb)
    assert text.startswith("route=grade")
    assert "warmer=+" in text


def test_caption_target_is_grounded():
    ok, issues = C.caption_is_grounded(_mb("proc_style_teal-orange"))
    assert ok, issues


def test_prompts_reference_styles_and_measured_look():
    mb = _mb("proc_attr_muted")
    sysp = C.build_caption_system_prompt(len(C.CAPTION_STYLES))
    assert "EXACTLY 5" in sysp and "GLOBAL" in sysp
    user = C.build_caption_user_text(mb, title="Faded Kodak")
    for style in C.CAPTION_STYLES:
        assert style in user
    assert "attribute_spec" in user and "Faded Kodak" in user


def test_validate_caption():
    assert C.validate_caption("Make it warm and faded like an old photo.")[0]
    assert not C.validate_caption("")[0]
    assert not C.validate_caption("hi")[0]
    assert not C.validate_caption("x" * 401)[0]


def test_caption_rows_from_cache_flattens_styles():
    cache = {
        "lut_a": {"status": "generated", "attribute_spec_text": "route=grade | warmer=+2.0",
                  "captions": {"literal": "Make it warmer.", "slang": "give it some warmth",
                               "bad": "  "}},
        "lut_b": {"status": "error", "error": "boom"},
    }
    rows = gc._caption_rows_from(cache)
    # 2 valid captions from lut_a (the blank one dropped), 0 from the errored lut_b
    assert len(rows) == 2
    assert all(r["attribute_spec_text"] == "route=grade | warmer=+2.0" for r in rows)
    assert all(r["route"] == "grade" and r["source_lut_id"] == "lut_a" for r in rows)
    assert {r["style"] for r in rows} == {"literal", "slang"}


def test_build_messages_no_image_is_teacher_free():
    row = {"source_lut_id": "x", "measured_behavior": _mb("proc_attr_warmer"), "image_path": None}
    client = gc.CaptionTeacherClient(attach_image=False)
    msgs = client.build_messages(row, C.CAPTION_STYLES)
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    text_parts = [p for p in msgs[1]["content"] if p.get("type") == "text"]
    assert text_parts and "attribute_spec" in text_parts[0]["text"]


def test_load_done_ignores_error_rows_so_they_retry(tmp_path):
    # An error record must NOT count as done — otherwise a cred-less first run permanently
    # poisons resume (the whole point of the Phase-0 fix).
    cache = tmp_path / "caption_cache.jsonl"
    cache.write_text(
        '{"lut_id": "ok_lut", "status": "generated", "attribute_spec_text": "route=grade", "captions": {"literal": "warm"}}\n'
        '{"lut_id": "bad_lut", "status": "error", "error": "boom"}\n',
        encoding="utf-8")
    done = gc._load_done(str(cache))
    assert "ok_lut" in done and "bad_lut" not in done


def test_load_done_prefers_later_generated_over_earlier_error(tmp_path):
    # A retried LUT: an early error line then a later generated line for the same id → done.
    cache = tmp_path / "caption_cache.jsonl"
    cache.write_text(
        '{"lut_id": "lut_a", "status": "error", "error": "boom"}\n'
        '{"lut_id": "lut_a", "status": "generated", "attribute_spec_text": "route=grade", "captions": {"slang": "pop"}}\n',
        encoding="utf-8")
    done = gc._load_done(str(cache))
    assert done["lut_a"]["status"] == "generated"


def test_run_hands_off_cleanly_when_endpoint_unresolvable(tmp_path, monkeypatch):
    # With the profile pinned but creds unset, resolve_endpoint raises → run must return 2 and
    # write NO cache/error rows (instead of burning every LUT into a status:"error" record).
    active = tmp_path / "active_rows.jsonl"
    active.write_text(json.dumps({
        "id": "r0", "source_lut_id": "lut_x", "is_supported": True,
        "measured_behavior": {"temperature_delta_b": 2.0},  # plain (JSON-safe); never captioned
    }) + "\n", encoding="utf-8")
    cache = tmp_path / "caption_cache.jsonl"
    out = tmp_path / "caption_rows.jsonl"

    def _boom(_profile):
        raise gc.openai_compat.OpenAICompatError("$TFY_BASE_URL is not set")

    monkeypatch.setattr(gc.openai_compat, "resolve_endpoint", _boom)
    # Force the YAML profile to look "available" regardless of the repo config.
    monkeypatch.setattr(gc.CaptionTeacherClient, "is_available", lambda self: True)

    rc = gc.run(str(active), str(out), str(cache), "configs/model_clients.yaml",
                limit=None, attach_image=False, dry_run=False)
    assert rc == 2
    assert not cache.exists() and not out.exists()
