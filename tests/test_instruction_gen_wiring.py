"""Wiring tests for the teacher + judge HTTP path.

Exercises the wired call end-to-end WITHOUT network by monkeypatching the shared transport's
lazy client factory (repo idiom: monkeypatch.setattr on the module function, not respx/httpx).
The openai SDK is never imported because build_client is replaced.
"""

import json

import pytest

from data_pipeline.active_dataset import assemble_active
from data_pipeline.errors import RequiresTeacher, TeacherGenerationError
from data_pipeline.instruction_gen import (
    TeacherClient,
    generate_instructions_for_rows,
)
from eval import judge_client, openai_compat


# --- fake OpenAI-compatible client ------------------------------------------------
class _Usage:
    prompt_tokens = 12
    completion_tokens = 34
    total_tokens = 46
    completion_tokens_details = None


class _Msg:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, content, finish="stop"):
        self.message = _Msg(content)
        self.finish_reason = finish


class _Resp:
    def __init__(self, content, finish="stop"):
        self.choices = [_Choice(content, finish)]
        self.usage = _Usage()


class _Completions:
    def __init__(self, content, finish):
        self._content = content
        self._finish = finish
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _Resp(self._content, self._finish)


class _Chat:
    def __init__(self, content, finish):
        self.completions = _Completions(content, finish)


class _FakeClient:
    def __init__(self, content, finish="stop"):
        self.chat = _Chat(content, finish)


def _patch_client(monkeypatch, content, finish="stop"):
    client = _FakeClient(content, finish)
    monkeypatch.setattr(openai_compat, "build_client", lambda *a, **k: client)
    return client


_TEACHER_CFG = (
    "teacher_primary:\n"
    "  provider: openai_compatible\n"
    "  model_id: claude-group/claude-sonnet-4-6\n"
    "  endpoint_env: TFY_TEST_URL\n"
    "  api_key_env: TFY_TEST_KEY\n"
    "  prompt_version: teacher_prompt_v1\n"
    "  batch_id: b1\n"
    "  effort: high\n"
    "  send_effort: true\n"
    "  request_defaults: { max_tokens: 256, temperature: 0.7 }\n"
)

_JUDGE_CFG = (
    "judge_primary:\n"
    "  provider: openai_compatible\n"
    "  model_id: claude-group/claude-opus-4-8\n"
    "  endpoint_env: TFY_TEST_URL\n"
    "  api_key_env: TFY_TEST_KEY\n"
    "  prompt_version: judge_prompt_v1\n"
    "  batch_id: jb1\n"
    "  request_defaults: { max_tokens: 128, temperature: 0.0 }\n"
)

_ROW = {
    "id": "r1",
    "gold_tags": ["warmer"],
    "measured_behavior": {"temperature_delta_b": 3.0, "chroma_delta": -2.0},
    "image_path": None,
}


# --- transport primitive ----------------------------------------------------------
def test_parse_json_object_tolerates_fences():
    assert openai_compat.parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert openai_compat.parse_json_object('noise {"b": 2} tail') == {"b": 2}
    with pytest.raises(openai_compat.OpenAICompatError):
        openai_compat.parse_json_object("no json here")


def test_resolve_endpoint_requires_env(monkeypatch):
    monkeypatch.delenv("TFY_TEST_URL", raising=False)
    with pytest.raises(openai_compat.OpenAICompatError):
        openai_compat.resolve_endpoint({"endpoint_env": "TFY_TEST_URL", "api_key_env": "TFY_TEST_KEY"})


# --- teacher ----------------------------------------------------------------------
def test_teacher_generate_wired(tmp_path, monkeypatch):
    cfg = tmp_path / "mc.yaml"
    cfg.write_text(_TEACHER_CFG)
    monkeypatch.setenv("TFY_TEST_URL", "https://gateway.example/v1")
    monkeypatch.setenv("TFY_TEST_KEY", "sk-test")
    client = _patch_client(monkeypatch, json.dumps({
        "gold_tags": ["warmer", "muted", "bogus_tag"],
        "concise": "Make it warmer and more muted.",
        "natural": "Give it a warm, gentle look.",
    }))
    tc = TeacherClient(cfg)
    out = tc.generate(_ROW)
    # high reasoning effort is threaded through as reasoning_effort (extra_body)
    assert client.chat.completions.last_kwargs["extra_body"] == {"reasoning_effort": "high"}
    assert out["provenance"]["effort"] == "high"
    assert out["concise"] and out["natural"]
    # hallucinated tag dropped, known tags kept
    assert set(out["gold_tags"]) == {"warmer", "muted"}
    assert out["provenance"]["teacher_model_id"] == "claude-group/claude-sonnet-4-6"
    assert out["provenance"]["teacher_prompt_version"] == "teacher_prompt_v1"
    assert out["provenance"]["usage"]["output_tokens"] == 34


def test_teacher_generate_raises_when_env_unset(tmp_path, monkeypatch):
    cfg = tmp_path / "mc.yaml"
    cfg.write_text(_TEACHER_CFG)
    monkeypatch.delenv("TFY_TEST_URL", raising=False)
    monkeypatch.delenv("TFY_TEST_KEY", raising=False)
    tc = TeacherClient(cfg)
    assert tc.is_available() is True  # pure config check — env not required for availability
    with pytest.raises(RequiresTeacher):
        tc.generate(_ROW)


def test_dry_run_builds_prompts_without_network(tmp_path):
    # No client patched, no env set: dry_run must not touch the transport.
    cfg = tmp_path / "mc.yaml"
    cfg.write_text(_TEACHER_CFG)
    tc = TeacherClient(cfg)
    manifest = generate_instructions_for_rows([_ROW], tc, dry_run=True)
    assert manifest["counts"]["dry_run"] == 1
    row = manifest["rows"][0]
    assert row["instruction_status"] == "dry_run"
    assert "MEASURED COLOR BEHAVIOR" in row["prompt_preview"]


# --- orchestration + judge --------------------------------------------------------
def test_orchestration_generated_and_rejected(tmp_path, monkeypatch):
    cfg = tmp_path / "mc.yaml"
    cfg.write_text(_TEACHER_CFG)
    monkeypatch.setenv("TFY_TEST_URL", "https://gateway.example/v1")
    monkeypatch.setenv("TFY_TEST_KEY", "sk-test")
    _patch_client(monkeypatch, json.dumps({
        "gold_tags": ["cooler"],  # wrong direction: behavior is warmer -> deterministic reject
        "concise": "Make it cooler.",
        "natural": "Cool it down.",
    }))
    tc = TeacherClient(cfg)
    manifest = generate_instructions_for_rows([_ROW], tc, run_judge=False)
    assert manifest["counts"]["rejected"] == 1
    assert manifest["rows"][0]["instruction_status"] == "rejected_teacher"
    assert any("tag_not_backed" in i for i in manifest["rows"][0]["validation_issues"])


def test_judge_score_instruction_wired(tmp_path, monkeypatch):
    cfg = tmp_path / "mc.yaml"
    cfg.write_text(_JUDGE_CFG)
    monkeypatch.setenv("TFY_TEST_URL", "https://gateway.example/v1")
    monkeypatch.setenv("TFY_TEST_KEY", "sk-test")
    assert judge_client.is_available(str(cfg)) is True

    _patch_client(monkeypatch, json.dumps({"pass": True, "issues": []}))
    lr = judge_client.score_instruction("Make it warmer.", "Warm it up.", ["warmer"],
                                        {"temperature_delta_b": 3.0}, model_clients_path=str(cfg))
    assert lr.status == "pass"
    assert lr.details["authority"] == "non_authoritative"

    _patch_client(monkeypatch, json.dumps({"pass": False, "issues": ["local_edit"]}))
    lr2 = judge_client.score_instruction("Blur the background and warm it.", "…", ["warmer"],
                                         {"temperature_delta_b": 3.0}, model_clients_path=str(cfg))
    assert lr2.status == "fail"
    assert "local_edit" in (lr2.reason or "")


def test_sdk_error_downgrades_row_not_crash(tmp_path, monkeypatch):
    """A gateway error (429/5xx/timeout) must downgrade one row, not abort the batch."""
    openai = pytest.importorskip("openai")
    cfg = tmp_path / "mc.yaml"
    cfg.write_text(_TEACHER_CFG)
    monkeypatch.setenv("TFY_TEST_URL", "https://gateway.example/v1")
    monkeypatch.setenv("TFY_TEST_KEY", "sk-test")

    class _BoomCompletions:
        def create(self, **kwargs):
            raise openai.OpenAIError("simulated gateway 429")

    class _BoomClient:
        def __init__(self):
            self.chat = type("_C", (), {"completions": _BoomCompletions()})()

    monkeypatch.setattr(openai_compat, "build_client", lambda *a, **k: _BoomClient())
    tc = TeacherClient(cfg)
    # raw SDK error is normalized to TeacherGenerationError, not propagated as openai.OpenAIError
    with pytest.raises(TeacherGenerationError):
        tc.generate(_ROW)
    # and the batch orchestration records it as rejected instead of crashing
    manifest = generate_instructions_for_rows([_ROW], tc, run_judge=False)
    assert manifest["counts"]["error"] == 1
    assert manifest["rows"][0]["instruction_status"] == "rejected_teacher"


def test_clean_tags_separator_tolerant(monkeypatch, tmp_path):
    """Teacher emitting 'bleach-bypass' / 'teal orange' maps to the canonical vocabulary."""
    cfg = tmp_path / "mc.yaml"
    cfg.write_text(_TEACHER_CFG)
    monkeypatch.setenv("TFY_TEST_URL", "https://gateway.example/v1")
    monkeypatch.setenv("TFY_TEST_KEY", "sk-test")
    _patch_client(monkeypatch, json.dumps({
        "gold_tags": ["bleach-bypass", "teal orange"],
        "concise": "Bleach-bypass, teal/orange.", "natural": "Gritty teal-orange film look.",
    }))
    out = TeacherClient(cfg).generate(_ROW)
    assert set(out["gold_tags"]) == {"bleach bypass", "teal-orange"}


def test_authored_rows_skip_teacher(tmp_path):
    """MMArt-PPR10K authored rows are preserved and never hit the teacher (no client patched)."""
    cfg = tmp_path / "mc.yaml"
    cfg.write_text(_TEACHER_CFG)
    tc = TeacherClient(cfg)
    row = {"id": "ppr1", "gold_tags": [], "measured_behavior": {"temperature_delta_b": 2.0},
           "instruction": "Warm the photo and lift the shadows.",
           "instruction_status": "source_authored"}
    # No client patched and no env set: if the teacher were called it would raise -> the row
    # must be short-circuited as authored instead.
    manifest = generate_instructions_for_rows([row], tc, run_judge=False, limit=5)
    assert manifest["counts"]["authored"] == 1
    assert manifest["rows"][0]["instruction_status"] == "source_authored"
    assert manifest["rows"][0]["concise"] == "Warm the photo and lift the shadows."


def test_assemble_active_sets_authored_instruction():
    cand = {"id": "p1", "source_family": "ppr10k_derived", "measured_behavior": {"x": 1.0},
            "authored_instruction": "Warm it up.", "authored_instruction_natural": "Cozy warm vibe."}
    row = assemble_active([cand])[0]
    assert row.instruction == "Warm it up."
    assert row.instruction_natural == "Cozy warm vibe."
    assert row.instruction_status == "source_authored"


def test_judge_gated_without_config():
    lr = judge_client.score_instruction("x", "y", [], {}, model_clients_path=None)
    assert lr.status == "not_evaluated"
    # L8 output judge stays gated / not-run in the decode-disabled spine
    assert judge_client.score({"id": "r"}, None, {}, None).status == "not_evaluated"
