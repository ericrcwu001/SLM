"""Tests for the prompt_to_lut CLI (refusal path, blocked decode, self-check)."""

import json
import os

from PIL import Image

from cli import prompt_to_lut
from eval.output_parsers import format_tokens


def _make_image(path):
    Image.new("RGB", (16, 16), (120, 90, 70)).save(path, format="PNG")


def _metrics(out_dir):
    with open(os.path.join(out_dir, "metrics.json")) as fh:
        return json.load(fh)


def test_self_check_passes():
    assert prompt_to_lut.run_self_check() == 0


def test_unsupported_writes_refusal_artifacts_only(tmp_path):
    img = tmp_path / "in.png"
    _make_image(img)
    out = tmp_path / "run_u"
    rc = prompt_to_lut.run(str(img), "make only the sky bluer", str(out),
                           mock_output=None, model="mock", mode="runtime_constrained")
    assert rc == 0
    assert os.path.exists(out / "output_tokens.txt")
    assert open(out / "output_tokens.txt").read().strip() == "<unsupported>"
    m = _metrics(str(out))
    assert m["output"]["kind"] == "unsupported"
    assert m["status"]["blocked"] is False
    # supported-only artifacts must NOT be written
    assert not os.path.exists(out / "output.cube")
    assert not os.path.exists(out / "graded.png")
    assert not os.path.exists(out / "preview_side_by_side.png")
    # refusal artifact set present
    for name in ("input.png", "output_tokens.txt", "metrics.json", "version_manifest.json"):
        assert os.path.exists(out / name)


def test_supported_tokens_block_at_decode(tmp_path):
    img = tmp_path / "in.png"
    _make_image(img)
    out = tmp_path / "run_s"
    mock = format_tokens(list(range(64)))
    rc = prompt_to_lut.run(str(img), "make it warmer", str(out),
                           mock_output=mock, model="mock", mode="runtime_constrained")
    assert rc == 0
    m = _metrics(str(out))
    assert m["output"]["kind"] == "lut_tokens"
    assert m["output"]["syntax_pass"] is True
    assert m["output"]["token_count"] == 64
    assert m["status"]["blocked"] is True
    assert m["status"]["block_reason"] == "decoder_disabled"
    # no silent identity LUT / graded output
    assert not os.path.exists(out / "output.cube")
    assert not os.path.exists(out / "graded.png")


def test_gated_model_raises(tmp_path):
    from eval.baseline_adapters import RequiresModel

    img = tmp_path / "in.png"
    _make_image(img)
    try:
        prompt_to_lut.run(str(img), "x", str(tmp_path / "g"), None, "qwen", "runtime_constrained")
        assert False, "expected RequiresModel"
    except RequiresModel:
        pass


def test_version_manifest_written(tmp_path):
    img = tmp_path / "in.png"
    _make_image(img)
    out = tmp_path / "run_m"
    prompt_to_lut.run(str(img), "make it warmer", str(out), None, "mock", "runtime_constrained")
    with open(out / "version_manifest.json") as fh:
        man = json.load(fh)
    assert man["canonical_domain_id"] == "slm_lut_v1_srgb_display_encoded_17_trilinear"
    assert len(man["added_special_token_ids"]) == 259
    assert man["decoder_enabled"] is False
