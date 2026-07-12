import json
from pathlib import Path

import numpy as np
from PIL import Image

from eval.cube_io import parse_cube
from webapp.models_config import WebappConfig
from webapp.pipeline import PromptToLutPipeline


def _make_references(root: Path):
    root.mkdir(parents=True)
    names = ["City", "Landscape", "Portrait", "Close-up", "Food", "Interior"]
    rows = []
    for index, name in enumerate(names):
        filename = f"{index}_{name.lower().replace('-', '_')}.jpg"
        Image.new("RGB", (32, 24), (40 + index * 25, 80 + index * 12, 120 + index * 8)).save(root / filename)
        rows.append({"name": name, "filename": filename})
    (root / "references.json").write_text(json.dumps({"references": rows}), encoding="utf-8")


def _pipeline(tmp_path: Path):
    refs = tmp_path / "references"
    _make_references(refs)
    cfg = WebappConfig()
    cfg.generator.stub = True
    cfg.server.references_dir = str(refs)
    cfg.server.runs_dir = str(tmp_path / "runs")
    return PromptToLutPipeline(cfg)


def test_stub_grade_writes_cube_and_seven_visibly_graded_previews(tmp_path):
    pipeline = _pipeline(tmp_path)
    image = Image.fromarray(np.tile(np.arange(64, dtype=np.uint8), (48, 1))[:, :, None].repeat(3, axis=2))
    payload = pipeline.run("make it warmer with strong teal-orange contrast", image, tmp_path / "runs" / "abc")
    assert payload["route"] == "grade"
    assert payload["attribute_spec_text"].startswith("route=grade")
    assert len(payload["previews"]) == 7
    assert [p["name"] for p in payload["previews"]] == ["user_image", "City", "Landscape", "Portrait", "Close-up", "Food", "Interior"]
    lut, _ = parse_cube((tmp_path / "runs" / "abc" / "output.cube").read_bytes())
    assert lut.shape == (17, 17, 17, 3)
    assert not np.allclose(lut[..., 0], np.linspace(0, 1, 17)[:, None, None])
    original = np.asarray(Image.open(tmp_path / "runs" / "abc" / "user_image_original.png"))
    graded = np.asarray(Image.open(tmp_path / "runs" / "abc" / "user_image_graded.png"))
    assert np.mean(np.abs(original.astype(float) - graded.astype(float))) > 2.0


def test_stub_clarify_and_refuse_short_circuit_without_artifacts(tmp_path):
    pipeline = _pipeline(tmp_path)
    image = Image.new("RGB", (24, 24), "gray")
    clarify = pipeline.run("make it pop", image, tmp_path / "clarify")
    assert clarify["route"] == "clarify" and clarify["lut"] is None and clarify["previews"] == []
    assert clarify["clarify_message"] and clarify["prompt_feedback"]["suggested_terms"]
    refuse = pipeline.run("remove the person", image, tmp_path / "refuse")
    assert refuse["route"] == "refuse" and refuse["refuse_reason"] == "out_of_scope"
    assert refuse["lut"] is None and refuse["previews"] == []


def test_stub_health_requires_exact_reference_set(tmp_path):
    pipeline = _pipeline(tmp_path)
    health = pipeline.self_check()
    assert health["ok"] and health["stub"] and health["references"] == 6
