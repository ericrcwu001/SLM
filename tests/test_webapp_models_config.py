import json

import pytest

from webapp.models_config import WebappConfig, _match_embedding_rows


class _Embedding:
    def __init__(self, rows: int):
        self.num_embeddings = rows


class _Model:
    def __init__(self, rows: int):
        self.embedding = _Embedding(rows)
        self.resizes: list[int] = []

    def get_input_embeddings(self):
        return self.embedding

    def resize_token_embeddings(self, rows: int):
        self.resizes.append(rows)
        self.embedding.num_embeddings = rows


class _Tokenizer:
    def __init__(self, rows: int):
        self.rows = rows

    def __len__(self):
        return self.rows


def test_embedding_rows_grow_or_shrink_to_authoritative_adapter_tokenizer():
    for model_rows, tokenizer_rows in ((151665, 151924), (151936, 151924)):
        model = _Model(model_rows)
        _match_embedding_rows(model, _Tokenizer(tokenizer_rows))
        assert model.resizes == [151924]


def test_matching_embedding_rows_are_left_untouched():
    model = _Model(151924)
    _match_embedding_rows(model, _Tokenizer(151924))
    assert model.resizes == []


def _write_config(tmp_path, server_overrides: dict):
    server = {
        "runs_dir": "webapp/_runs",
        "static_dir": "webapp/static",
        "references_dir": "webapp/assets/references",
        **server_overrides,
    }
    path = tmp_path / "webapp.json"
    path.write_text(json.dumps({"device": "cpu", "server": server}), encoding="utf-8")
    return path


def test_gallery_config_defaults_and_overrides(tmp_path):
    default_cfg = WebappConfig.load(_write_config(tmp_path, {}))
    assert default_cfg.server.gallery_dir == "webapp/_gallery"
    assert default_cfg.server.gallery_max_entries == 40
    assert default_cfg.server.gallery_enabled is True

    override = WebappConfig.load(
        _write_config(tmp_path, {"gallery_dir": "/data/gallery", "gallery_max_entries": 60, "gallery_enabled": False})
    )
    assert override.server.gallery_dir == "/data/gallery"
    assert override.server.gallery_max_entries == 60
    assert override.server.gallery_enabled is False


def test_gallery_max_entries_must_be_positive(tmp_path):
    with pytest.raises(ValueError, match="gallery_max_entries"):
        WebappConfig.load(_write_config(tmp_path, {"gallery_max_entries": 0}))
