from webapp.models_config import _match_embedding_rows


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
