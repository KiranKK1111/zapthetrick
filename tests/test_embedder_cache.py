"""Single-string embed cache (gap G7)."""
from __future__ import annotations

from app.rag import embedder as emb


def test_single_string_embed_is_cached(monkeypatch):
    emb._ONE_CACHE.clear()
    emb._CACHE_ENABLED = True          # conftest disables it; these tests need it
    calls = {"n": 0}

    class _FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            calls["n"] += 1
            import numpy as np
            return np.array([[0.6, 0.8]] * len(texts))

    monkeypatch.setattr(emb, "_model", lambda: _FakeModel())

    v1 = emb.embed(["same query"])
    v2 = emb.embed(["same query"])          # cache hit → no second encode
    assert v1 == v2
    assert calls["n"] == 1


def test_cache_returns_copies_not_shared(monkeypatch):
    emb._ONE_CACHE.clear()
    emb._CACHE_ENABLED = True          # conftest disables it; these tests need it

    class _FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            import numpy as np
            return np.array([[0.6, 0.8]])

    monkeypatch.setattr(emb, "_model", lambda: _FakeModel())
    a = emb.embed(["q"])[0]
    a[0] = 999.0                            # mutate the returned list
    b = emb.embed(["q"])[0]                 # cache must be uncorrupted
    assert b[0] != 999.0


def test_multi_string_not_cached(monkeypatch):
    emb._ONE_CACHE.clear()
    emb._CACHE_ENABLED = True          # conftest disables it; these tests need it
    calls = {"n": 0}

    class _FakeModel:
        def encode(self, texts, normalize_embeddings=True):
            calls["n"] += 1
            import numpy as np
            return np.array([[0.6, 0.8]] * len(texts))

    monkeypatch.setattr(emb, "_model", lambda: _FakeModel())
    emb.embed(["a", "b"])
    emb.embed(["a", "b"])
    assert calls["n"] == 2                   # batch calls bypass the 1-string cache
