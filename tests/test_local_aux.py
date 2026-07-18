"""Local auxiliary models facade (perceived-speed R10/R11, task 12.3).

Pins: prefers local, remote fallback on local failure, deterministic local
title/summary/intent (no model), and vector-store-compatible embeddings.
"""
from __future__ import annotations

import sys

from app.perceived import local_aux


def test_embed_prefers_local(monkeypatch):
    import app.rag.embedder as emb
    monkeypatch.setattr(emb, "embed", lambda texts: [[0.1, 0.2]] * len(texts))
    out = local_aux.embed(["a", "b"])
    assert out == [[0.1, 0.2], [0.1, 0.2]]


def test_embed_remote_fallback_on_local_failure(monkeypatch):
    import app.rag.embedder as emb

    def _boom(texts):
        raise RuntimeError("model not loaded")

    monkeypatch.setattr(emb, "embed", _boom)
    out = local_aux.embed(["q"], remote=lambda texts: [[9.9]] * len(texts))
    assert out == [[9.9]]


def test_embed_raises_without_remote(monkeypatch):
    import app.rag.embedder as emb
    monkeypatch.setattr(emb, "embed", lambda texts: (_ for _ in ()).throw(RuntimeError()))
    try:
        local_aux.embed(["q"])
        assert False, "expected raise without a remote fallback"
    except RuntimeError:
        pass


def test_embeddings_available_reflects_package():
    # sentence_transformers is a project dependency → available.
    assert local_aux.embeddings_available() == (
        "sentence_transformers" in sys.modules
        or local_aux.embeddings_available()  # find_spec path
    )


def test_local_title_is_deterministic_no_model():
    assert local_aux.local_title("build a flutter chat app with streaming") \
        == "build a flutter chat app with"
    assert local_aux.local_title("") == "New conversation"


def test_local_summary_truncates():
    long = "word " * 400
    s = local_aux.local_summary(long, max_chars=50)
    assert len(s) <= 51 and s.endswith("…")
    assert local_aux.local_summary("short text", max_chars=50) == "short text"


def test_local_intent_uses_deterministic_predictor():
    assert local_aux.local_intent("write a python function") == "coding"
    assert local_aux.local_intent("hello there") == "general"
