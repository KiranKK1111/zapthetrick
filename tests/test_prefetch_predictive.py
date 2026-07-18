"""Predictive context prefetch (#3) + intelligent reuse across a changed prompt
(#5).

Pins that warm() precomputes a real query embedding + optional context (not just
a warm socket), and that reuse_artifacts salvages that work when the user kept
typing (a similar-but-changed submit) rather than throwing it away.
"""
from __future__ import annotations

import asyncio

import app.rag.embedder as embedder
from app.perceived.prefetch import PrefetchManager, ReusedContext


class _AlwaysBudget:
    def allow(self, *, kind="work"):
        return True

    def account(self, n=1):
        pass


def _mgr():
    return PrefetchManager(budget=_AlwaysBudget())


def test_warm_precomputes_embedding_and_context(monkeypatch):
    calls = {"embed": 0, "retrieve": 0}

    def fake_embed_one(text):
        calls["embed"] += 1
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(embedder, "embed_one", fake_embed_one)

    def retrieve(text):
        calls["retrieve"] += 1
        return [{"snippet": "ctx for " + text}]

    m = _mgr()
    q = "how does kubernetes scheduling actually work"
    token = asyncio.run(m.warm(q, retrieve=retrieve))
    assert token
    assert calls["embed"] == 1 and calls["retrieve"] == 1

    reused = m.reuse_artifacts(token, q)         # exact submit
    assert isinstance(reused, ReusedContext)
    assert reused.exact and reused.embedding == [0.1, 0.2, 0.3]
    assert reused.retrieval and reused.any


def test_short_fragment_skips_embedding(monkeypatch):
    called = {"n": 0}
    monkeypatch.setattr(embedder, "embed_one",
                        lambda t: called.__setitem__("n", called["n"] + 1) or [0.0])
    m = _mgr()
    token = asyncio.run(m.warm("hi"))            # < _MIN_PREFETCH_CHARS
    assert token
    assert called["n"] == 0                       # no expensive embed on a fragment


def test_reuse_across_changed_prompt_keeps_retrieval(monkeypatch):
    monkeypatch.setattr(embedder, "embed_one", lambda t: [0.5, 0.5])
    m = _mgr()
    warmed = "explain the raft consensus algorithm in detail"
    token = asyncio.run(m.warm(warmed, retrieve=lambda t: ["raft-ctx"]))
    # User kept typing — submit differs slightly but is very similar.
    submit = "explain the raft consensus algorithm in detail please"
    reused = m.reuse_artifacts(token, submit)
    assert not reused.exact
    assert reused.similarity >= 0.6
    assert reused.retrieval == ["raft-ctx"]       # retrieval salvaged as warm start
    assert reused.embedding is None               # embedding is text-exact → dropped


def test_reuse_unrelated_prompt_salvages_nothing(monkeypatch):
    monkeypatch.setattr(embedder, "embed_one", lambda t: [1.0])
    m = _mgr()
    token = asyncio.run(m.warm("describe the CAP theorem tradeoffs",
                               retrieve=lambda t: ["cap-ctx"]))
    reused = m.reuse_artifacts(token, "write a haiku about the ocean waves")
    assert not reused.exact and not reused.any
    assert reused.retrieval is None


def test_reuse_artifacts_failopen_on_missing_token():
    m = _mgr()
    reused = m.reuse_artifacts(None, "anything")
    assert isinstance(reused, ReusedContext) and not reused.any


def test_embedding_failure_is_failopen(monkeypatch):
    def boom(text):
        raise RuntimeError("embedder down")
    monkeypatch.setattr(embedder, "embed_one", boom)
    m = _mgr()
    token = asyncio.run(m.warm("a reasonably long query about databases here"))
    assert token                                  # warm still succeeds
    reused = m.reuse_artifacts(token, "a reasonably long query about databases here")
    assert reused.embedding is None               # no embedding, but no crash
