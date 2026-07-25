"""Tests for predictive pre-answering (vNext §9.5, Stage 10 Component D)."""
from __future__ import annotations

import app.live.preanswer as P


def _on(monkeypatch):
    monkeypatch.setattr(P, "enabled", lambda: True)


# ---- trigger --------------------------------------------------------------
def test_fires_on_silence_and_idle_gpu(monkeypatch):
    _on(monkeypatch)
    d = P.should_pregenerate(silence_ms=5000, gpu_idle=True, top_n=2)
    assert d.pregenerate and d.slots == 2


def test_no_fire_before_silence_floor(monkeypatch):
    _on(monkeypatch)
    assert not P.should_pregenerate(silence_ms=1000, gpu_idle=True).pregenerate


def test_no_fire_when_gpu_busy(monkeypatch):
    _on(monkeypatch)
    assert not P.should_pregenerate(silence_ms=9000, gpu_idle=False).pregenerate


def test_no_fire_when_slots_full(monkeypatch):
    _on(monkeypatch)
    d = P.should_pregenerate(silence_ms=9000, gpu_idle=True, already_pregen=2, top_n=2)
    assert not d.pregenerate and d.slots == 0


def test_partial_slots(monkeypatch):
    _on(monkeypatch)
    d = P.should_pregenerate(silence_ms=9000, gpu_idle=True, already_pregen=1, top_n=2)
    assert d.pregenerate and d.slots == 1


def test_disabled_never_fires(monkeypatch):
    monkeypatch.setattr(P, "enabled", lambda: False)
    assert not P.should_pregenerate(silence_ms=99999, gpu_idle=True).pregenerate


def test_trigger_never_raises(monkeypatch):
    _on(monkeypatch)
    d = P.should_pregenerate(silence_ms=None, gpu_idle=True)  # type: ignore[arg-type]
    assert isinstance(d, P.PregenDecision)


# ---- cache + flush --------------------------------------------------------
def _cache():
    c = P.PreAnswerCache()
    c.store("what is kafka", "Kafka is a distributed log.",
            embedding=[1.0, 0.0, 0.0], deps={"topic:kafka"})
    c.store("explain partitions", "Partitions enable parallelism.",
            embedding=[0.0, 1.0, 0.0], deps={"topic:kafka"})
    return c


def _emb(q):
    return [0.99, 0.01, 0.0] if "kafka" in q.lower() else [0.0, 0.0, 1.0]


def test_flush_on_high_match(monkeypatch):
    _on(monkeypatch)
    hit = _cache().match("so, what is kafka exactly?", embed_fn=_emb)
    assert hit is not None
    assert hit.question == "what is kafka"
    assert hit.similarity >= 0.92
    assert "distributed log" in hit.answer


def test_no_flush_below_threshold(monkeypatch):
    _on(monkeypatch)
    assert _cache().match("an unrelated question", embed_fn=_emb) is None


def test_custom_threshold(monkeypatch):
    _on(monkeypatch)
    c = P.PreAnswerCache()
    c.store("q", "a", embedding=[1.0, 0.0])
    # cosine of [1,0] vs [0.8,0.6] = 0.8 → passes 0.75, fails 0.92.
    assert c.match("x", embed_fn=lambda q: [0.8, 0.6], threshold=0.75) is not None
    assert c.match("x", embed_fn=lambda q: [0.8, 0.6], threshold=0.92) is None


def test_store_ignores_empty():
    c = P.PreAnswerCache()
    c.store("", "answer", embedding=[1.0])
    c.store("q", "", embedding=[1.0])
    assert c.items == []


def test_match_disabled_returns_none(monkeypatch):
    monkeypatch.setattr(P, "enabled", lambda: False)
    assert _cache().match("what is kafka", embed_fn=_emb) is None


def test_match_no_embedder_returns_none(monkeypatch):
    _on(monkeypatch)
    assert _cache().match("what is kafka") is None


# ---- staleness ------------------------------------------------------------
def test_invalidate_by_dependency(monkeypatch):
    _on(monkeypatch)
    c = _cache()
    assert c.fresh_count() == 2
    n = c.invalidate("topic:kafka")
    assert n == 2 and c.fresh_count() == 0


def test_stale_preanswer_never_flushes(monkeypatch):
    _on(monkeypatch)
    c = _cache()
    c.invalidate("topic:kafka")
    assert c.match("what is kafka", embed_fn=_emb) is None   # all stale


def test_invalidate_only_matching_dep(monkeypatch):
    _on(monkeypatch)
    c = P.PreAnswerCache()
    c.store("q1", "a1", embedding=[1.0], deps={"dep:a"})
    c.store("q2", "a2", embedding=[1.0], deps={"dep:b"})
    assert c.invalidate("dep:a") == 1 and c.fresh_count() == 1


def test_clear():
    c = _cache()
    c.clear()
    assert c.items == []


def test_end_to_end_predict_pregen_flush(monkeypatch):
    # §9.5 acceptance: a predicted question, pre-answered during silence, flushes
    # instantly when the real question arrives → perceived TTFT ≈ 0.
    _on(monkeypatch)
    assert P.should_pregenerate(silence_ms=5000, gpu_idle=True).pregenerate
    c = P.PreAnswerCache()
    c.store("what is a hash map", "A hash map is O(1) average lookup.",
            embedding=[1.0, 0.0], deps={"topic:hashmap"})
    hit = c.match("can you explain what a hash map is",
                  embed_fn=lambda q: [1.0, 0.0])
    assert hit and hit.answer.startswith("A hash map")
