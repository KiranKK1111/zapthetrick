"""Tests for auto-compaction → structured L4 digest (vNext §8.4, Stage 7 D)."""
from __future__ import annotations

import asyncio

import app.memory.compaction as C


class _Res:
    def __init__(self, obj):
        self.obj = obj


def _run(coro):
    return asyncio.run(coro)


# ---- window trigger -------------------------------------------------------
def test_fires_at_seventy_percent():
    d = C.should_compact(70, 100, threshold=0.7)
    assert d.compact is True
    assert d.ratio == 0.7
    assert d.headroom_tokens == 30


def test_below_threshold_does_not_fire():
    d = C.should_compact(50, 100, threshold=0.7)
    assert d.compact is False
    assert d.headroom_tokens == 50


def test_zero_window_is_safe():
    d = C.should_compact(10, 0)
    assert d.compact is False
    assert d.ratio == 0.0


def test_trigger_never_raises_on_garbage():
    d = C.should_compact(None, None)   # type: ignore[arg-type]
    assert d.compact is False


# ---- structured L4 digest -------------------------------------------------
def _stub(obj):
    async def fn(schema, msgs, **kw):
        return _Res(obj)
    return fn


def test_digest_parses_all_sections(monkeypatch):
    monkeypatch.setattr(C, "enabled", lambda: True)
    obj = {"decisions": ["use postgres"], "facts": ["port 8080"],
           "entities": ["Kafka"], "open_threads": ["pick a reranker"],
           "goals": ["ship vNext"], "artifacts": ["router.py"]}
    d = _run(C.build_digest([{"role": "user", "content": "we chose postgres"}],
                            structured_fn=_stub(obj)))
    assert d.decisions == ["use postgres"]
    assert d.entities == ["Kafka"]
    assert not d.is_empty()


def test_digest_disabled_is_empty(monkeypatch):
    monkeypatch.setattr(C, "enabled", lambda: False)
    called = {"n": 0}

    async def fn(schema, msgs, **kw):
        called["n"] += 1
        return _Res({"facts": ["x"]})
    d = _run(C.build_digest([{"role": "user", "content": "hi"}], structured_fn=fn))
    assert d.is_empty()
    assert called["n"] == 0            # disabled → the LLM is never called


def test_digest_empty_messages(monkeypatch):
    monkeypatch.setattr(C, "enabled", lambda: True)
    d = _run(C.build_digest([], structured_fn=_stub({"facts": ["x"]})))
    assert d.is_empty()


def test_digest_fail_open_on_bad_obj(monkeypatch):
    monkeypatch.setattr(C, "enabled", lambda: True)
    d = _run(C.build_digest([{"role": "user", "content": "hi there friend"}],
                            structured_fn=_stub("not a dict")))
    assert d.is_empty()


def test_digest_fail_open_on_raise(monkeypatch):
    monkeypatch.setattr(C, "enabled", lambda: True)

    async def boom(schema, msgs, **kw):
        raise RuntimeError("llm down")
    d = _run(C.build_digest([{"role": "user", "content": "hello world here"}],
                            structured_fn=boom))
    assert d.is_empty()


def test_digest_coerces_non_list_fields(monkeypatch):
    monkeypatch.setattr(C, "enabled", lambda: True)
    d = _run(C.build_digest([{"role": "user", "content": "content here now"}],
                            structured_fn=_stub({"facts": "a single string",
                                                 "goals": ["g1", "", "g2"]})))
    assert d.facts == ["a single string"]
    assert d.goals == ["g1", "g2"]     # blank dropped


def test_digest_to_text_renders_labels(monkeypatch):
    monkeypatch.setattr(C, "enabled", lambda: True)
    d = C.StructuredDigest(goals=["ship it"], decisions=["use pg"])
    txt = C.digest_to_text(d)
    assert "Goals: ship it" in txt
    assert "Decisions: use pg" in txt


def test_digest_to_text_empty_is_blank():
    assert C.digest_to_text(C.StructuredDigest()) == ""
