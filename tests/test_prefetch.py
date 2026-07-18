"""Intent prediction + prefetch (perceived-speed R1, task 3.4 backend).

Pins: the predictor is LOCAL/deterministic (no model call — Property 2), warming
is reuse-or-discard (Property 3), and nothing runs when speculation is disabled
(R19.4).
"""
from __future__ import annotations

import asyncio

from app.core.config_loader import cfg
from app.perceived.budget import SpeculationBudget
from app.perceived.prefetch import IntentPredictor, PrefetchManager


def _enable(monkeypatch, on=True):
    monkeypatch.setattr(cfg.perceived, "speculation_enabled", on, raising=False)
    monkeypatch.setattr(cfg.perceived, "speculation_period_budget", 0, raising=False)
    # Don't touch the network when warming in tests.
    monkeypatch.setattr("app.core.http_pool.get_http_client", lambda: None, raising=False)


def _mgr(monkeypatch):
    m = PrefetchManager(budget=SpeculationBudget())

    async def _noop(_pred):
        return None

    monkeypatch.setattr(m, "_warm_connection", _noop)
    return m


def test_predict_is_local_and_deterministic():
    p = IntentPredictor()
    assert p.predict("write a python function to reverse a string").topic == "coding"
    assert p.predict("design a java microservice").topic == "coding"
    short = p.predict("hi")
    assert short.topic == "general" and short.complexity == "trivial"
    # Pure + deterministic: same input → same output.
    assert p.predict("explain REST").topic == p.predict("explain REST").topic
    assert p.predict("") == IntentPredictor().predict("")


def test_warm_disabled_is_noop(monkeypatch):
    _enable(monkeypatch, on=False)
    m = _mgr(monkeypatch)
    tok = asyncio.run(m.warm("how do I sort a list in python"))
    assert tok is None            # no token, no warmed work (R19.4 / Property 2)
    assert m.pending == 0


def test_warm_then_reuse_consumes_once(monkeypatch):
    _enable(monkeypatch, on=True)
    m = _mgr(monkeypatch)
    tok = asyncio.run(m.warm("explain hashmap internals"))
    assert tok is not None
    assert m.pending == 1
    assert m.reuse(tok, "explain hashmap internals") is True   # reused (R1.3)
    assert m.reuse(tok) is False                               # only once
    assert m.pending == 0


def test_discard_drops_on_mismatch(monkeypatch):
    _enable(monkeypatch, on=True)
    m = _mgr(monkeypatch)
    tok = asyncio.run(m.warm("partial query"))
    m.discard(tok)                                             # mismatch (R1.4)
    assert m.reuse(tok) is False
    assert m.pending == 0


def test_reuse_none_is_safe():
    m = PrefetchManager(budget=SpeculationBudget())
    assert m.reuse(None) is False
    m.discard(None)   # no-op
