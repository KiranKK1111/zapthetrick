"""Operational modes — unified offline-first + reproducible layer (P5 #27).

Pins: reproducible mode stamps temperature 0 + a fixed seed onto provider options
(and the seed reaches the wire payload); offline-first sorts local platforms
first; both are OFF by default → byte-identical to today.
"""
from __future__ import annotations

from app.llm.operational import (OperationalMode, is_local_platform,
                                 order_by_locality)
from app.llm.providers.base import BaseAdapter


def _payload(options):
    # _payload doesn't touch instance state, so an unbound call with a dummy
    # self exercises the wire-shaping logic without constructing a real adapter.
    return BaseAdapter._payload(object(), [{"role": "user", "content": "hi"}],
                                "m", options, False)


def test_defaults_off_is_noop():
    mode = OperationalMode(offline=False, reproducible=False, seed=7)
    opts = {"temperature": 0.7}
    out = mode.apply_to_options(opts)
    assert out == {"temperature": 0.7}          # untouched
    assert not mode.prefer_local()


def test_reproducible_stamps_temp_and_seed():
    mode = OperationalMode(reproducible=True, seed=42)
    opts = {}
    mode.apply_to_options(opts)
    assert opts["temperature"] == 0
    assert opts["seed"] == 42


def test_reproducible_respects_explicit_values():
    mode = OperationalMode(reproducible=True, seed=42)
    opts = {"temperature": 0.9, "seed": 99}
    mode.apply_to_options(opts)
    assert opts["temperature"] == 0.9 and opts["seed"] == 99   # not overridden


def test_seed_reaches_provider_payload():
    payload = _payload({"seed": 1234})
    assert payload["seed"] == 1234


def test_no_seed_no_payload_key():
    payload = _payload({})
    assert "seed" not in payload                 # today's behaviour untouched


def test_offline_first_orders_local_first():
    assert is_local_platform("ollama") and not is_local_platform("groq")
    ordered = order_by_locality(["groq", "ollama", "gemini", "local"])
    assert ordered[:2] == ["ollama", "local"] or set(ordered[:2]) == {"ollama", "local"}


def test_prefer_local_reflects_offline_flag():
    assert OperationalMode(offline=True).prefer_local()
    assert OperationalMode(offline=True).allow_cloud()     # cloud is last resort, still allowed
