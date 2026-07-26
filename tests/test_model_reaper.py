"""Proactive dead-model reaper: confirm-then-prune, last-known-good-safe.

Only an ENABLED routing model the provider has DROPPED from a successful /models
list AND which a probe confirms is permanently dead may be pruned. A still-listed
model, an absent-but-alive model, an absent-but-rate-limited model, and any
platform whose list fetch failed (None/empty) must all survive.
"""
import asyncio

from app.llm import model_reaper


def _wire(monkeypatch):
    async def _fake_enabled():
        return [
            (1, "nim", "alive-listed"),         # still in the live list → skip
            (2, "nim", "gone-eol"),             # dropped + probe dead → PRUNE
            (3, "nim", "dropped-but-alive"),    # dropped but probe answers → keep
            (4, "nim", "dropped-ratelimited"),  # dropped, probe 429 → keep
            (5, "ghost", "no-list"),            # list fetch fails → keep
        ]

    async def _fake_fetch_ids(platform, api_key=None):
        if platform == "ghost":
            return None  # unknown → prune nothing
        return {"alive-listed"}

    async def _fake_key(platform):
        return "sk-test"

    async def _fake_probe(platform, model_id, api_key):
        return model_id == "gone-eol"

    disabled: list[int] = []

    async def _fake_disable(model_db_id):
        disabled.append(model_db_id)

    monkeypatch.setattr(model_reaper, "_enabled_routing_models", _fake_enabled)
    monkeypatch.setattr(model_reaper.discovery, "fetch_model_ids", _fake_fetch_ids)
    monkeypatch.setattr(model_reaper, "_decrypt_first_key", _fake_key)
    monkeypatch.setattr(model_reaper, "_probe_is_dead", _fake_probe)
    monkeypatch.setattr("app.llm.engine._disable_model", _fake_disable)
    return disabled


def test_reap_prunes_only_confirmed_dead(monkeypatch):
    disabled = _wire(monkeypatch)
    pruned = asyncio.run(model_reaper.reap_dead_models())
    assert pruned == 1
    assert disabled == [2]  # only 'gone-eol' (id=2); 1/3/4/5 survive


def test_probe_classifies_dead(monkeypatch):
    """_probe_is_dead is True only for a permanent_dead ProviderError."""
    from app.llm.providers import ProviderError

    class _Adapter:
        def __init__(self, exc):
            self._exc = exc

        async def complete(self, *a, **k):
            if self._exc:
                raise self._exc
            return "ok"

    # 410 EOL → dead
    monkeypatch.setattr(model_reaper, "get_adapter",
                        lambda p: _Adapter(ProviderError("gone", status=410)))
    assert asyncio.run(model_reaper._probe_is_dead("nim", "m", "k")) is True

    # 429 rate limit → NOT dead
    monkeypatch.setattr(model_reaper, "get_adapter",
                        lambda p: _Adapter(ProviderError("busy", status=429)))
    assert asyncio.run(model_reaper._probe_is_dead("nim", "m", "k")) is False

    # answers → NOT dead
    monkeypatch.setattr(model_reaper, "get_adapter", lambda p: _Adapter(None))
    assert asyncio.run(model_reaper._probe_is_dead("nim", "m", "k")) is False
