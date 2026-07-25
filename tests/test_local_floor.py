"""Local generation floor — T4 + the §8.7 grammar floor (vNext §2.1/§8.7).

The local runtime is inert until llm.local.enabled + a seeded model. These tests
stub those so no llama.cpp server is needed.
"""
from __future__ import annotations

import asyncio

from app.llm import router as R
from app.llm import structured as S
from app.llm.catalog import ProviderSpec
from app.llm.providers.openai_compat import OpenAICompatAdapter


# ── T4 rung in select_route ──────────────────────────────────────────────
def test_t4_local_floor_answers_when_cloud_exhausts(monkeypatch):
    async def _no_cloud(**kw):
        raise R.NoRouteAvailable("all cloud down", transient=False)

    async def _floor():
        return R.RouteResult("local", "qwen3-4b-instruct", 42, "Local", "", 0,
                             rung="T4", degraded=["local_floor"])

    monkeypatch.setattr(R, "route_request", _no_cloud)
    monkeypatch.setattr(R, "_ladder_enabled", lambda: True)
    monkeypatch.setattr(R, "_local_floor_route", _floor)

    d = asyncio.run(R.select_route())
    assert d.rung == "T4"
    assert d.local is True
    assert d.route.platform == "local"
    assert d.route.model_db_id == 42


def test_exhausted_when_no_local_floor(monkeypatch):
    async def _no_cloud(**kw):
        raise R.NoRouteAvailable("down", transient=True)

    async def _no_floor():
        return None

    monkeypatch.setattr(R, "route_request", _no_cloud)
    monkeypatch.setattr(R, "_ladder_enabled", lambda: True)
    monkeypatch.setattr(R, "_local_floor_route", _no_floor)

    d = asyncio.run(R.select_route())
    assert d.route is None
    assert d.rung == "exhausted"


def test_local_floor_route_is_none_when_disabled(monkeypatch):
    # Default config has no llm.local section → local disabled → None (no DB hit).
    monkeypatch.setattr("app.llm.catalog.local_enabled", lambda: False)
    assert asyncio.run(R._local_floor_route()) is None


# ── grammar passthrough is LOCAL-ONLY ────────────────────────────────────
def test_grammar_only_reaches_the_local_provider():
    local = OpenAICompatAdapter(
        ProviderSpec("local", "Local", "http://127.0.0.1:8081/v1"))
    cloud = OpenAICompatAdapter(
        ProviderSpec("groq", "Groq", "https://api.groq.com/openai/v1"))
    opts = {"grammar": "root ::= object"}
    assert local._payload([], "m", opts, stream=False)["grammar"] == "root ::= object"
    # A cloud provider must NEVER receive the llama.cpp-only grammar field.
    assert "grammar" not in cloud._payload([], "m", opts, stream=False)


# ── §8.7 grammar floor in structured() ───────────────────────────────────
_SCHEMA = {"type": "object", "properties": {"lang": {"type": "string"}},
           "required": ["lang"]}


def test_grammar_floor_produces_valid_json_when_cloud_fails(monkeypatch):
    calls = {"n": 0}

    async def _complete(messages, options, *, session_key=None,
                        preferred_model_db_id=None):
        calls["n"] += 1
        # Cloud attempts (no grammar) return junk; the local floor call carries a
        # grammar and returns valid JSON.
        if options.get("grammar"):
            class _R:
                display_name = "Local"
            return '{"lang": "dart"}', _R()

        class _R2:
            model_db_id = 1
            display_name = "cloud"
        return "not json at all", _R2()

    monkeypatch.setattr("app.llm.engine.route_and_complete", _complete)
    monkeypatch.setattr("app.llm.router.local_model_db_id",
                        lambda: _async(99))

    res = asyncio.run(S.structured(_SCHEMA, [{"role": "user", "content": "x"}],
                                   retries=1))
    assert res.valid
    assert res.obj == {"lang": "dart"}
    assert "schema_local_floor" in res.degraded


def _async(value):
    async def _coro():
        return value
    return _coro()
