"""Stage-6 §4.6 / §2.7 F — Live session-plan wiring (Component B)."""
from __future__ import annotations

import asyncio
import types

import pytest

from app.live import session_plan as SP
from app.llm import live_plan as LP
from app.llm import quota_plan as Q
from app.llm.live_plan import Candidate


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _fresh():
    LP.reset_for_tests()
    Q.reset_for_tests()
    yield
    LP.reset_for_tests()
    Q.reset_for_tests()


@pytest.fixture
def _on(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.routing, "live_plan", True, raising=False)


def _cands(_profile=None):
    return [
        Candidate("llama|70|3.3|-", "groq", model_db_id=1, key_id=10),
        Candidate("llama|70|3.3|-", "cerebras", model_db_id=2, key_id=20),
        Candidate("qwen|72|2.5|-", "nvidia", model_db_id=3, key_id=30),
    ]


class TestOrchestration:
    def test_disabled_returns_none(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "live_plan", False, raising=False)
        assert _run(SP.plan_live_session("s1", candidate_fn=_cands)) is None

    def test_pins_primary_and_standby(self, _on):
        plan = _run(SP.plan_live_session(
            "s1", candidate_fn=_cands, warm=False))
        assert plan is not None
        assert plan.primary.provider == "groq"
        assert plan.standby.provider == "cerebras"     # same model, other provider

    def test_async_candidate_fn(self, _on):
        async def afn(_profile):
            return _cands()
        plan = _run(SP.plan_live_session("s1", candidate_fn=afn, warm=False))
        assert plan is not None and plan.primary.provider == "groq"

    def test_no_candidates_returns_none(self, _on):
        plan = _run(SP.plan_live_session(
            "s1", candidate_fn=lambda p: [], warm=False))
        assert plan is None

    def test_candidate_fn_error_is_fail_open(self, _on):
        def boom(_profile):
            raise RuntimeError("candidate source down")
        assert _run(SP.plan_live_session(
            "s1", candidate_fn=boom, warm=False)) is None

    def test_reservation_is_held(self, _on):
        before = Q.quota_planner().headroom("groq", 10)
        _run(SP.plan_live_session(
            "s1", candidate_fn=_cands, warm=False, expected_requests=50))
        assert Q.quota_planner().headroom("groq", 10) == before - 50

    def test_release_refunds(self, _on):
        before = Q.quota_planner().headroom("groq", 10)
        _run(SP.plan_live_session(
            "s1", candidate_fn=_cands, warm=False, expected_requests=50))
        SP.release_live_session("s1")
        assert Q.quota_planner().headroom("groq", 10) == before


class TestWarm:
    def test_warm_schedules_pins_without_blocking(self, _on, monkeypatch):
        warmed: list[str] = []

        async def fake_warm(platform):
            warmed.append(platform)
        monkeypatch.setattr(SP, "_warm_provider", fake_warm)

        async def go():
            plan = await SP.plan_live_session("s1", candidate_fn=_cands, warm=True)
            await asyncio.sleep(0.01)      # let the fire-and-forget tasks run
            return plan
        plan = _run(go())
        assert plan is not None
        assert set(warmed) == {"groq", "cerebras"}     # both pinned providers


class TestDefaultRouterSource:
    def test_router_candidates_from_primary_plus_same_model(self, monkeypatch):
        # Stub the router's primary pick + a controlled seed catalog with the
        # SAME model on two providers.
        from app.llm import catalog
        route = types.SimpleNamespace(
            platform="groq", model_id="llama-3.3-70b-versatile",
            model_db_id=1, key_id=10)
        dec = types.SimpleNamespace(route=route)

        async def fake_select_route(**kw):
            return dec
        import app.llm.router as R
        monkeypatch.setattr(R, "select_route", fake_select_route)
        monkeypatch.setattr(catalog, "MODEL_SEED", [
            ("groq", "llama-3.3-70b-versatile", "L", 5, 5, "L", 1, 1, 1, None, "", 1),
            ("cerebras", "Llama-3.3-70B-Instruct", "L", 5, 5, "L", 1, 1, 1, None, "", 1),
            ("nvidia", "qwen-2.5-72b", "Q", 6, 6, "L", 1, 1, 1, None, "", 1),
        ], raising=False)

        cands = _run(SP._router_candidates("live_answer"))
        provs = [c.provider for c in cands]
        assert provs[0] == "groq"                       # the router's primary
        assert "cerebras" in provs                      # same model elsewhere
        assert "nvidia" not in provs                    # different model excluded
        # All share the primary's canonical identity.
        assert len({c.cid_key for c in cands}) == 1

    def test_router_candidates_fail_open_on_no_route(self, monkeypatch):
        import app.llm.router as R

        async def none_route(**kw):
            return types.SimpleNamespace(route=None)
        monkeypatch.setattr(R, "select_route", none_route)
        assert _run(SP._router_candidates("live_answer")) == []
