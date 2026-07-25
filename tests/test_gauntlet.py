"""Stage-5 §2.5 — onboarding gauntlet: quarantine gate, probe battery, re-probe."""
from __future__ import annotations

import asyncio
import types

import pytest

import app.llm.router as R
from app.llm import gauntlet as G
from app.llm.gauntlet import Gauntlet, ProbeSuite, Scorecard
from app.llm.identity import canonicalize


def _run(coro):
    return asyncio.run(coro)


class _Clock:
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


@pytest.fixture(autouse=True)
def _fresh():
    G.reset_for_tests()
    yield
    G.reset_for_tests()


@pytest.fixture
def _on(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.routing, "gauntlet", True, raising=False)


# --------------------------------------------------------------------------- #
class TestQuarantine:
    def test_new_pair_is_quarantined(self):
        g = Gauntlet()
        assert g.is_quarantined("llama|70|3.3|-", "groq") is True
        assert g.is_probed("llama|70|3.3|-", "groq") is False

    def test_probed_pair_is_not_quarantined(self):
        g = Gauntlet()
        _run(g.run_battery("llama|70|3.3|-", "groq", probes=ProbeSuite()))
        assert g.is_quarantined("llama|70|3.3|-", "groq") is False

    def test_module_gate_off_never_quarantines(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "gauntlet", False, raising=False)
        assert G.is_quarantined("anything|1|-|-", "groq") is False


class TestProbeBattery:
    def test_battery_writes_a_scorecard_from_stubbed_probes(self):
        g = Gauntlet()

        async def schema(c, p): return 0.99
        async def instr(c, p): return 0.8
        async def code(c, p): return 0.6
        async def needle(c, p): return 0.7
        async def timed(c, p): return (0.4, 55.0)
        async def conf(c, p): return {"streaming": True, "tools": False}

        card = _run(g.run_battery(
            "qwen|72|2.5|-", "cerebras",
            probes=ProbeSuite(schema=schema, instruction=instr, code_smoke=code,
                              needle=needle, timed=timed, conformance=conf)))
        assert card.json_reliability == pytest.approx(0.99)
        assert card.quality_prior == pytest.approx(0.8)
        assert card.coder_prior == pytest.approx(0.6)
        assert card.context_effective == pytest.approx(0.7)
        assert card.ttft_s == pytest.approx(0.4) and card.tps == pytest.approx(55.0)
        assert card.capabilities["streaming"] is True
        assert card.probed_at > 0

    def test_code_smoke_failure_lowers_the_coder_prior(self):
        g = Gauntlet()

        async def bad_code(c, p): return 0.1     # most code-smoke tasks failed
        card = _run(g.run_battery("m|7|-|-", "x",
                                  probes=ProbeSuite(code_smoke=bad_code)))
        assert card.coder_prior == pytest.approx(0.1)

    def test_probe_error_is_neutral_not_fatal(self):
        g = Gauntlet()

        async def boom(c, p): raise RuntimeError("probe harness bug")
        card = _run(g.run_battery("m|7|-|-", "x",
                                  probes=ProbeSuite(schema=boom)))
        # A probe HARNESS failure leaves the stat neutral (1.0) and still probes.
        assert card.json_reliability == 1.0
        assert card.probed_at > 0


class TestReprobe:
    def test_never_probed_needs_reprobe(self):
        g = Gauntlet()
        assert g.needs_reprobe("m|7|-|-", "x") is True

    def test_fresh_probe_does_not_need_reprobe(self, ):
        clock = _Clock()
        g = Gauntlet(now=clock)
        _run(g.run_battery("m|7|-|-", "x", probes=ProbeSuite()))
        assert g.needs_reprobe("m|7|-|-", "x") is False

    def test_stale_probe_needs_reprobe(self):
        clock = _Clock()
        g = Gauntlet(now=clock)
        _run(g.run_battery("m|7|-|-", "x", probes=ProbeSuite()))
        clock.advance(31 * 86_400)
        assert g.needs_reprobe("m|7|-|-", "x") is True

    def test_error_signature_change_triggers_reprobe(self):
        g = Gauntlet()
        _run(g.run_battery("m|7|-|-", "x", probes=ProbeSuite(),
                           error_signature="sig-A"))
        assert g.needs_reprobe("m|7|-|-", "x", error_signature="sig-A") is False
        assert g.needs_reprobe("m|7|-|-", "x", error_signature="sig-B") is True

    def test_note_error_signature_invalidates_probe(self):
        g = Gauntlet()
        _run(g.run_battery("m|7|-|-", "x", probes=ProbeSuite(),
                           error_signature="sig-A"))
        g.note_error_signature("m|7|-|-", "x", "sig-B")
        assert g.is_quarantined("m|7|-|-", "x") is True   # re-quarantined


# --------------------------------------------------------------------------- #
def _cand(model_id, platform):
    m = types.SimpleNamespace(id=id(model_id) % 100000, model_id=model_id,
                              platform=platform)
    return {"model": m, "cid": canonicalize(platform, model_id).key(),
            "score": 1.0}


class TestRouterFilter:
    def test_unprobed_dropped_when_a_probed_one_exists(self, _on):
        # groq's llama is PROBED (known-good); nvidia's is not.
        G.gauntlet().record("llama-3-3|70|3.3|-",
                            "groq", Scorecard(probed_at=1.0))
        good = _cand("llama-3.3-70b-versatile", "groq")
        unproven = _cand("some-new-90b", "nvidia")
        out = R._quarantine_pool([good, unproven])
        ids = {c["model"].model_id for c in out}
        assert "llama-3.3-70b-versatile" in ids
        assert "some-new-90b" not in ids              # quarantined out

    def test_all_unprobed_keeps_pool_never_empty(self, _on):
        # Everything is unproven → availability wins, pool returned untouched.
        pool = [_cand("a-70b", "groq"), _cand("b-70b", "nvidia")]
        assert len(R._quarantine_pool(pool)) == 2

    def test_off_is_byte_identical(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "gauntlet", False, raising=False)
        pool = [_cand("a-70b", "groq"), _cand("b-70b", "nvidia")]
        assert R._quarantine_pool(pool) is pool        # no filtering at all


class TestPersistence:
    def test_snapshot_round_trips(self, _on):
        g = G.gauntlet()
        _run(g.run_battery("m|7|-|-", "x",
                           probes=ProbeSuite(code_smoke=_const(0.42))))
        rows = g.snapshot()
        assert rows and rows[0]["cid"] == "m|7|-|-" and rows[0]["provider"] == "x"
        # Re-load into a fresh gauntlet (rehydrate's inner path).
        g.clear()
        r = rows[0]
        cid = r.pop("cid"); prov = r.pop("provider")
        g.record(cid, prov, Scorecard(**r))
        assert g.scorecard("m|7|-|-", "x").coder_prior == pytest.approx(0.42)

    def test_rehydrate_no_db_returns_zero(self, _on):
        assert _run(G.rehydrate()) == 0


def _const(v):
    async def _f(c, p):
        return v
    return _f
