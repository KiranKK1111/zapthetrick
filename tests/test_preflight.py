"""Stage-6 §4.6 — Live pre-flight systems board (refuses a broken session)."""
from __future__ import annotations

import asyncio

import pytest

from app.live import preflight as PF
from app.live.preflight import PreflightProbes, run_preflight


def _run(coro):
    return asyncio.run(coro)


def _ok():
    async def _p():
        return True, "ok"
    return _p


def _fail(detail="down"):
    async def _p():
        return False, detail
    return _p


def _boom():
    async def _p():
        raise RuntimeError("probe exploded")
    return _p


def _all_ok() -> PreflightProbes:
    return PreflightProbes(
        backend=_ok(), interviewer_audio=_ok(), mic_channel=_ok(),
        stt_roundtrip=_ok(), llm_first_token=_ok(), gpu_lanes=_ok(),
        session_context=_ok())


def _check(board, name):
    return next(c for c in board.checks if c.name == name)


class TestGreenBoard:
    def test_all_green_is_ready(self):
        board = _run(run_preflight("s1", "live_answer", probes=_all_ok()))
        assert board.ready is True
        assert board.blocking_failures == []

    def test_board_has_all_checks(self):
        board = _run(run_preflight("s1", "live_answer", probes=_all_ok()))
        names = {c.name for c in board.checks}
        assert {"backend", "stt_roundtrip", "llm_first_token", "model_plan",
                "mic_channel", "gpu_lanes", "session_context",
                "interviewer_audio"} <= names


class TestBlockingFailures:
    @pytest.mark.parametrize("bad", ["backend", "stt_roundtrip",
                                     "llm_first_token"])
    def test_blocking_failure_refuses_session(self, bad):
        probes = _all_ok()
        setattr(probes, bad, _fail())
        board = _run(run_preflight("s1", "live_answer", probes=probes))
        assert board.ready is False
        assert bad in [c.name for c in board.blocking_failures]
        # A refused check carries an actionable fix hint.
        assert _check(board, bad).hint

    def test_non_blocking_failure_still_ready(self):
        # mic / gpu / context / interviewer_audio are advisory, not blocking.
        for soft in ("mic_channel", "gpu_lanes", "session_context",
                     "interviewer_audio"):
            probes = _all_ok()
            setattr(probes, soft, _fail())
            board = _run(run_preflight("s1", "live_answer", probes=probes))
            assert board.ready is True, soft
            assert _check(board, soft).ok is False


class TestFailOpen:
    def test_probe_error_is_reported_not_fatal(self):
        probes = _all_ok()
        probes.stt_roundtrip = _boom()
        board = _run(run_preflight("s1", "live_answer", probes=probes))
        # A probe ERROR is unknown (None), never a blocking refusal.
        assert board.ready is True
        assert _check(board, "stt_roundtrip").ok is None

    def test_absent_probe_is_unknown_not_blocking(self):
        # No probes at all → every env check is None → the board never refuses.
        board = _run(run_preflight("s1", "live_answer"))
        assert board.ready is True
        assert _check(board, "backend").ok is None


class TestModelPlanCheck:
    def test_reads_the_live_plan_pin(self, monkeypatch):
        from app.core.config_loader import cfg
        from app.llm import live_plan as LP
        from app.llm.live_plan import Candidate
        monkeypatch.setattr(cfg.routing, "live_plan", True, raising=False)
        LP.reset_for_tests()
        LP.live_planner().plan(
            "s1", "live_answer",
            [Candidate("llama|70|3.3|-", "groq", key_id=1),
             Candidate("llama|70|3.3|-", "cerebras", key_id=2)],
            reserve=False)
        board = _run(run_preflight("s1", "live_answer", probes=_all_ok()))
        assert _check(board, "model_plan").ok is True
        assert "groq" in _check(board, "model_plan").detail
        LP.reset_for_tests()

    def test_no_pin_when_plan_enabled_refuses(self, monkeypatch):
        from app.core.config_loader import cfg
        from app.llm import live_plan as LP
        monkeypatch.setattr(cfg.routing, "live_plan", True, raising=False)
        LP.reset_for_tests()
        board = _run(run_preflight("no-session", "live_answer", probes=_all_ok()))
        assert board.ready is False
        assert _check(board, "model_plan").ok is False

    def test_plan_disabled_is_non_blocking(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.routing, "live_plan", False, raising=False)
        board = _run(run_preflight("s1", "live_answer", probes=_all_ok()))
        mp = _check(board, "model_plan")
        assert mp.ok is None and mp.blocking is False
        assert board.ready is True


class TestFlag:
    def test_enabled_default_off(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "preflight", False, raising=False)
        assert PF.enabled() is False

    def test_enabled_reads_flag(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "preflight", True, raising=False)
        assert PF.enabled() is True
