"""Wiring test — the four Phase 2 gap modules are actually USED in the live WS
pipeline (readiness/contract/rhythm/steer), gated + fail-open, and cleaned up on
teardown. Static source assertion (no socket/audio/models).
"""
from __future__ import annotations

import pathlib

_WS = pathlib.Path(__file__).resolve().parents[1] / "app" / "api" / "routes_ws.py"


def test_conversation_signals_wired():
    src = _WS.read_text(encoding="utf-8")
    # imports of all four
    for mod in ("readiness", "rhythm", "contract", "steer"):
        assert f"from app.live import {mod} as _" in src, f"{mod} not imported into pipeline"
    # each is invoked
    assert "_readiness.readiness_score(" in src
    assert "_rhythm.get_rhythm(sid)" in src and ".observe_now()" in src
    assert "_contract.ensure_contract(" in src
    assert "_steer.steering_directive(" in src
    # surfaced in meta
    for k in ('"answer_readiness"', '"cadence"', '"fatigue"', '"max_answer_seconds"'):
        assert k in src, f"meta field {k} not surfaced"
    # gated by a flag (default-safe)
    assert 'getattr(cfg.live, "conversation_signals"' in src


def test_conversation_signals_teardown():
    src = _WS.read_text(encoding="utf-8")
    assert "from app.live.contract import forget_session" in src
    assert "from app.live.rhythm import forget_session" in src


def test_end_to_end_signal_shapes():
    # Exercise the exact calls the pipeline makes, together.
    from app.live import contract as C
    from app.live import readiness as R
    from app.live import rhythm as RH
    from app.live import steer as S

    RH.forget_session("wire"); C.forget_session("wire")
    rt = RH.get_rhythm("wire"); rt.observe_now()
    assert rt.snapshot()["questions"] == 1
    assert 0.0 <= R.readiness_score(confidence=0.7) <= 1.0
    ct = C.ensure_contract("wire", phase="hr")
    assert ct.max_answer_seconds == 45  # HR is tighter
    assert S.steering_directive("tell me about yourself")  # open -> steers
    RH.forget_session("wire"); C.forget_session("wire")
