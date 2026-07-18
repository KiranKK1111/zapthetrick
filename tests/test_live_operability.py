"""Live knowledge, prediction & operability
(live-conversational-intelligence R29, R35, R36, R37, R38; tasks 25.3).

Pins Properties 29, 35, 36, 37, 38: question prediction ranking + precompute
gating, interview-knowledge retrieval + directive, in-session gap detection +
recovery from summary, session-health warnings, and delivery coaching.
"""
from __future__ import annotations

from app.live import coach, health, knowledge, predict, validate
from app.live.world_model import InterviewWorldModel


# ---- question prediction ----------------------------------------------
def test_predict_next_ranks_known_subtopics():
    w = InterviewWorldModel(topic="kafka")
    preds = predict.predict_next(world_model=w, max_n=5)
    assert preds
    assert any("partitions" in p.lower() for p in preds)
    assert len(preds) <= 5


def test_predict_next_empty_without_topic():
    assert predict.predict_next(world_model=InterviewWorldModel()) == []


def test_should_precompute_requires_speculation(monkeypatch=None):
    from app.core.config_loader import cfg
    saved = cfg.perceived.speculation_enabled
    try:
        cfg.perceived.speculation_enabled = False
        assert predict.should_precompute() is False
        cfg.perceived.speculation_enabled = True
        assert predict.should_precompute() is True
    finally:
        cfg.perceived.speculation_enabled = saved


# ---- interview knowledge ----------------------------------------------
def test_interview_knowledge_known_topic():
    snips = knowledge.interview_knowledge("kafka")
    assert snips
    assert "Relevant angles" in knowledge.directive(snips)


def test_interview_knowledge_unknown_topic():
    assert knowledge.interview_knowledge("quantum basket weaving") == []
    assert knowledge.directive([]) == ""


# ---- state validation + recovery --------------------------------------
def test_gap_detected_when_topic_but_no_active_question():
    w = InterviewWorldModel(topic="kafka")
    assert validate.detect_gap(w) is True


def test_recover_from_summary():
    w = InterviewWorldModel()
    gap, recovered = validate.validate_and_recover(
        w, summary="Topics covered: kafka, redis. Recent questions: ...",
        recent_questions=["q"])
    assert gap is True
    assert recovered is True
    assert w.topic == "redis"


def test_no_gap_when_consistent():
    w = InterviewWorldModel(topic="kafka", active_question="How do partitions work?")
    assert validate.detect_gap(w) is False


def test_should_validate_periodic():
    assert validate.should_validate(5, interval=5) is True
    assert validate.should_validate(3, interval=5) is False


# ---- session health ----------------------------------------------------
def test_health_warns_on_low_stt_and_latency():
    h = health.session_health(stt_conf=0.2, latency_ms=6000)
    assert h is not None
    assert "low_stt_confidence" in h["warnings"]
    assert "high_latency" in h["warnings"]


def test_health_none_when_ok():
    assert health.session_health(stt_conf=0.9, latency_ms=500) is None


# ---- delivery coaching -------------------------------------------------
def test_coach_flags_fillers_and_missing_example():
    tips = coach.coach(
        "um so like basically you know i would um use a cache and like store stuff there "
        "and you know it would be fast and basically scale well over time somehow")
    assert any("filler" in t.lower() for t in tips)


def test_coach_brief_answer():
    assert any("detail" in t.lower() for t in coach.coach("Yes."))


def test_coach_empty():
    assert coach.coach("") == []


# ---- delivery coaching is WIRED into the live pipeline ------------------
# `app/live/coach.py` had a `true` config flag and no consumer. It is now
# consumed by app/api/routes_ws.py: the candidate channel emits the tips on
# their OWN meta frame. These pin that (a) the tips appear when the flag is on,
# (b) they vanish (fail-open) when the module raises, and (c) they can never
# reach the interview answer we generate FOR the candidate.

_FILLER_ANSWER = (
    "um so like basically you know i would um use a cache and like store stuff "
    "there and you know it would be fast and basically scale well over time")


def _ws_app():
    """A standalone app carrying only the live WS router — never imports
    app.main (which loads ML models)."""
    from fastapi import FastAPI

    from app.api.routes_ws import router

    app = FastAPI()
    app.include_router(router)
    return app


def _drain_until_ready(ws) -> None:
    while ws.receive_json().get("type") != "ready":
        pass


def _candidate_frames(ws, text: str) -> list[dict]:
    """Send candidate speech, then a `source_role` control frame whose reply is
    a deterministic SENTINEL — so we can read every frame the candidate turn
    produced without ever blocking on a frame that may not come."""
    ws.send_json({"type": "candidate_text", "content": text})
    ws.send_json({"type": "source_role", "role": "interviewer"})
    frames: list[dict] = []
    while True:
        f = ws.receive_json()
        if f.get("channel_role") == "interviewer":
            return frames
        frames.append(f)


def test_candidate_delivery_coaching_surfaces_on_its_own_frame(monkeypatch):
    from starlette.testclient import TestClient

    from app.core.config_loader import cfg

    monkeypatch.setattr(cfg.live, "delivery_coaching", True)
    monkeypatch.setattr(cfg.live, "candidate_channel", True)

    with TestClient(_ws_app()).websocket_connect("/ws/live") as ws:
        _drain_until_ready(ws)
        frames = _candidate_frames(ws, _FILLER_ANSWER)

    # The candidate's speech is absorbed...
    assert any(f.get("absorbed") is True for f in frames)
    # ...and the coaching rides on a SEPARATE frame the FE can render.
    coaching = [f for f in frames if f.get("coaching")]
    assert len(coaching) == 1
    assert coaching[0]["role"] == "candidate"
    assert any("filler" in t.lower() for t in coaching[0]["coaching"])
    # It is metadata, not answer content: the candidate turn generates no
    # answer at all, so the coaching cannot pollute what we tell the candidate.
    assert not any(f.get("type") in ("token", "done") for f in frames)
    assert not any("directive" in f for f in frames)


def test_candidate_coaching_absent_but_turn_succeeds_when_coach_raises(
        monkeypatch):
    from starlette.testclient import TestClient

    from app.core.config_loader import cfg
    from app.live import coach as _coach

    monkeypatch.setattr(cfg.live, "delivery_coaching", True)
    monkeypatch.setattr(cfg.live, "candidate_channel", True)

    def _boom(text):
        raise RuntimeError("coach exploded")

    monkeypatch.setattr(_coach, "coach", _boom)

    with TestClient(_ws_app()).websocket_connect("/ws/live") as ws:
        _drain_until_ready(ws)
        frames = _candidate_frames(ws, _FILLER_ANSWER)

    # FAIL-OPEN: no coaching frame, but the turn behaves exactly as before.
    assert not any(f.get("coaching") for f in frames)
    assert any(f.get("absorbed") is True for f in frames)


def test_candidate_coaching_absent_when_flag_off(monkeypatch):
    from starlette.testclient import TestClient

    from app.core.config_loader import cfg

    monkeypatch.setattr(cfg.live, "delivery_coaching", False)
    monkeypatch.setattr(cfg.live, "candidate_channel", True)

    with TestClient(_ws_app()).websocket_connect("/ws/live") as ws:
        _drain_until_ready(ws)
        frames = _candidate_frames(ws, _FILLER_ANSWER)

    assert not any(f.get("coaching") for f in frames)
    assert any(f.get("absorbed") is True for f in frames)


def test_coaching_tips_helper_fails_open(monkeypatch):
    from app.api import routes_ws as _ws
    from app.core.config_loader import cfg
    from app.live import coach as _coach

    monkeypatch.setattr(cfg.live, "delivery_coaching", True)
    assert _ws._coaching_tips(_FILLER_ANSWER)          # flag on -> tips

    monkeypatch.setattr(_coach, "coach",
                        lambda t: (_ for _ in ()).throw(RuntimeError("boom")))
    assert _ws._coaching_tips(_FILLER_ANSWER) == []    # raises -> silent []

    monkeypatch.setattr(cfg.live, "delivery_coaching", False)
    assert _ws._coaching_tips(_FILLER_ANSWER) == []    # flag off -> nothing
