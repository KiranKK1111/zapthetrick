"""
Explicit interview state machine (live-conversational-intelligence R4).

Tracks the live session's state across:

    IDLE -> LISTENING -> CONTEXT_BUILDING -> QUESTION_CONFIRMED -> ANSWERING
         -> FOLLOWUP_WAITING ; TOPIC_SWITCHING ; INTERRUPTED ; QUESTION_HYPOTHESIS

Deterministic and per-session. It **records** state for the additive
`{"type":"state"}` frame and never **gates** the existing concurrency — two
questions asked close together are still answered simultaneously (each with its
own `qid`); the machine simply stays in ANSWERING until the last answer
finishes. Disabled / on error the live loop behaves exactly as today.
"""
from __future__ import annotations

from app.live import events as _events

IDLE = "IDLE"
LISTENING = "LISTENING"
CONTEXT_BUILDING = "CONTEXT_BUILDING"
QUESTION_HYPOTHESIS = "QUESTION_HYPOTHESIS"
QUESTION_CONFIRMED = "QUESTION_CONFIRMED"
ANSWERING = "ANSWERING"
FOLLOWUP_WAITING = "FOLLOWUP_WAITING"
TOPIC_SWITCHING = "TOPIC_SWITCHING"
INTERRUPTED = "INTERRUPTED"

STATES = {
    IDLE, LISTENING, CONTEXT_BUILDING, QUESTION_HYPOTHESIS, QUESTION_CONFIRMED,
    ANSWERING, FOLLOWUP_WAITING, TOPIC_SWITCHING, INTERRUPTED,
}


class InterviewStateMachine:
    """Per-session interview state. Pure/deterministic; concurrency-safe in the
    sense that it never blocks answers — it only tracks how many are active."""

    def __init__(self) -> None:
        self.state: str = IDLE
        self._active_answers: int = 0
        self._last_event_kind: str = ""

    def advance(self, event: "_events.UtteranceEvent") -> str:
        """Move the state forward for a typed utterance. Returns the new state.

        Does not touch answer tracking — answer start/done drive ANSWERING via
        `on_answer_start`/`on_answer_done` so concurrency is preserved."""
        kind = getattr(event, "kind", "") or ""
        self._last_event_kind = kind
        if kind == _events.TOPIC_CHANGE:
            self.state = TOPIC_SWITCHING
        elif kind in _events.ANSWERABLE:
            # A confirmed question. If we're mid-answer, stay ANSWERING (a
            # concurrent question), else mark it confirmed/ready.
            self.state = ANSWERING if self._active_answers > 0 else QUESTION_CONFIRMED
        elif kind in (_events.EXPLANATION, _events.GREETING, _events.SMALL_TALK,
                      _events.TRANSITION, _events.ACKNOWLEDGEMENT,
                      _events.ANSWER_HINT):
            # Non-answerable speech: building context (don't disturb an
            # in-flight answer's ANSWERING state).
            if self._active_answers == 0:
                self.state = CONTEXT_BUILDING
        return self.state

    def on_answer_start(self) -> str:
        """An answer began streaming (one per `qid`)."""
        self._active_answers += 1
        self.state = ANSWERING
        return self.state

    def on_answer_done(self) -> str:
        """An answer finished. When the last concurrent answer ends, move to
        FOLLOWUP_WAITING (ready for the next turn)."""
        if self._active_answers > 0:
            self._active_answers -= 1
        if self._active_answers == 0:
            self.state = FOLLOWUP_WAITING
        return self.state

    def mark_interrupted(self) -> str:
        self.state = INTERRUPTED
        return self.state

    def mark_listening(self) -> str:
        if self._active_answers == 0:
            self.state = LISTENING
        return self.state

    def snapshot(self) -> dict:
        """Serializable view for the additive `state` WebSocket frame."""
        return {
            "state": self.state,
            "active_answers": self._active_answers,
            "last_event": self._last_event_kind,
        }


# ---- Per-session registry (in-process; no DB) --------------------------
_machines: dict[str, InterviewStateMachine] = {}


def get_state_machine(session_id: str) -> InterviewStateMachine:
    """Return (creating if needed) the state machine for a live session."""
    sm = _machines.get(session_id)
    if sm is None:
        sm = InterviewStateMachine()
        _machines[session_id] = sm
    return sm


def forget_session(session_id: str) -> None:
    _machines.pop(session_id, None)
