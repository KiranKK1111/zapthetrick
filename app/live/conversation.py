"""
Role-aware shared conversation graph
(live-conversational-intelligence — dual-source continuity).

The live memory graph must retain ALL three voices of an interview, not just the
interviewer's detected questions:

  - **interviewer** — the questions/challenges (drives what the assistant shows)
  - **candidate**   — what the candidate actually SAID out loud (absorbed, never
                      answered) so later suggestions build on their real words
  - **assistant**   — what we already suggested, so follow-ups stay consistent

`ConversationLog` is a bounded, per-session, role-tagged transcript attached to
the existing context tracker (in-process; no DB). `context_lines` renders it as a
readable "Interviewer / You / Assistant" transcript the answer path can fold in.
Deterministic + fail-open.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from time import time

INTERVIEWER = "interviewer"
CANDIDATE = "candidate"
ASSISTANT = "assistant"

_ROLE_LABEL = {INTERVIEWER: "Interviewer", CANDIDATE: "You", ASSISTANT: "Assistant suggested"}


@dataclass
class ConversationEntry:
    role: str
    text: str
    topic: str = ""
    ts: float = field(default_factory=time)

    def to_dict(self) -> dict:
        return {"role": self.role, "text": self.text, "topic": self.topic,
                "ts": round(self.ts, 3)}


class ConversationLog:
    """Bounded, role-tagged per-session transcript."""

    def __init__(self, maxlen: int = 80) -> None:
        self._entries: deque = deque(maxlen=maxlen)

    def add(self, role: str, text: str, topic: str = "") -> None:
        try:
            r = (role or INTERVIEWER).strip().lower()
            t = (text or "").strip()
            if not t:
                return
            if r not in (INTERVIEWER, CANDIDATE, ASSISTANT):
                r = INTERVIEWER
            self._entries.append(ConversationEntry(role=r, text=t, topic=(topic or "").strip()))
        except Exception:  # noqa: BLE001
            pass

    def entries(self) -> list[ConversationEntry]:
        return list(self._entries)

    def for_topic(self, topic: str) -> list[ConversationEntry]:
        name = (topic or "").strip().lower()
        if not name:
            return []
        out = []
        for e in self._entries:
            et = (e.topic or "").lower()
            if et and (et == name or et in name or name in et):
                out.append(e)
        return out

    def context_lines(self, topic: str = "", n: int = 8) -> list[str]:
        """Render the recent transcript (topic-scoped first, then recent) as
        role-labeled lines. Never raises → []."""
        try:
            picked: list[ConversationEntry] = []
            seen = set()
            for e in self.for_topic(topic):
                key = (e.role, e.text)
                if key not in seen:
                    seen.add(key)
                    picked.append(e)
            for e in list(self._entries)[-n:]:
                key = (e.role, e.text)
                if key not in seen:
                    seen.add(key)
                    picked.append(e)
            picked = picked[-n:]
            lines = []
            for e in picked:
                label = _ROLE_LABEL.get(e.role, e.role.title())
                lines.append(f"{label}: {e.text}")
            return lines
        except Exception:  # noqa: BLE001
            return []

    def last_candidate(self) -> str:
        for e in reversed(self._entries):
            if e.role == CANDIDATE:
                return e.text
        return ""


def for_tracker(tracker) -> ConversationLog:
    """Per-session ConversationLog attached to the context tracker (lazily)."""
    c = getattr(tracker, "_live_conversation", None)
    if c is None:
        c = ConversationLog()
        try:
            tracker._live_conversation = c
        except Exception:  # noqa: BLE001
            pass
    return c
