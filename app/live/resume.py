"""
Session resume + qid correctness (live-conversational-intelligence R23).

Two related concerns for transport reliability:

  - **QidRegistry** — tracks each answer's `qid` as active → answered, so a
    cancel targets EXACTLY its own answer (never an unrelated concurrent one)
    and a late/duplicate/out-of-order STT final for an already-answered `qid`
    is a no-op (no duplicate answer).
  - **Session_Resume** — a per-session snapshot (active/answered qids) kept on
    reconnect so the new socket continues the same session without re-answering
    work that already streamed. The interview state machine, topic graph, and
    multi-level memory already persist in-process keyed by `sid`, so resume only
    needs the qid bookkeeping to avoid duplicates.

In-process; no DB. Deterministic + fail-open.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QidRegistry:
    """Per-session lifecycle tracking of answer `qid`s."""
    active: set = field(default_factory=set)
    answered: set = field(default_factory=set)

    def open(self, qid: str) -> None:
        if qid:
            self.active.add(qid)

    def close(self, qid: str) -> None:
        """Mark an answer finished (answered)."""
        if not qid:
            return
        self.active.discard(qid)
        self.answered.add(qid)

    def cancel(self, qid: str) -> bool:
        """Cancel an ACTIVE qid only. Returns True if it was active (so the
        caller cancels exactly that answer task); a qid that already finished
        or is unknown returns False (no-op)."""
        if qid and qid in self.active:
            self.active.discard(qid)
            return True
        return False

    def is_active(self, qid: str) -> bool:
        return bool(qid) and qid in self.active

    def is_answered(self, qid: str) -> bool:
        return bool(qid) and qid in self.answered

    def snapshot(self) -> dict:
        return {"active": sorted(self.active), "answered": sorted(self.answered)}


# ---- per-session registry (in-process; no DB) -------------------------
_registries: dict[str, QidRegistry] = {}


def get_registry(session_id: str) -> QidRegistry:
    reg = _registries.get(session_id)
    if reg is None:
        reg = QidRegistry()
        _registries[session_id] = reg
    return reg


def forget_session(session_id: str) -> None:
    _registries.pop(session_id, None)


def has_session(session_id: str) -> bool:
    """True when a snapshot exists for this session (a reconnect can resume)."""
    return session_id in _registries
