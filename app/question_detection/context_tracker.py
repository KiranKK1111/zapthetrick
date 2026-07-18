"""
Conversation-context tracker.

Keeps a per-session memory of recent (question, answer, embedding,
timestamp) tuples. Used by the classifier (for follow-up detection) and
the orchestrator (so a follow-up answer can read prior Q+A).

In-memory only — fine for a single-user desktop client, which is the
Phase-4 target. A future phase can swap this for a Redis or per-user DB
table without touching callers.
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from time import time

from app.core.config_loader import cfg
from app.rag import embedder


@dataclass
class Turn:
    """One Q+A turn from an interview, plus the question's embedding."""
    question: str
    answer: str = ""
    embedding: list[float] = field(default_factory=list)
    topic: str = ""
    qtype: str = "unknown"
    timestamp: float = field(default_factory=time)


class ContextTracker:
    """Per-session ring buffer of recent interview turns."""

    def __init__(self, max_turns: int = 20):
        self._turns: deque[Turn] = deque(maxlen=max_turns)

    def recent_questions(self, n: int | None = None) -> list[str]:
        """Most recent N questions, oldest first. Default: configured window."""
        n = n or cfg.question_detection.recent_q_window
        items = list(self._turns)[-n:]
        return [t.question for t in items]

    def last_turn(self) -> Turn | None:
        return self._turns[-1] if self._turns else None

    async def add_question(
        self, question: str, qtype: str, topic: str, embedding: list[float] | None = None
    ) -> Turn:
        """Record a new question. If the caller already computed the question
        embedding (the orchestrator does, with a timeout), pass it in to avoid
        a second bge-m3 call. Falls back to a threaded compute otherwise."""
        emb = embedding
        if emb is None:
            try:
                emb = await asyncio.to_thread(embedder.embed_one, question)
            except Exception:  # noqa: BLE001 -- embedder cold/unavailable
                emb = []
        turn = Turn(question=question, qtype=qtype, topic=topic, embedding=emb or [])
        self._turns.append(turn)
        return turn

    def complete_answer(self, turn: Turn, answer: str) -> None:
        """Attach the final assistant answer once streaming finishes."""
        turn.answer = answer

    def is_followup(self, new_q_embedding: list[float]) -> bool:
        """Return True if the new question is similar enough to the most recent
        and arrived within `followup_window_seconds` of it."""
        last = self.last_turn()
        if last is None or not last.embedding:
            return False
        if time() - last.timestamp > cfg.question_detection.followup_window_seconds:
            return False
        sim = _cosine(new_q_embedding, last.embedding)
        return sim >= cfg.question_detection.followup_similarity_threshold


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors. Returns 0.0 on mismatch."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    # Embeddings from the bge models are already L2-normalised, so dot == cosine.
    return dot


# ---- Session registry --------------------------------------------------
# Maps a chat-session id (UUID string) to its tracker. Sessions are
# garbage-collected by `prune_inactive` when they go untouched.
_sessions: dict[str, ContextTracker] = {}


def get_tracker(session_id: str) -> ContextTracker:
    """Return (and create if needed) the tracker for a given chat session."""
    tracker = _sessions.get(session_id)
    if tracker is None:
        tracker = ContextTracker()
        _sessions[session_id] = tracker
    return tracker


def forget_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
