"""Reflector — background analysis of completed sessions.

P2 — runs after the user-facing response is done. Architecture.md §4:

  Reads episodes from [memory.episodic]:
    - what questions did the user accept first try?
    - what was edited / regenerated?
    - which retrievals were ignored?
  Extracts new semantic-memory entries (preferences, working patterns).
  Updates skill embeddings.

Today this is upvote/downvote pattern extraction via [extract_skills],
persisted to the `skills` table via [record_skill]. Idempotent across
runs: [record_skill] de-dupes on `(session_id, text)`.

TODO: LLM-driven reflection on the actual draft text — story names
that recurred, framing patterns, weak phrasing. The heuristic is the
floor, not the ceiling.
"""
from __future__ import annotations

from ..blackboard.board import Blackboard
from ..blackboard.scheduler import P2
from .base import Agent


_REFLECTION_WINDOW = 20


class ReflectorAgent(Agent):
    name = "reflector"
    priority = P2
    expected_latency_ms = 5_000
    reads = frozenset({"drafts_current"})
    writes = frozenset({"semantic_updates"})

    async def run(self, board: Blackboard) -> None:
        extras = board.get("extras", {}) or {}
        session_id = extras.get("session_id")

        if not session_id:
            board.write("semantic_updates", [], agent=self.name)
            return

        # P2 agents run after the user-facing route returns, so the
        # request-scoped DB session from `extras` is already closed.
        # Open a fresh one from the factory and manage its lifetime.
        from ..database import SessionFactory
        from ..memory import extract_skills, record_skill
        from ..memory.episodic import recent_episodes

        try:
            async with SessionFactory() as fresh_session:
                episodes = await recent_episodes(
                    fresh_session,
                    session_id=session_id,
                    limit=_REFLECTION_WINDOW,
                )
                skills = extract_skills(episodes, session_id=session_id)
                persisted_ids: list[str] = []
                for skill in skills:
                    try:
                        sid = await record_skill(fresh_session, skill)
                        persisted_ids.append(sid)
                    except Exception:
                        continue
        except Exception:
            board.write("semantic_updates", [], agent=self.name)
            return

        board.write(
            "semantic_updates",
            [
                {
                    "id": sid,
                    "text": s.text,
                    "kind": s.kind,
                    "confidence": s.confidence,
                }
                for sid, s in zip(persisted_ids, skills)
            ],
            agent=self.name,
        )
