"""Memory — recalls similar past episodes and distilled semantic skills.

P1 — runs in parallel with Persona. If it's not done by the time the
response stream finishes, the supervisor drops its contribution rather
than blocking the user.

Reads `extras.db_session` + `extras.session_id` from the blackboard
and queries:
  - [search_episodes_similar] for past Q&As that look like this one
  - [relevant_skills_for_question] for distilled lessons that apply

Writes a [MemoryHits] block with up to N episodes and skills. The
Persona agent doesn't currently consume this — the next step is to
inject these into the system prompt the same way Evidence is injected.
"""
from __future__ import annotations

from ..blackboard.board import Blackboard
from ..blackboard.schema import KEY_MEMORY_HITS, KEY_QUESTION, MemoryHits
from ..blackboard.scheduler import P1
from .base import Agent


class MemoryAgent(Agent):
    name = "memory"
    priority = P1
    expected_latency_ms = 200
    reads = frozenset({KEY_QUESTION})
    writes = frozenset({KEY_MEMORY_HITS})

    async def run(self, board: Blackboard) -> None:
        extras = board.get("extras", {}) or {}
        db_session = extras.get("db_session")
        session_id = extras.get("session_id")
        # §17: when the conversation is in a project, recall scopes to the whole
        # project (every sibling chat), not just this session.
        project_id = extras.get("project_id")
        question = board.get(KEY_QUESTION, "")

        if db_session is None or not question.strip():
            board.write(KEY_MEMORY_HITS, MemoryHits(), agent=self.name)
            return

        from ..memory import (
            relevant_skills_for_question,
            search_episodes_similar,
        )

        try:
            episodes = await search_episodes_similar(
                db_session, question, session_id=session_id,
                project_id=project_id, top_k=3
            )
        except Exception:
            episodes = []
        try:
            skills = await relevant_skills_for_question(
                db_session, question, session_id=session_id,
                project_id=project_id, top_k=3
            )
        except Exception:
            skills = []

        board.write(
            KEY_MEMORY_HITS,
            MemoryHits(
                episodes=[
                    {
                        "id": ep.id,
                        "question": ep.question,
                        "final": ep.final[:280],
                        "intent": ep.intent,
                        "feedback": ep.feedback,
                    }
                    for ep in episodes
                ],
                skills=[
                    {
                        "id": s.id,
                        "text": s.text,
                        "kind": s.kind,
                        "confidence": s.confidence,
                    }
                    for s in skills
                ],
            ),
            agent=self.name,
        )
