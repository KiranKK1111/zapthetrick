"""Run the mesh's quality agents (Clarifier, Grounder) standalone.

The upload-stream path answers via a direct LLM call rather than the agent mesh,
so it historically skipped clarification + grounding. These helpers run those
two agents on a throwaway blackboard, giving the upload path capability parity
with /api/agents/stream without rewriting its streaming onto the supervisor.
Both degrade to "nothing to add" on any failure.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


async def maybe_clarify(
    message: str,
    prior_messages: list[dict] | None = None,
    *,
    clarify_priority: bool = False,
    has_artifact: bool = False,
    attachment_slots: dict | None = None,
) -> list[dict]:
    """Ask the Clarifier whether this turn needs a clarifying question. Returns
    the question list (empty = answer directly), same shape the UI renders.

    `clarify_priority` (an ambiguous build request) forces a relevant ask with a
    deterministic fallback, mirroring the agent-mesh path. `has_artifact` —
    this turn carries uploaded files/images, so an "analyze my code" ask is
    answerable and must never trigger a "please attach it" clarification.
    `attachment_slots` — StackProfile slots detected inside the upload
    (Phase 2): they satisfy required slots so the gate never asks what the
    project already answers."""
    if not (message or "").strip():
        return []
    try:
        from app.agents.clarifier import ClarifierAgent, default_build_questions
        from app.blackboard.board import Blackboard
        from app.blackboard.schema import KEY_QUESTION

        board = Blackboard()
        board.write(KEY_QUESTION, message)
        _prior = prior_messages or []
        board.write("extras", {
            "prior_messages": _prior,
            "clarify_priority": clarify_priority,
            "has_attachments": has_artifact,
            "attachment_slots": {k: v for k, v
                                 in (attachment_slots or {}).items() if v},
            # Whether this chat already has content to archive/document —
            # without these the Clarifier can't tell an "archive the project"
            # turn has no target and may ask a nonsensical question.
            "has_prior_code": any(
                "```" in (m.get("content") or "") for m in _prior),
            "has_prior_content": any(
                m.get("role") == "assistant"
                and (m.get("content") or "").strip()
                for m in _prior),
        })
        await ClarifierAgent().run(board)
        qs = board.get("clarifying_questions", []) or []
        if clarify_priority and not qs:
            qs = default_build_questions()
        return qs
    except Exception as exc:  # noqa: BLE001 — never block the answer
        log.info("maybe_clarify failed: %s", exc)
        return []


async def check_grounding(answer: str, docs: list[tuple[str, str]]) -> list[str]:
    """Run the Grounder over `answer` against the uploaded `docs` (the turn's
    evidence). Returns claims the evidence does NOT support (empty = grounded)."""
    if not (answer or "").strip() or not docs:
        return []
    try:
        from app.agents.grounder import GrounderAgent
        from app.blackboard.board import Blackboard
        from app.blackboard.schema import (
            KEY_EVIDENCE,
            KEY_GROUNDING,
            Evidence,
            EvidenceChunk,
        )

        board = Blackboard()
        board.write("drafts_current", answer)
        chunks = [EvidenceChunk(text=(t or "")[:8000], source=fn, score=1.0,
                                parent_id=None)
                  for fn, t in docs if (t or "").strip()]
        if not chunks:
            return []
        board.write(KEY_EVIDENCE, Evidence(
            chunks=chunks, sources=[c.source for c in chunks],
            confidences=[1.0] * len(chunks)))
        await GrounderAgent().run(board)
        g = board.get(KEY_GROUNDING)
        return list(getattr(g, "unverified", []) or [])
    except Exception as exc:  # noqa: BLE001
        log.info("check_grounding failed: %s", exc)
        return []
