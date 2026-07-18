"""Reflector pipeline — turn completed episodes into [Skill] entries.

The [ReflectorAgent] runs this on idle or at session end. Reads recent
[Episode]s from [EpisodicMemory] / the `episodes` table, looks for
patterns the user accepted or rejected, emits new [Skill]s for
[SemanticMemory] / the `skills` table.

Two signal families today:
  - **Upvote clusters** — when the user thumbs-up multiple turns of the
    same intent type, we record a `pattern` skill ("…responds well to
    the current X framing").
  - **Downvote clusters** — same shape, but a `gap` skill telling
    future runs the current approach isn't landing.

TODO: LLM-driven reflection on the actual draft text — story names
that recurred, framing patterns, weak phrasing. Heuristic clusters are
the floor, not the ceiling.
"""
from __future__ import annotations

from .episodic import Episode
from .semantic import Skill


def _scope_from(evidence: list[Episode]) -> tuple[str | None, str | None]:
    """(user_id, project_id) inherited from the evidence episodes (§17/§18) —
    they share a session, so the first with a value is authoritative."""
    uid = next((ep.user_id for ep in evidence if ep.user_id), None)
    pid = next((ep.project_id for ep in evidence if ep.project_id), None)
    return uid, pid


def extract_skills(
    episodes: list[Episode],
    *,
    session_id: str = "",
    max_skills: int = 5,
) -> list[Skill]:
    out: list[Skill] = []
    upvoted = [ep for ep in episodes if ep.feedback == "up"]
    downvoted = [ep for ep in episodes if ep.feedback == "down"]

    # Cheap pattern: if multiple upvoted episodes share an intent, flag
    # that as a working approach.
    intent_count: dict[str, int] = {}
    for ep in upvoted:
        intent_count[ep.intent] = intent_count.get(ep.intent, 0) + 1
    for intent, n in sorted(intent_count.items(), key=lambda kv: -kv[1]):
        if n >= 2:
            _ev = [ep for ep in upvoted if ep.intent == intent]
            _uid, _pid = _scope_from(_ev)
            out.append(
                Skill(
                    session_id=session_id,
                    user_id=_uid,
                    project_id=_pid,
                    text=f"User responds well to the current {intent} framing.",
                    kind="pattern",
                    confidence=min(0.9, 0.5 + 0.1 * n),
                    evidence_episode_ids=[ep.id for ep in _ev],
                )
            )
        if len(out) >= max_skills:
            return out

    # Mirror image for downvotes — surface as gaps.
    gap_count: dict[str, int] = {}
    for ep in downvoted:
        gap_count[ep.intent] = gap_count.get(ep.intent, 0) + 1
    for intent, n in sorted(gap_count.items(), key=lambda kv: -kv[1]):
        if n >= 2:
            _ev = [ep for ep in downvoted if ep.intent == intent]
            _uid, _pid = _scope_from(_ev)
            out.append(
                Skill(
                    session_id=session_id,
                    user_id=_uid,
                    project_id=_pid,
                    text=(
                        f"Current approach to {intent} questions isn't landing "
                        f"— revise tone or structure."
                    ),
                    kind="gap",
                    confidence=min(0.9, 0.5 + 0.1 * n),
                    evidence_episode_ids=[ep.id for ep in _ev],
                )
            )
        if len(out) >= max_skills:
            return out

    return out
