"""Detect when a new turn continues a previous session.

Looks at the active session's first user turn and compares against
recent prior sessions for the same user. If the keyword overlap is
high — or the new turn is short and obviously a follow-up ("what
about Kafka instead?") — we emit a [Continuation] suggestion that
the UI can surface as a "Continue from <session>?" chip.

Heuristic-only:
  1. Token overlap (Jaccard) between the new turn and the topic
     keywords of recent sessions.
  2. Pronoun-/anaphora-heavy short turns ("what about it") bump
     the confidence — these almost certainly need context.

The auto-link writer (called from the chat route) stores any
detection above `auto_link_threshold` as a `continues` edge in
the link graph. Below that threshold, the UI shows a suggest-
confirm chip rather than auto-linking.
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from sqlalchemy import text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from .topic_threads import _label  # reuse the same labeller


@dataclass
class Continuation:
    candidate_session: str
    confidence: float
    matched_keywords: list[str]
    rationale: str


_PRONOUN_RE = re.compile(r"\b(it|that|this|those|these|they|them)\b", re.I)
_FOLLOWUP_HINT_RE = re.compile(
    r"^(?:what about|and|also|but|how about|instead|why not|same for)\b", re.I
)


async def detect_continuation(
    session: AsyncSession,
    *,
    user_id: str | uuid.UUID | None,
    new_turn_text: str,
    limit_recent: int = 15,
    auto_link_threshold: float = 0.55,
) -> list[Continuation]:
    """Return ranked candidate prior sessions this turn continues.

    Empty list when no recent session looks like a parent. The
    threshold is exposed so the caller can decide whether to
    auto-link or prompt.
    """
    if not new_turn_text or not new_turn_text.strip():
        return []

    # Pull recent sessions for this user with their topic labels.
    # No user_id → no filter (matches the resume list path).
    where = ""
    params: dict[str, object] = {"lim": limit_recent}
    if user_id is not None:
        where = "WHERE s.user_id = :uid"
        params["uid"] = str(user_id)

    # Hand-rolled SQL — keeps the query short and works whether
    # `session_topics` exists yet (LEFT JOIN handles the empty case).
    rows = await session.execute(
        sa_text(
            f"""
            SELECT s.id, s.title, st.topic, st.keywords
            FROM sessions s
            LEFT JOIN session_topics st ON st.session_id = s.id
            {where}
            ORDER BY s.updated_at DESC
            LIMIT :lim
            """
        ),
        params,
    )

    new_tokens = set(_tokens(new_turn_text))
    short_followup = bool(_FOLLOWUP_HINT_RE.match(new_turn_text.strip())) or (
        len(new_turn_text.split()) <= 8 and bool(_PRONOUN_RE.search(new_turn_text))
    )

    candidates: list[Continuation] = []
    for row in rows.all():
        sid, title, topic, kws = row
        kws = list(kws or [])
        if not kws and topic:
            _, kws = _label(topic)
        old_tokens = set(t.lower() for t in (kws + ([] if not title else _tokens(title))))
        overlap = (new_tokens & old_tokens)
        union = (new_tokens | old_tokens) or {""}
        jaccard = len(overlap) / max(len(union), 1)
        if jaccard < 0.05 and not short_followup:
            continue
        confidence = min(1.0, jaccard + (0.25 if short_followup else 0.0))
        candidates.append(
            Continuation(
                candidate_session=str(sid),
                confidence=confidence,
                matched_keywords=sorted(overlap)[:8],
                rationale=(
                    "short follow-up phrasing"
                    if short_followup and jaccard < 0.05
                    else f"keyword overlap {jaccard:.2f}"
                ),
            )
        )

    candidates.sort(key=lambda c: -c.confidence)
    return candidates


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in re.findall(r"\w+", text or "") if len(w) > 2]


__all__ = ["detect_continuation", "Continuation"]
