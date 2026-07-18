"""Solve-history repository.

Persists each Solve-screen click as a row so the history drawer can
reload a past problem statement + response. Matches the shape of the
Chat tab's `conversations` list — title (short identifier), full body,
and a created-at timestamp drive the UI.
"""
from __future__ import annotations

import re
import uuid

from sqlalchemy import select

from ..models import SolveSession
from .base import Repo


# Section markers emitted by the OCR / structured-output prompt
# (`=== TITLE ===`, `## Problem`, `**Constraints**`, etc.). If the
# first non-empty line is one of these we skip it and use whatever
# follows — otherwise the persisted title literally reads
# "=== TITLE ===".
_SECTION_MARKER_RE = re.compile(
    r"""
    ^\s*
    (?:[#=*\-]{1,6}\s*)?                 # leading ##, ===, **, etc.
    (?:
        TITLE
      | FUNCTION\s*SIGNATURE
      | PROBLEM(?:\s*STATEMENT)?
      | QUESTION
      | EXAMPLES?
      | CONSTRAINTS?
      | INPUT
      | OUTPUT
      | NOTES?
      | APPROACH
      | SOLUTION
    )
    (?:\s*[#=*\-]{1,6})?                 # trailing markers
    \s*[:\-]?\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


# A title placeholder we consider "auto-derived from text we don't
# trust" — empty, fallback string, or one of the section markers
# above. The async auto_title task replaces these via LLM.
_PLACEHOLDER_TITLES = {
    "",
    "untitled solve",
    "(no problem statement captured)",
}


def is_placeholder_title(title: str) -> bool:
    """True if the title looks like a placeholder that should be
    overwritten by the LLM auto-title task."""
    norm = (title or "").strip().lower()
    if norm in _PLACEHOLDER_TITLES:
        return True
    if _SECTION_MARKER_RE.match(title or ""):
        return True
    # Long descriptive sentences ("Given a sorted array of integers,
    # return the indices...") — usually the first sentence of the
    # problem rather than a real title.
    return len(title or "") >= 40


def _derive_title(description: str, fallback: str = "Untitled solve") -> str:
    """Walk the description looking for the actual title text.

    Rules, in order:
      1. Skip leading blank lines.
      2. If the first non-empty line is a section marker
         (`=== TITLE ===`, `## Problem`, etc.), consume it and try
         the next non-empty line.
      3. Strip "Problem:" / "Question:" / "Title:" inline prefixes.
      4. Truncate to 120 chars. Empty → fallback.
    """
    if not description:
        return fallback

    lines = description.splitlines()
    skipped_marker = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Marker line — skip it and look at the next non-empty line
        # for the actual title content. We tolerate up to one marker
        # in case the OCR output has a `=== TITLE ===` followed by
        # blank lines.
        if _SECTION_MARKER_RE.match(stripped):
            if skipped_marker:
                # Two markers in a row — bail before we read the body.
                break
            skipped_marker = True
            continue
        # Strip common inline prefixes that bloat the title.
        for prefix in ("Problem:", "Question:", "Title:"):
            if stripped.lower().startswith(prefix.lower()):
                stripped = stripped[len(prefix):].strip()
                break
        if stripped:
            return stripped[:120]
    return fallback


class SolveRepo(Repo):
    async def create(
        self,
        *,
        description: str,
        response: str,
        title: str | None = None,
        user_id: uuid.UUID | None = None,
        language: str | None = None,
        source: str = "text",
        image_path: str | None = None,
        vision_model: str | None = None,
        code_model: str | None = None,
        latency_ms: int = 0,
    ) -> SolveSession:
        row = SolveSession(
            user_id=user_id,
            title=title or _derive_title(description),
            description=description,
            response=response,
            language=language,
            source=source,
            image_path=image_path,
            vision_model=vision_model,
            code_model=code_model,
            latency_ms=latency_ms,
        )
        self.session.add(row)
        await self.session.flush()
        return row

    async def get(self, solve_id: uuid.UUID | str) -> SolveSession | None:
        if isinstance(solve_id, str):
            try:
                solve_id = uuid.UUID(solve_id)
            except ValueError:
                return None
        return await self.session.get(SolveSession, solve_id)

    async def list(
        self,
        *,
        user_id: uuid.UUID | None = None,
        limit: int = 200,
    ) -> list[SolveSession]:
        stmt = (
            select(SolveSession)
            .order_by(SolveSession.created_at.desc())
            .limit(limit)
        )
        if user_id is not None:
            stmt = (
                select(SolveSession)
                .where(SolveSession.user_id == user_id)
                .order_by(SolveSession.created_at.desc())
                .limit(limit)
            )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def delete(self, solve_id: uuid.UUID | str) -> bool:
        row = await self.get(solve_id)
        if row is None:
            return False
        await self.session.delete(row)
        await self.session.flush()
        return True
