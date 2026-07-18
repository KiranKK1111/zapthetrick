"""Background auto-titling for solve sessions.

The first-line extractor in [SolveRepo.create] gives every row a
title immediately so the history list renders without a spinner.
That title is often a long descriptive sentence ("Given a sorted
array of integers, return the two indices that sum to a target...").

This module replaces that placeholder with a tight 3–6-word LLM-
generated title in the background, similar to chat auto_title.
Failures are silent — the placeholder title remains, and the row
is still searchable.
"""
from __future__ import annotations

import logging
import re
import uuid

from app.core.config_loader import cfg
from app.core.llm_client import LLMError, llm
from storage.db import get_session_factory
from storage.models import SolveSession
from app.core.prompt import fill

log = logging.getLogger(__name__)

_PROMPT = """Summarise the coding problem below as a 3–6 word title.
No quotes, no trailing punctuation, no "Title:" prefix. Use Title Case.

Examples of good titles:
  - "Two Sum (Sorted)"
  - "LRU Cache"
  - "Word Ladder Shortest Path"
  - "Detect Cycle in Linked List"

PROBLEM:
{description}

RESPONSE FROM MODEL (for hints — do not include in title):
{response}
"""

async def maybe_title(
    solve_id: uuid.UUID | str,
    *,
    description: str,
    response: str,
    current_title: str,
) -> None:
    """Generate + persist a tight title when the current one is just
    a placeholder (long descriptive sentence, section marker like
    `=== TITLE ===`, or the literal `Untitled solve`).

    Returns silently if:
      - the LLM is unreachable
      - the user already renamed the solve to something short
      - the response is empty or unusable
    """
    # One source of truth for what counts as a placeholder lives in
    # SolveRepo — shared so the extractor and this background task
    # can't drift.
    from storage.repos.solve_repo import is_placeholder_title

    if not is_placeholder_title(current_title):
        return

    classifier_model = cfg.llm.classifier_model or cfg.llm.model
    prompt = fill(_PROMPT, 
        description=description[:2000],
        response=response[:1500],
    )
    try:
        raw = await llm.complete(
            [{"role": "user", "content": prompt}],
            model=classifier_model,
            options={"temperature": cfg.temperature.planning,
                     "num_predict": cfg.output_tokens.label},
        )
    except LLMError as exc:
        log.info("solve auto-title LLM call failed (keep placeholder): %s", exc)
        return

    title = _clean(raw)
    if not title:
        return

    factory = get_session_factory()
    if factory is None:
        return
    try:
        async with factory() as write_session:
            row = await write_session.get(
                SolveSession, uuid.UUID(str(solve_id))
            )
            if row is None:
                return
            row.title = title[:120]
            await write_session.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("solve auto-title persist failed: %s", exc)

def _clean(raw: str) -> str:
    """Strip quotes / trailing punctuation / 'Title:' prefixes."""
    text = (raw or "").strip().splitlines()[0] if raw else ""
    text = re.sub(r"^(title|problem|name)\s*:\s*", "", text, flags=re.I)
    text = text.strip(' "\'`.:;—-')
    if not text or len(text) > 80:
        return ""
    return text

__all__ = ["maybe_title"]
