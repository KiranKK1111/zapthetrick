"""Background auto-titling for new chat sessions.

A new conversation starts with a placeholder title ("New conversation"
/ "New session"). On the first user/assistant turn we kick off a
small-model LLM call that produces a 3–6 word title and writes it back
to the session row. Cheap, async, non-blocking — failures are
silently swallowed (the placeholder remains).

Hooked into [routes_chat.chat_stream] via `asyncio.create_task` after
the assistant message is committed.
"""
from __future__ import annotations

import logging
import re
import uuid

from app.core.config_loader import cfg
from app.core.llm_client import LLMError, llm
from storage.db import get_session_factory
from storage.repos import SessionRepo
from app.core.prompt import fill

log = logging.getLogger(__name__)

_PLACEHOLDER_TITLES = {"new conversation", "new session", "untitled", ""}

_PROMPT = """Summarise the conversation below as a 3–6 word title.
No quotes, no trailing punctuation, no leading "Title:".
Use Title Case.

USER FIRST MESSAGE:
{first_user}

ASSISTANT REPLY (excerpt):
{first_assistant}
"""

async def maybe_title(
    session_id: uuid.UUID | str,
    *,
    current_title: str,
    first_user: str,
    first_assistant: str,
) -> None:
    """Generate + persist a title when the session still has a placeholder.

    Returns silently if:
      - the session already has a real title (user renamed it, or this
        wasn't the first turn)
      - LLM is unreachable
      - the response is unusable (too long, empty, etc.)
    """
    norm = (current_title or "").strip().lower()
    if norm not in _PLACEHOLDER_TITLES and not norm.startswith("new "):
        return

    classifier_model = cfg.llm.classifier_model or cfg.llm.model
    prompt = fill(_PROMPT, 
        first_user=first_user[:1500],
        first_assistant=first_assistant[:1500],
    )
    try:
        raw = await llm.complete(
            [{"role": "user", "content": prompt}],
            model=classifier_model,
            options={"temperature": cfg.temperature.planning,
                     "num_predict": cfg.output_tokens.label},
        )
    except LLMError as exc:
        log.info("auto-title LLM call failed (will keep placeholder): %s", exc)
        return

    title = _clean(raw)
    if not title:
        return

    factory = get_session_factory()
    if factory is None:
        return
    try:
        async with factory() as write_session:
            await SessionRepo(write_session).set_flags(session_id, title=title)
            await write_session.commit()
    except Exception as exc:  # noqa: BLE001
        log.warning("auto-title persist failed: %s", exc)

def _clean(raw: str) -> str:
    """Strip quotes / trailing punctuation / "Title:" prefixes."""
    text = (raw or "").strip().splitlines()[0] if raw else ""
    text = re.sub(r"^(title|chat|conversation)\s*:\s*", "", text, flags=re.I)
    text = text.strip(' "\'`.:;—-')
    # Keep it short — anything longer than 80 chars is almost certainly
    # the model writing prose, not a title.
    if not text or len(text) > 80:
        return ""
    return text

__all__ = ["maybe_title"]
