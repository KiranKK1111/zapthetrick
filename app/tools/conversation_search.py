"""conversation_search tool (vNext §8.4, Stage 7 Component D).

A thin registry wrapper over `app.memory.conversation_search`: loads the
conversation's past turns (and, when present, compaction digests) from the store
and runs the hybrid BM25+dense, reranked, CITED search so the loop can reach back
into history that has scrolled out of the live window.

Fail-open: no store / an error → an empty result (never a crash). The heavy
embedder + reranker are the injected seams inside the memory module (real models
on-pod); here they're left None so the BM25 lexical floor answers on the dev box.
Registered only behaviourally — the loop's tool allow-list is gated by
`tool_loop.conversation_search` (default OFF), so the tool is hidden until enabled.
"""
from __future__ import annotations

from typing import Any

from app.memory import conversation_search as _cs
from app.tools.registry import Tool, register

INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": ("What to find in the earlier conversation — a topic, "
                            "decision, fact, name, or file mentioned before."),
        },
        "conversation_id": {
            "type": "string",
            "description": "ID of the conversation to search within.",
        },
        "top_k": {
            "type": "integer",
            "description": "Max results (default 5).",
        },
    },
    "required": ["query", "conversation_id"],
}


async def _load_items(conversation_id: str, session) -> "list[_cs.SearchItem]":
    """Load past turns as SearchItems from the store. Fail-open → []."""
    try:
        from sqlalchemy import select
        from storage.models import Message
        rows = (
            await session.execute(
                select(Message)
                .where(Message.session_id == conversation_id)
                .order_by(Message.created_at)
            )
        ).scalars().all()
        items: list[_cs.SearchItem] = []
        for m in rows:
            text = (getattr(m, "content", "") or "").strip()
            if text:
                items.append(_cs.SearchItem(
                    id=str(getattr(m, "id", "")), text=text,
                    source=getattr(m, "role", "turn") or "turn"))
        return items
    except Exception:  # noqa: BLE001 — no store / dev box → nothing to search
        return []


async def conversation_search(*, query: str, conversation_id: str,
                              session=None, top_k: int = 5,
                              items=None) -> dict[str, Any]:
    """Search earlier conversation turns/digests → cited hits. `items` may be
    supplied directly (tests / an in-memory caller); otherwise they're loaded
    from the store via `session`. Never raises."""
    try:
        pool = items if items is not None else await _load_items(
            conversation_id, session)
        hits = _cs.search(query, pool, top_k=int(top_k or 5))
        return {"query": query,
                "hits": [h.to_dict() for h in hits],
                "citations": _cs.format_citations(hits)}
    except Exception:  # noqa: BLE001
        return {"query": query, "hits": [], "citations": "No earlier matches."}


# Register at import time (hidden until the loop's allow-list includes it, which
# is gated by `tool_loop.conversation_search`).
register(
    Tool(
        name="conversation_search",
        description=(
            "Search EARLIER in this conversation — turns and compaction digests "
            "that have scrolled out of the recent context window. Use when the "
            "user refers to something discussed before that you can't see. "
            "Returns cited snippets with their source."
        ),
        input_schema=INPUT_SCHEMA,
        handler=conversation_search,
    )
)

__all__ = ["conversation_search", "INPUT_SCHEMA"]
