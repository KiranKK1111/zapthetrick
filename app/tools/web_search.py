"""
Free web search tool via DuckDuckGo.

Used by the orchestrator when a question references something the
candidate's resume can't answer (e.g. "what is GraphQL Federation v2?").
Returns a short list of {title, url, snippet} dicts.
"""
from __future__ import annotations

from typing import Any

from app.core.config_loader import cfg
from app.tools.registry import Tool, register


class WebSearchError(RuntimeError):
    pass


INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "Search query."},
        "max_results": {
            "type": "integer",
            "description": "Maximum number of results (default from config).",
        },
    },
    "required": ["query"],
}


async def search(
    *, query: str, max_results: int | None = None
) -> list[dict[str, str]]:
    """Run a DuckDuckGo search and return concise hits.

    Lazy-imports so the app boots even when duckduckgo-search isn't
    installed; only fails when the tool is actually used.
    """
    k = max_results or cfg.web_search.max_results
    if cfg.web_search.provider != "duckduckgo":
        raise WebSearchError(
            f"Web search provider '{cfg.web_search.provider}' is not implemented."
        )
    try:
        from duckduckgo_search import DDGS
    except ImportError as exc:
        raise WebSearchError(
            "duckduckgo-search is not installed. Run: pip install duckduckgo-search"
        ) from exc

    out: list[dict[str, str]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=k):
            out.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
    return out


register(
    Tool(
        name="web_search",
        description=(
            "Search the public web with DuckDuckGo. Use only when the "
            "question is about something outside the candidate's resume "
            "(definitions, news, technology basics). Returns up to "
            "max_results entries with title/url/snippet."
        ),
        input_schema=INPUT_SCHEMA,
        handler=search,
    )
)
