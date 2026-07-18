"""
Resume-lookup tool: hybrid RAG retrieval over an uploaded resume.

The orchestrator invokes this when the question depends on specific
resume facts ("tell me more about that Kafka project"). Returns a list
of relevant chunks plus their section labels, which the orchestrator
splices into the answer prompt.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.rag import retriever
from app.tools.registry import Tool, register


INPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "A natural-language query about the candidate's resume.",
        },
        "resume_id": {
            "type": "string",
            "description": "ID of the resume to search within.",
        },
        "section": {
            "type": "string",
            "description": "Optional section filter: experience | skills | projects | education | summary.",
        },
    },
    "required": ["query", "resume_id"],
}


async def lookup(
    *,
    query: str,
    resume_id: str,
    session: AsyncSession,
    section: str | None = None,
) -> list[dict[str, Any]]:
    """Run hybrid RAG and return the top hits as plain dicts."""
    hits = await retriever.retrieve(
        query,
        resume_id=resume_id,
        session=session,
        section=section,
    )
    return [
        {
            "text": h.text,
            "section": h.section,
            "position": h.position,
            "score": h.score,
        }
        for h in hits
    ]


# Register at import time.
register(
    Tool(
        name="resume_lookup",
        description=(
            "Search the candidate's uploaded resume for chunks relevant to a "
            "specific topic, technology, or project. Use this whenever the "
            "question asks about specific experience or skills."
        ),
        input_schema=INPUT_SCHEMA,
        handler=lookup,
    )
)
