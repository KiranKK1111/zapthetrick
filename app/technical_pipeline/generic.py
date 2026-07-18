"""Generic technical-pipeline fallback.

Used when no specialised domain matches. Falls back to a single
LLM call wrapped in the response-architecture shaping layer so the
output still gets uniform markdown / polish treatment.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from app.core.config_loader import cfg
from app.core.llm_client import LLMError, llm
from app.response_arch import finalize
from app.core.prompt import fill

_PROMPT = """You are a senior interview coach. Answer the question below
clearly and concisely, prefer headings + bullet points where natural.
Question:
{question}
"""

async def run(question: str) -> AsyncIterator[dict]:
    yield {"kind": "stage", "name": "classifier", "data": {"domain": "generic"}}

    try:
        raw = await llm.complete(
            [{"role": "user", "content": fill(_PROMPT, question=question)}],
            model=cfg.llm.code_model or cfg.llm.model,
        )
    except LLMError as exc:
        yield {"kind": "done", "data": {"warning": f"llm failed: {exc}"}}
        return

    shaped = finalize(raw or "", question=question)
    yield {"kind": "markdown", "text": shaped.text}
    yield {"kind": "done", "data": {"shape": shaped.shape.value, "depth": shaped.depth}}
