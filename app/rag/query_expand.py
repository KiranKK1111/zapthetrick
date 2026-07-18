"""Query expansion for retrieval (HyDE).

Vague or short questions retrieve poorly because the query embedding doesn't
look like the documents. HyDE (Hypothetical Document Embeddings) asks the LLM
for a short *hypothetical answer*, then embeds THAT for dense search — it lands
much closer to the real passages. Keyword/BM25 search keeps using the literal
question (so exact terms still match).

Best-effort: any failure falls back to the original question, so retrieval
never depends on the expansion succeeding.
"""
from __future__ import annotations

import logging

from app.core.config_loader import cfg
from app.core.llm_client import LLMError, llm
from app.core.prompt import fill

log = logging.getLogger(__name__)

_HYDE_PROMPT = """Write a short, factual passage (3-5 sentences) that would \
directly answer the question below, as if quoting an authoritative document. \
Do not say "I think" or address the user — just write the passage. If you don't \
know specifics, write a plausible passage using the right terminology.

Question: {question}

Passage:"""

async def hyde_text(question: str) -> str:
    """Return a query string for DENSE retrieval — the question plus a
    hypothetical answer (HyDE). Falls back to the question alone."""
    q = (question or "").strip()
    if len(q) < 8:
        return q
    try:
        raw = await llm.complete(
            [{"role": "user", "content": fill(_HYDE_PROMPT, question=q)}],
            model=(cfg.llm.classifier_model or cfg.llm.model),
            options={"temperature": 0.3, "num_predict": 200},
        )
    except LLMError as exc:
        log.info("HyDE expansion unavailable, using raw query: %s", exc)
        return q
    passage = (raw or "").strip()
    if not passage:
        return q
    # Embed question + hypothetical together so the vector reflects both.
    return f"{q}\n{passage}"[:2000]

__all__ = ["hyde_text"]
