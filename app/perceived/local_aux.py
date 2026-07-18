"""Local auxiliary models facade (perceived-speed R10, R11).

Cheap helper tasks (embeddings, reranking, titles, summaries, intent) should run
LOCALLY so they add no network latency to a request, reserving remote reasoning
models for the final answer (R10.3). Most of this already exists locally —
`app/rag/embedder.py` (sentence-transformers, R11) and `app/rag/rerank.py`
(cross-encoder, R10) — so this is a thin facade with availability gating + a
remote fallback (R10.2/R11.2), not a new model stack.

Local + remote embeddings share the configured model, so vectors stay
vector-store compatible regardless of source (R11.3).
"""
from __future__ import annotations

import importlib.util
import logging
from typing import Callable

log = logging.getLogger(__name__)


def embeddings_available() -> bool:
    """True when the local embedding model can be used (without loading it)."""
    try:
        return importlib.util.find_spec("sentence_transformers") is not None
    except Exception:  # noqa: BLE001
        return False


def embed(texts: list[str], *, remote: Callable[[list[str]], list] | None = None):
    """Local embeddings (R11.1); on local failure fall back to `remote` if given
    (R11.2). Output is vector-store compatible either way (R11.3)."""
    try:
        from app.rag import embedder
        return embedder.embed(texts)
    except Exception as exc:  # noqa: BLE001
        if remote is not None:
            log.info("local embeddings unavailable (%s) — remote fallback", exc)
            return remote(texts)
        raise


def rerank_available() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.reranker, "enabled", False))
    except Exception:  # noqa: BLE001
        return False


def rerank(query: str, hits: list[dict], top_k: int = 10) -> list[dict]:
    """Local cross-encoder rerank (R10.1); falls back to input order internally."""
    try:
        from app.rag.rerank import rerank_hits
        return rerank_hits(query, hits, top_k)
    except Exception:  # noqa: BLE001 — rerank is best-effort
        return list(hits)[:top_k]


def local_title(text: str, max_words: int = 6) -> str:
    """Deterministic conversation title — no model call (R10.3)."""
    words = (text or "").strip().split()
    title = " ".join(words[:max_words])[:80]
    return title or "New conversation"


def local_summary(text: str, max_chars: int = 600) -> str:
    """Extractive truncation summary — no model call (R10.3)."""
    t = " ".join((text or "").split())
    if len(t) <= max_chars:
        return t
    return t[:max_chars].rsplit(" ", 1)[0] + "…"


def local_intent(text: str) -> str:
    """Local intent topic via the deterministic predictor — no model call."""
    from app.perceived.prefetch import IntentPredictor
    return IntentPredictor().predict(text).topic


__all__ = [
    "embeddings_available", "embed", "rerank_available", "rerank",
    "local_title", "local_summary", "local_intent",
]
