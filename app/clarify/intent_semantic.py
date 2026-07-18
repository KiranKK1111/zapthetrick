"""Semantic intent classification via sentence-transformers (bge-m3).

Understands intent by SEMANTIC SIMILARITY to labeled example phrasings instead
of keyword/regex matching — so paraphrases the regex misses ("walk me through
this code" ≈ "explain this code") classify correctly, and *extending* coverage
means adding an example sentence (data), not writing a new regex rule.

Design notes:
  • Reuses the already-loaded embedder (`app.rag.embedder`, model = bge-m3 from
    config) — no new model, no new dependency, no extra memory.
  • Vectors are unit-normalized by the embedder, so cosine similarity is a plain
    dot product.
  • FAIL-OPEN: any embedder error (e.g. sentence-transformers/model unavailable)
    → returns None, and the caller falls back to the deterministic regex path.
  • Exemplars are DATA (editable below / overridable via config later), not rules.
  • Pure/injectable: `classify` accepts an `embed_fn` so it is unit-testable
    without loading the real model.
"""
from __future__ import annotations

import logging
from typing import Callable, Sequence

from app.clarify.intent_pipeline import (
    INTENT_ARCHIVE,
    INTENT_CHITCHAT,
    INTENT_CODE_GEN,
    INTENT_COMPARISON,
    INTENT_DEBUGGING,
    INTENT_DESIGN,
    INTENT_DOCS,
    INTENT_KNOWLEDGE,
    INTENT_PROJECT_BUILD,
    INTENT_TEST_GEN,
)

log = logging.getLogger(__name__)

EmbedFn = Callable[[Sequence[str]], list[list[float]]]

# Seed exemplars — a few natural phrasings per intent. The embedding model
# GENERALIZES from these: "explain this program" matches the KNOWLEDGE examples
# even though no exemplar is an exact match. Edit/extend freely — this is data,
# not code. (INTENT_UNKNOWN is intentionally absent: it's the "no confident
# match" fallback, decided by the similarity threshold, not by an exemplar.)
EXEMPLARS: dict[str, list[str]] = {
    INTENT_CHITCHAT: [
        "hey there", "hello, how are you", "thanks so much", "good morning",
        "that was helpful, cheers", "ok cool",
    ],
    INTENT_KNOWLEDGE: [
        "explain this code", "what does this function do",
        "walk me through this program", "how does this algorithm work",
        "help me understand this snippet", "why does this work the way it does",
        "describe what happens here", "what is a hash map",
    ],
    INTENT_COMPARISON: [
        "compare Postgres and MySQL", "React vs Vue, which is better",
        "what's the difference between a list and a tuple",
        "pros and cons of microservices vs monolith",
        "when should I use a queue instead of a stack",
        "how is a mutex different from a semaphore",
        "how does X differ from Y",
    ],
    INTENT_DEBUGGING: [
        "why is this throwing an error", "fix this stack trace",
        "my code isn't working, what's wrong", "this function crashes on empty input",
        "debug this null pointer exception", "why does this test keep failing",
        "my function blows up on an empty list", "it breaks on this edge case",
    ],
    INTENT_TEST_GEN: [
        "write unit tests for this function", "add pytest tests for this module",
        "generate test cases for this API", "cover this with jest tests",
        "add test coverage for this handler", "increase the test coverage here",
    ],
    INTENT_DOCS: [
        "write documentation for this module", "generate a README for the project",
        "add docstrings to these functions", "document this API",
        "write inline comments explaining this", "document the codebase",
        "give me a PDF of the design", "turn this into a Word document",
        "export this as a document", "make a report out of this",
    ],
    INTENT_DESIGN: [
        "design the architecture for a chat app", "propose a database schema for orders",
        "high-level system design for a URL shortener", "how should I model this data",
        "draw an ER diagram for this domain", "design a scalable system for payments",
        "what's a good architecture for this",
    ],
    INTENT_CODE_GEN: [
        "write a function to reverse a string", "implement binary search in Python",
        "give me a snippet to parse JSON", "code an API endpoint that returns users",
        "implement a rate limiter", "write a regex to validate emails",
        "can you give me a program for fibonacci",
        "write a program that finds the third non-repeated character",
        "give me the code for a linked list", "how do I sort a list in place",
        # API-building phrasings — without these, "write a login api" lands on
        # the DOCS exemplar "document this API" (semantic-intent bench 2026-07).
        "write a login api", "create an api endpoint for user signup",
        "write an api for managing orders",
        "show me how to write a binary search tree", "solve this coding problem",
        "give me the source code for a stack", "write me a script to rename files",
    ],
    INTENT_PROJECT_BUILD: [
        "build me a todo web app", "create a REST API service from scratch",
        "scaffold a Flutter mobile app", "make a full-stack dashboard",
        "set up a new microservice with auth", "just give me a whole project",
        "generate an entire application for me", "build a complete solar UI project",
    ],
    INTENT_ARCHIVE: [
        "zip the whole project", "compress all the files into an archive",
        "give me a downloadable archive of the codebase",
        "package the project as a zip", "get me the archive of this project",
        "i want the archive", "can I get an archive of everything",
        "export the project", "bundle everything up for me",
        "download the whole project", "give me the project archive",
        "send me the whole thing as a single file", "make an archive of the code",
        "give me everything in one file to download", "archive it",
        "package it all up", "get me a compressed file of the project",
    ],
}


def _default_embed(texts: Sequence[str]) -> list[list[float]]:
    from app.rag.embedder import embed
    return embed(list(texts))


# Cache of the exemplar matrix for the REAL embedder: (learned_version, labels,
# matrix). Rebuilt when the learned-exemplar version changes. Injected embedders
# (tests) bypass the cache.
_CACHE: tuple[int, list[str], object] | None = None


def _merged_exemplars() -> dict[str, list[str]]:
    """Seed exemplars + learned POSITIVE exemplars (#12). Learning off / no data
    → exactly the seed set. Never mutates EXEMPLARS."""
    merged = {k: list(v) for k, v in EXEMPLARS.items()}
    try:
        from app.clarify import learned_exemplars as _le
        for intent, phrases in _le.positives().items():
            merged.setdefault(intent, [])
            merged[intent] = merged[intent] + list(phrases)
    except Exception:  # noqa: BLE001 — learned exemplars are best-effort
        pass
    return merged


def _build_matrix(embed_fn: EmbedFn):
    import numpy as np
    labels: list[str] = []
    flat: list[str] = []
    for intent, phrases in _merged_exemplars().items():
        for p in phrases:
            labels.append(intent)
            flat.append(p)
    vecs = np.asarray(embed_fn(flat), dtype="float32")
    return labels, vecs


def _learned_version() -> int:
    try:
        from app.clarify import learned_exemplars as _le
        return _le.version() if _le.enabled() else 0
    except Exception:  # noqa: BLE001
        return 0


def _matrix(embed_fn: EmbedFn | None):
    """Return (labels, matrix). Cached for the real embedder (keyed on the
    learned-exemplar version so new feedback rebuilds it); fresh for injected."""
    global _CACHE
    if embed_fn is not None:
        return _build_matrix(embed_fn)
    ver = _learned_version()
    if _CACHE is None or _CACHE[0] != ver:
        labels, mat = _build_matrix(_default_embed)
        _CACHE = (ver, labels, mat)
    return _CACHE[1], _CACHE[2]


def reset_cache() -> None:
    """Drop the cached exemplar matrix (e.g. after the embedding model changes
    or learned exemplars are updated)."""
    global _CACHE
    _CACHE = None


def _negative_penalty(intent: str, v, embed_fn: EmbedFn) -> float:
    """Penalty to subtract from the winning intent's score when the query
    closely resembles a learned NEGATIVE exemplar for that intent (#12). 0 when
    learning is off or there's no strong negative match."""
    try:
        from app.clarify import learned_exemplars as _le
        negs = _le.negatives().get(intent)
        if not negs:
            return 0.0
        import numpy as np
        from app.core.config_loader import cfg
        weight = float(getattr(cfg.semantic_intent, "negative_penalty", 0.15))
        nmat = np.asarray(embed_fn(list(negs)), dtype="float32")
        nsim = float(np.max(nmat @ v))
        return weight * nsim if nsim >= 0.6 else 0.0
    except Exception:  # noqa: BLE001
        return 0.0


def classify(text: str, *, embed_fn: EmbedFn | None = None) -> tuple[str, float] | None:
    """Return ``(intent, similarity)`` for the nearest exemplar, or ``None`` on
    any embedder failure (caller then falls back to the regex classifier).

    `similarity` is cosine in [-1, 1] (vectors are unit-normalized). The caller
    applies its own confidence threshold; this function never decides UNKNOWN.
    """
    t = (text or "").strip()
    if not t:
        return None
    # Cold-start protection: NEVER load the model synchronously inside a
    # request — that freezes the event loop (no SSE keepalives) until the
    # client watchdog kills the connection. While the model warms in the
    # background, fail open to the regex classifier.
    if embed_fn is None:
        try:
            from app.rag import embedder as _emb
            if not _emb.is_ready():
                _emb.ensure_loading_in_background()
                return None
        except Exception:  # noqa: BLE001
            return None
    try:
        import numpy as np
        labels, mat = _matrix(embed_fn)
        ef = embed_fn or _default_embed
        v = np.asarray(ef([t])[0], dtype="float32")
        sims = mat @ v                      # cosine (unit-normalized) per exemplar
        idx = int(np.argmax(sims))
        intent, sim = labels[idx], float(sims[idx])
        # #12: demote when the query strongly matches a learned NEGATIVE exemplar
        # for this intent (a phrasing the user marked as wrongly classified).
        pen = _negative_penalty(intent, v, ef)
        return intent, sim - pen
    except Exception as exc:  # noqa: BLE001 — fail-open to the regex classifier
        log.info("semantic intent unavailable (%s); falling back to regex", exc)
        return None


__all__ = ["classify", "reset_cache", "EXEMPLARS"]
