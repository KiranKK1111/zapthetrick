"""Candidate self-answer echo detection.

A dual-source live session hears BOTH the interviewer and the candidate. When
the candidate answers — often by reading or paraphrasing the answer the copilot
just put on screen — that speech must NOT be treated as a new interviewer
question. This module remembers the recent answers shown and, using the
already-loaded embedder, recognizes when an utterance is (semantically) the
candidate echoing one of them — paraphrase-tolerant, so it catches the intent
even when the words differ from what's on screen.

In-process, per session, fully fail-open: any error => "not an echo", so a real
question is never dropped by a hiccup here.
"""
from __future__ import annotations

import threading
from collections import defaultdict, deque

_LOCK = threading.Lock()
# sid -> deque[(text, normalized_vector)] of the most recent answers shown.
_ANSWERS: "dict[str, deque]" = defaultdict(lambda: deque(maxlen=6))


def remember_answer(sid: str, text: str) -> None:
    """Record an answer that was shown, so a later candidate echo of it can be
    recognized. No-op on short/empty text or any embed failure."""
    text = (text or "").strip()
    if not sid or len(text) < 12:
        return
    try:
        from app.rag.embedder import embed_one
        vec = embed_one(text[:2000])
    except Exception:  # noqa: BLE001 — never break the live path
        return
    if not vec:
        return
    with _LOCK:
        _ANSWERS[sid].append((text, vec))


def is_candidate_echo(
    sid: str, utterance: str, threshold: float = 0.72
) -> "tuple[bool, float]":
    """Return (is_echo, best_similarity). True when `utterance` is semantically
    the candidate speaking back one of the recent answers shown. The embedder
    normalizes vectors, so cosine similarity is a plain dot product."""
    u = (utterance or "").strip()
    if not sid or len(u) < 8:
        return False, 0.0
    with _LOCK:
        items = list(_ANSWERS.get(sid) or ())
    if not items:
        return False, 0.0
    try:
        from app.rag.embedder import embed_one
        uv = embed_one(u[:2000])
    except Exception:  # noqa: BLE001
        return False, 0.0
    if not uv:
        return False, 0.0
    best = 0.0
    for _text, vec in items:
        s = 0.0
        for a, b in zip(uv, vec):
            s += a * b
        if s > best:
            best = s
    return best >= threshold, best


def forget_session(sid: str) -> None:
    with _LOCK:
        _ANSWERS.pop(sid, None)
