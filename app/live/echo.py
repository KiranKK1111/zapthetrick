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



# How much the streaming answer must GROW before its prefix is re-registered.
# Embedding per token would be absurd; per sentence-ish keeps a read-along
# matchable from the first sentence onward at negligible cost.
_PARTIAL_STEP = 140
# Per-session high-water mark of what has already been registered while
# streaming, so growth is measured rather than re-embedded from scratch.
_PARTIAL_AT: "dict[str, int]" = {}


def remember_streaming(sid: str, text: str) -> bool:
    """Register the answer PREFIX while it is still streaming.

    Registering only the FINISHED answer carries two bugs, and both bite in
    exactly the reported scenario — a candidate reading the answer aloud in solo
    mode while it is still arriving:

    1. **Timing.** The read-along starts before the answer completes, so at that
       moment nothing is registered and the echo check has nothing to match
       against. The utterance gets transcribed and answered as a new question —
       the assistant interrupts the candidate to answer its own words back.
    2. **Partial reads.** Someone who reads only the opening of a long answer is
       not very similar to the WHOLE answer, so a single final-text entry can
       score below threshold. A prefix, though, is highly similar to the prefix
       that was read.

    Registering prefixes as they grow fixes both. Returns whether anything was
    recorded — the caller does not care, but it makes the behaviour testable.
    """
    text = (text or "").strip()
    if not sid or len(text) < 12:
        return False
    with _LOCK:
        last = _PARTIAL_AT.get(sid, 0)
        if len(text) - last < _PARTIAL_STEP:
            return False
        _PARTIAL_AT[sid] = len(text)
    remember_answer(sid, text)
    return True


def reset_streaming(sid: str) -> None:
    """Clear the growth mark at the end of a turn, so the next answer registers
    from its own beginning rather than inheriting this one's length."""
    if not sid:
        return
    with _LOCK:
        _PARTIAL_AT.pop(sid, None)


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


def best_match(sid: str, utterance: str) -> "tuple[str, float]":
    """Return (displayed_text, similarity) of the shown answer the candidate is
    most likely reading. Empty text on any miss/failure. Used by delivery
    tracking (§4.14) to align the SPOKEN utterance against the DISPLAYED script."""
    u = (utterance or "").strip()
    if not sid or len(u) < 8:
        return "", 0.0
    with _LOCK:
        items = list(_ANSWERS.get(sid) or ())
    if not items:
        return "", 0.0
    try:
        from app.rag.embedder import embed_one
        uv = embed_one(u[:2000])
    except Exception:  # noqa: BLE001
        return "", 0.0
    if not uv:
        return "", 0.0
    best_text, best = "", 0.0
    for text, vec in items:
        s = 0.0
        for a, b in zip(uv, vec):
            s += a * b
        if s > best:
            best, best_text = s, text
    return best_text, best


def forget_session(sid: str) -> None:
    with _LOCK:
        _ANSWERS.pop(sid, None)
        _PARTIAL_AT.pop(sid, None)
