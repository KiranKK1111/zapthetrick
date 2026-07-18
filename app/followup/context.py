"""Relevance-based context selection (followup-context-engine R8).

`select(state, resolved_turn, history) -> messages` ranks prior turns by lexical
relevance to the resolved follow-up and ALWAYS prepends the ConversationState
summary (R8.2), so a long thread stays on-point instead of resending everything.

Token-budget / latency mechanics are deliberately **deferred to the
`perceived-speed` Context_Budget** (R8.4) — this module only chooses *which*
turns are relevant, not how many tokens fit. Relevance scoring unavailable /
error → recent-history fallback (R8.3 / Property 1).
"""
from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]+")
_STOP = frozenset((
    "the", "a", "an", "and", "or", "to", "of", "in", "on", "for", "is", "it",
    "this", "that", "with", "as", "be", "are", "i", "you", "we", "do", "make",
    "use", "can", "how", "what", "please", "me", "my", "your",
))


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall((text or "").lower())
            if w not in _STOP and len(w) > 1}


def _relevance(query_toks: set[str], msg_text: str) -> float:
    """Jaccard-ish overlap of content tokens (cheap, deterministic)."""
    mt = _tokens(msg_text)
    if not query_toks or not mt:
        return 0.0
    inter = len(query_toks & mt)
    if inter == 0:
        return 0.0
    return inter / float(len(query_toks | mt))


def select(state, resolved_turn: str, history: list[dict], *,
           max_turns: int = 8, recent_floor: int = 4) -> list[dict]:
    """Return the messages to include for a follow-up.

    Always begins with a synthetic ``system`` message carrying the
    ConversationState summary (R8.2), then the most relevant prior turns (by
    relevance to `resolved_turn`), preserving chronological order. The most
    recent `recent_floor` turns are always kept regardless of score so the
    immediate thread is never dropped. Fail-open → recent history.
    """
    try:
        return _select(state, resolved_turn, history, max_turns, recent_floor)
    except Exception:  # noqa: BLE001 — fall back to recent history (R8.3)
        return list(history or [])[-max_turns:]


def _select(state, resolved_turn, history, max_turns, recent_floor):
    history = list(history or [])
    messages: list[dict] = []

    # 1) State summary floor — always included (R8.2).
    summary = ""
    try:
        summary = state.summary() if state is not None else ""
    except Exception:  # noqa: BLE001
        summary = ""
    if summary:
        messages.append({"role": "system",
                         "content": f"Conversation so far:\n{summary}"})

    if not history:
        return messages

    # 2) Always keep the most recent `recent_floor` turns.
    n = len(history)
    floor_start = max(0, n - max(0, recent_floor))
    forced_idx = set(range(floor_start, n))

    # 3) Rank the older turns by relevance to the resolved follow-up.
    qt = _tokens(resolved_turn)
    scored = []
    for i in range(floor_start):
        scored.append((i, _relevance(qt, history[i].get("content", ""))))
    scored.sort(key=lambda x: x[1], reverse=True)

    budget = max(0, max_turns - len(forced_idx))
    chosen = {i for i, s in scored[:budget] if s > 0.0}
    keep = sorted(forced_idx | chosen)        # chronological order preserved
    messages.extend(history[i] for i in keep)
    return messages


__all__ = ["select"]
