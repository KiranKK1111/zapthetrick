"""Graph-aware follow-up suggestion sources (Architecture.md §6).

Blends three sources into the envelope's `suggestions[]`, each tagged with its
provenance so the UI can style/label it:
  • profile         — intent-styled next steps (the LLM Suggester);
  • memory_graph    — cross-session threads recalled from user memory;
  • knowledge_graph — related concepts from the (content/code) knowledge graph.

All functions are PURE (operate on plain strings passed in — no DB/graph I/O
here, so they are trivially testable and cannot poison the request transaction)
and FAIL-OPEN (empty/absent input → no suggestions from that source). The KG
source produces nothing until the content KG is populated (roadmap #5); the
memory source is deliberately conservative until open-thread detection lands
(roadmap #6).
"""
from __future__ import annotations

import re
from typing import Callable

IntentOf = Callable[[str], str | None]


def _trim(text: str, cap: int = 80) -> str:
    t = " ".join((text or "").split())
    return t if len(t) <= cap else t[: cap - 1].rstrip() + "…"


def from_memory(contents, *, limit: int = 1, intent_of: IntentOf | None = None):
    """Cross-session suggestions from recalled memory contents (strings).

    Conservative: only substantial items (a real thread, not a one-word
    preference), capped. Tagged ``source="memory_graph"``.
    """
    out: list[dict] = []
    for c in contents or []:
        t = " ".join((c or "").split())
        if not (15 <= len(t) <= 200):          # skip trivial / huge blobs
            continue
        s = {"text": f"Revisit: {_trim(t)}", "source": "memory_graph"}
        if intent_of is not None:
            h = intent_of(t)
            if h:
                s["intent_hint"] = h
        out.append(s)
        if len(out) >= limit:
            break
    return out


_STOP = frozenset({
    "the", "a", "an", "and", "or", "but", "for", "this", "that", "these",
    "those", "with", "from", "into", "your", "you", "can", "could", "would",
    "get", "give", "make", "want", "need", "please", "how", "what", "why",
})


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) > 2 and w not in _STOP}


# Meta/housekeeping asks make poor "revisit" threads — they're one-off actions
# ("get me a word document", "zip the project", "download this"), not knowledge
# threads worth resuming. Suppress them as Revisit suggestions.
_META_REVISIT_RE = re.compile(
    r"\b(document|\.?pdf|docx?|word\s+doc|excel|xlsx|csv|pptx|powerpoint|"
    r"zip|archive|download|export|attachment|file)\b",
    re.IGNORECASE,
)


def from_episodes(episodes, *, current_question: str = "", limit: int = 1,
                  intent_of: IntentOf | None = None):
    """Cross-session "open thread" suggestions from prior EPISODES: a related
    prior question the user could resume — "Revisit: …". Tagged
    ``source="memory_graph"``.

    The caller passes episodes ALREADY ranked by semantic similarity to this
    turn (``search_episodes_similar``), so relevance is established upstream —
    here we only (a) skip a near-duplicate of the current turn (that's this
    thread, not an open one) and (b) suppress meta/housekeeping asks
    (doc/zip/download), which were surfacing as junk Revisit chips. Content
    relevance is NOT re-filtered by token overlap: that would wrongly reject a
    semantically-similar thread worded differently ("auth token" vs "JWT").
    """
    cur = _tokens(current_question)
    out: list[dict] = []
    seen: set[str] = set()
    for ep in episodes or []:
        q = (ep.get("question") if isinstance(ep, dict)
             else getattr(ep, "question", "")) or ""
        q = " ".join(q.split())
        if len(q) < 8:
            continue
        if _META_REVISIT_RE.search(q):
            continue
        qt = _tokens(q)
        # Not the current thread: skip a near-duplicate of this turn's question.
        if cur and qt and len(cur & qt) / max(len(cur | qt), 1) >= 0.6:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        s = {"text": f"Revisit: {_trim(q, 90)}", "source": "memory_graph"}
        if intent_of is not None:
            h = intent_of(q)
            if h:
                s["intent_hint"] = h
        out.append(s)
        if len(out) >= limit:
            break
    return out


def from_kg(entities, *, limit: int = 2, intent_of: IntentOf | None = None):
    """Related-concept suggestions from knowledge-graph neighbor entity names.

    Tagged ``source="knowledge_graph"``. Empty input (scaffold KG) → [].
    """
    out: list[dict] = []
    seen: set[str] = set()
    for e in entities or []:
        name = " ".join((e or "").split())
        key = name.lower()
        if not name or key in seen:
            continue
        seen.add(key)
        s = {"text": f"How does this relate to {name}?",
             "source": "knowledge_graph", "intent_hint": "knowledge"}
        if intent_of is not None:
            h = intent_of(s["text"])
            if h:
                s["intent_hint"] = h
        out.append(s)
        if len(out) >= limit:
            break
    return out


def blend(*, profile=None, memory=None, kg=None, limit: int = 3):
    """Combine sources (profile first, then memory, then KG), de-dupe by
    normalized text, and cap to `limit`. Returns envelope suggestion objects."""
    out: list[dict] = []
    seen: set[str] = set()
    for group in (profile or [], memory or [], kg or []):
        for s in group:
            if not isinstance(s, dict):
                continue
            key = " ".join(str(s.get("text", "")).split()).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(s)
            if len(out) >= limit:
                return out
    return out


__all__ = ["from_memory", "from_episodes", "from_kg", "blend"]
