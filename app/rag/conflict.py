"""Cross-source contradiction / truth maintenance for the CHAT turn (Phase 3 #11).

The live pipeline has `live/contradiction.py`; the chat side had no cross-source
conflict detection. When the turn injects *recalled memory* ("decision: use
Postgres") but the *answer* (or a retrieved chunk / KG relation) asserts the
opposite ("we'll use MySQL"), that contradiction should be surfaced — not
silently averaged away.

This is a deterministic, conservative detector: it only flags conflicts it can
justify (a negation flip on the same statement, or two different committed values
for the same subject/key). It never guesses. False negatives are preferred over
false positives — a spurious "contradiction" banner is worse than a missed one.

Consumed in `routes_agents.py`: memory items vs the final answer + KG relations →
surfaced in the envelope `grounding.conflicts` and the turn trace.

Fail-open: any error → no conflicts.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_NEG = re.compile(r"\b(not|no|never|without|avoid|don'?t|doesn'?t|isn'?t|"
                  r"aren'?t|won'?t|can'?t|shouldn'?t|drop|remove|stop using)\b",
                  re.IGNORECASE)
_WORD = re.compile(r"[a-z0-9+.#]+", re.IGNORECASE)
_STOP = {"the", "a", "an", "to", "of", "for", "and", "or", "is", "are", "be",
         "we", "i", "it", "this", "that", "will", "should", "use", "using",
         "with", "on", "in", "our", "your", "let", "lets", "let's", "please"}

# A committed statement of the form "<key>: <value>" (how memory objects render
# decisions/preferences: "decision: use postgres", "database: mysql").
_KV = re.compile(r"^\s*([a-z0-9 _/-]{2,40}?)\s*[:=]\s*(.+?)\s*$", re.IGNORECASE)


@dataclass
class Conflict:
    subject: str
    a: str            # the memory / first source statement
    b: str            # the contradicting statement
    kind: str         # "negation" | "value"

    def as_dict(self) -> dict:
        return {"subject": self.subject, "a": self.a[:160], "b": self.b[:160],
                "kind": self.kind}


def _content_words(text: str) -> set[str]:
    # `_WORD` keeps `.`/`+`/`#` so tech terms survive (c++, c#, .net); strip a
    # trailing/leading dot so a sentence-final "caching." still matches "caching".
    out: set[str] = set()
    for w in _WORD.findall(text or ""):
        w = w.lower().strip(".")
        if len(w) > 1 and w not in _STOP:
            out.add(w)
    return out


def _polarity(text: str) -> bool:
    """True when the statement is NEGATED (odd number of negation cues)."""
    return len(_NEG.findall(text or "")) % 2 == 1


def _kv(text: str) -> tuple[str, str] | None:
    m = _KV.match((text or "").strip())
    if not m:
        return None
    key = re.sub(r"\s+", " ", m.group(1).strip().lower())
    val = m.group(2).strip().lower()
    # collapse "decision:"/"preference:" prefixes to the real subject noun
    key = re.sub(r"^(decision|preference|entity|choice|note)\b[: ]*", "", key).strip()
    return (key or m.group(1).strip().lower(), val)


def _value_token(val: str) -> str | None:
    """The salient content token of a short committed value ("use postgres" ->
    "postgres"), or None when it's not a clean single choice."""
    words = [w.lower() for w in _WORD.findall(val or "")
             if w.lower() not in _STOP]
    if not words:
        return None
    return words[-1]           # the trailing noun is the committed choice


def _pair_conflicts(a: str, b: str) -> Conflict | None:
    """Do statements `a` and `b` contradict? Conservative."""
    wa, wb = _content_words(a), _content_words(b)
    overlap = wa & wb
    # Value conflict on the same key ("database: postgres" vs "database: mysql").
    ka, kb = _kv(a), _kv(b)
    if ka and kb and ka[0] == kb[0]:
        va, vb = _value_token(ka[1]), _value_token(kb[1])
        if va and vb and va != vb:
            return Conflict(ka[0], a, b, "value")
    # Negation flip: strong content overlap but opposite polarity.
    if len(overlap) >= 2 and _polarity(a) != _polarity(b):
        subject = " ".join(sorted(overlap)[:3])
        return Conflict(subject, a, b, "negation")
    return None


def detect(memory_statements, other_statements) -> list[Conflict]:
    """Conflicts between committed memory (`memory_statements`) and any other
    source text (answer sentences, KG relations, retrieved snippets).
    Deterministic + fail-open."""
    try:
        mem = [s for s in (memory_statements or []) if (s or "").strip()]
        others = [s for s in (other_statements or []) if (s or "").strip()]
        seen: set[tuple[str, str]] = set()
        out: list[Conflict] = []
        for a in mem:
            for b in others:
                c = _pair_conflicts(a, b)
                if c is None:
                    continue
                key = (c.subject, c.b[:60])
                if key in seen:
                    continue
                seen.add(key)
                out.append(c)
        return out
    except Exception:  # noqa: BLE001
        return []


def split_sentences(text: str, *, limit: int = 40) -> list[str]:
    """Cheap sentence split for scanning an answer for contradictions."""
    try:
        parts = re.split(r"(?<=[.!?])\s+|\n+", text or "")
        return [p.strip() for p in parts if p.strip()][:limit]
    except Exception:  # noqa: BLE001
        return []


__all__ = ["Conflict", "detect", "split_sentences"]
