"""Critic review pass (advanced-intent-reasoning R9).

A final, deterministic necessity check on the questions the gate wants to emit:
drop any question that (a) asks about a slot already known/suppressed, or
(b) has too little information gain to justify interrupting the user. If every
question is dropped, the turn answers directly. Blocking safety cards are never
removed (they don't pass through here — they short-circuit earlier — but the
`protect` guard makes the invariant explicit).
"""
from __future__ import annotations

# Map a question's `header`/`id` to the slot it asks about, so a known slot
# suppresses the corresponding question deterministically.
_HEADER_TO_SLOT = {
    "language": "language",
    "lang": "language",
    "framework": "framework",
    "stack": "framework",
    "platform": "platform",
    "target": "platform",
}


def _slot_for(question: dict) -> str | None:
    for field in ("header", "id"):
        v = str(question.get(field) or "").strip().lower()
        if v in _HEADER_TO_SLOT:
            return _HEADER_TO_SLOT[v]
    return None


def review(questions: list[dict], suppressed, known=None,
           min_options: int = 2) -> list[dict]:
    """Return the subset of [questions] worth asking (R9).

    Drops a question when:
      • the slot it asks about is in `suppressed`/`known` (already decided), or
      • it has fewer than `min_options` real choices (no information gain).
    A question flagged `blocking` is always kept (R9.4). Deterministic (R9.5).
    """
    if not questions:
        return questions
    sup = {str(s).strip().lower() for s in (suppressed or [])}
    for k in (known or {}):
        sup.add(str(k).strip().lower())
    out: list[dict] = []
    for q in questions:
        if q.get("blocking"):
            out.append(q)
            continue
        slot = _slot_for(q)
        if slot and slot in sup:
            continue  # already decided → no information gain
        if len(q.get("options") or []) < min_options:
            continue  # not a real choice
        out.append(q)
    return out


__all__ = ["review"]
