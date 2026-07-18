"""Expected-reply simulation (advanced-intent-reasoning R8).

When the system would clarify at borderline confidence but the most likely
answer is obvious, convert the question into a stated ASSUMPTION the user can
correct, instead of blocking on it. The gate already marks the best-fit option
`recommended`; that recommendation IS the estimated most-likely reply, so the
conversion is deterministic and grounded.
"""
from __future__ import annotations


def to_assumption(question: dict) -> dict | None:
    """Convert a clarifying [question] into an assumption using its own
    recommended option (R8). Returns None when there is no confident estimate
    (no recommended option) or the question is a blocking safety card (R8.4)."""
    if not isinstance(question, dict) or question.get("blocking"):
        return None
    recommended = next(
        (o for o in (question.get("options") or []) if o.get("recommended")),
        None,
    )
    if not recommended:
        return None
    header = str(question.get("header") or "").strip() or "Assumption"
    return {
        "id": str(question.get("id") or "").strip() or "a1",
        "label": header,
        "value": str(recommended.get("label") or "").strip(),
    }


def questions_to_assumptions(questions: list[dict]) -> list[dict]:
    """Map each convertible question to an assumption; skip the ones that can't
    be confidently estimated. Returns the assumptions list (may be shorter)."""
    out: list[dict] = []
    for q in questions or []:
        a = to_assumption(q)
        if a:
            out.append(a)
    return out


__all__ = ["to_assumption", "questions_to_assumptions"]
