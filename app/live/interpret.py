"""
Multi-hypothesis interpretation of ambiguous questions
(roadmap Phase 2 #15 / 2B-15).

A short or under-specified question ("what about it?", "and scaling?", "why?")
can be read several ways. Instead of committing to one silent reading, this
produces a small ranked set of candidate INTERPRETATIONS (reading + intent +
confidence) so the answer can address the most-likely reading and briefly
acknowledge the alternative rather than confidently answering the wrong one.

Deterministic heuristic (length, pronoun/anaphora, bare-topic, why/how bareness)
— no LLM. Advisory: surfaced as `meta.interpretations`; when genuinely
ambiguous it adds a one-line directive. Fail-open → single interpretation.
"""
from __future__ import annotations

from dataclasses import dataclass

_ANAPHORA = {"it", "that", "this", "they", "those", "these", "them", "one"}
_BARE_WH = {"why", "how", "what", "when", "where", "which", "who"}


@dataclass
class Interpretation:
    reading: str
    intent: str
    confidence: float = 0.5

    def to_dict(self) -> dict:
        return {"reading": self.reading, "intent": self.intent,
                "confidence": round(self.confidence, 3)}


def is_ambiguous(question: str) -> bool:
    """Whether a question is under-specified enough to warrant multiple
    readings. Never raises → False."""
    try:
        q = (question or "").strip()
        if not q:
            return False
        words = [w.strip("?.!,").lower() for w in q.split() if w.strip("?.!,")]
        if not words:
            return False
        n = len(words)
        # Very short probe.
        if n <= 3:
            return True
        # Leads with a bare wh + short and no concrete noun after.
        if words[0] in _BARE_WH and n <= 5:
            return True
        # Pronoun-driven with no antecedent noun in the utterance itself.
        if words[0] in ("and", "but", "so") and n <= 6:
            return True
        # Dominated by anaphora ("what about it then").
        if any(w in _ANAPHORA for w in words) and n <= 6:
            return True
        return False
    except Exception:  # noqa: BLE001
        return False


def interpretations(question: str, *, topic: str | None = None,
                    max_n: int = 3) -> list[Interpretation]:
    """Ranked candidate readings for an ambiguous question. Non-ambiguous input
    yields a single high-confidence literal reading. Never raises."""
    try:
        q = (question or "").strip()
        if not q:
            return []
        if not is_ambiguous(q):
            return [Interpretation(reading=q, intent="literal", confidence=0.9)]
        t = (topic or "").strip()
        low = q.lower()
        out: list[Interpretation] = []
        subject = t or "the current topic"
        if low.startswith("why"):
            out.append(Interpretation(f"Why does {subject} behave that way / why choose it?",
                                      "rationale", 0.6))
            out.append(Interpretation(f"Why is {subject} important here?",
                                      "motivation", 0.4))
        elif low.startswith("how"):
            out.append(Interpretation(f"How does {subject} work internally?",
                                      "mechanism", 0.55))
            out.append(Interpretation(f"How would you use/apply {subject}?",
                                      "application", 0.45))
        elif any(w in low for w in _ANAPHORA) or low.startswith(("and", "but", "so")):
            out.append(Interpretation(f"A follow-up drilling into {subject}.",
                                      "followup_depth", 0.55))
            out.append(Interpretation(f"A pivot to a related aspect of {subject}.",
                                      "followup_breadth", 0.45))
        else:
            out.append(Interpretation(f"A direct question about {subject}.",
                                      "literal", 0.6))
            out.append(Interpretation(f"A broader question around {subject}.",
                                      "broad", 0.4))
        return out[:max_n]
    except Exception:  # noqa: BLE001
        return []


def directive(hyps: list[Interpretation]) -> str:
    """When multiple genuine readings exist, instruct the answer to lead with
    the most likely and briefly acknowledge the alternative. '' otherwise."""
    try:
        real = [h for h in hyps if h.intent != "literal"]
        if len(hyps) < 2 or not real:
            return ""
        top = max(hyps, key=lambda h: h.confidence)
        return ("The question is ambiguous. Lead with the most likely reading — "
                f"{top.reading} — and, in one clause, acknowledge it could also "
                "mean something else so you don't answer the wrong question.")
    except Exception:  # noqa: BLE001
        return ""


__all__ = ["Interpretation", "is_ambiguous", "interpretations", "directive"]
