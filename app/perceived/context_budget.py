"""Context-window budgeting + conversation compression (perceived-speed R12, R13).

`allocate` splits the model's context window across memory / documents /
conversation / reserve per a configured weighting; the reserve is always kept
for the answer regardless of how much context is supplied (R12.3). `fit` drops
the lowest-relevance content first when the assembled context would exceed its
budget (R12.2). `ConversationCompressor` replaces a long history with a
structured summary that preserves key decisions while keeping the most recent
turns verbatim (R13), falling back to plain truncation if summarization fails.

Pure + deterministic — no model call required for the default extractive path.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

_DEFAULT_WEIGHTS = {
    "memory": 0.25,
    "documents": 0.30,
    "conversation": 0.25,
    "reserve": 0.20,
}

# Lines that look like a decision/constraint are preserved verbatim in a summary
# so later references to them remain answerable (R13.3).
_DECISION_CUES = (
    "use ", "using ", "decided", "must ", "should ", "don't use", "do not use",
    "no firebase", "chosen", "pick ", "prefer ", "instead of", "switch to",
    "requirement", "constraint",
)


@dataclass
class Budget:
    memory: int
    documents: int
    conversation: int
    reserve: int

    @property
    def total(self) -> int:
        return self.memory + self.documents + self.conversation + self.reserve


def allocate(window: int, weights: dict | None = None) -> Budget:
    """Split `window` tokens across the four buckets. The reserve is the
    remainder so the four always sum to exactly `window` and the reserve is
    never zeroed out by rounding (R12.1, R12.3)."""
    w = weights or _DEFAULT_WEIGHTS
    total = sum(w.values()) or 1.0
    memory = int(window * w.get("memory", 0) / total)
    documents = int(window * w.get("documents", 0) / total)
    conversation = int(window * w.get("conversation", 0) / total)
    reserve = window - memory - documents - conversation
    if reserve < 0:  # pathological weights — claw back from conversation
        conversation = max(0, conversation + reserve)
        reserve = window - memory - documents - conversation
    return Budget(memory, documents, conversation, reserve)


def fit(
    items: list,
    budget_tokens: int,
    *,
    relevance: Callable[[object], float],
    size: Callable[[object], int],
) -> list:
    """Keep the highest-relevance items that fit `budget_tokens`, dropping the
    lowest-relevance first (R12.2). Returns kept items in descending relevance."""
    ranked = sorted(items, key=relevance, reverse=True)
    out, used = [], 0
    for it in ranked:
        sz = size(it)
        if used + sz <= budget_tokens:
            out.append(it)
            used += sz
    return out


class ConversationCompressor:
    def __init__(self, recent_keep: int = 4) -> None:
        self.recent_keep = max(1, recent_keep)

    def compress(
        self,
        history: list[dict],
        *,
        summarize: Callable[[list[dict]], str] | None = None,
    ) -> list[dict]:
        """Replace older turns with a structured summary; keep the most recent
        `recent_keep` turns verbatim (R13.1, R13.2)."""
        if len(history) <= self.recent_keep:
            return list(history)
        old = history[: -self.recent_keep]
        recent = history[-self.recent_keep:]
        try:
            summary = summarize(old) if summarize else self._extractive(old)
            if not summary:
                summary = self._truncate(old)
        except Exception:  # noqa: BLE001 — fall back to truncation (R13)
            summary = self._truncate(old)
        head = {
            "role": "system",
            "content": "Summary of earlier conversation (key decisions kept):\n"
            + summary,
        }
        return [head] + list(recent)

    def _extractive(self, turns: list[dict]) -> str:
        lines: list[str] = []
        for t in turns:
            for ln in str(t.get("content", "")).splitlines():
                low = ln.lower()
                if any(c in low for c in _DECISION_CUES):
                    s = ln.strip()
                    if s and s not in lines:
                        lines.append(s)
        if not lines:  # no decisions detected → first slice of each old turn
            lines = [
                str(t.get("content", "")).strip()[:120]
                for t in turns
                if str(t.get("content", "")).strip()
            ]
        return "\n".join(f"- {ln}" for ln in lines[:20])

    def _truncate(self, turns: list[dict]) -> str:
        blob = " ".join(str(t.get("content", "")) for t in turns)
        blob = " ".join(blob.split())
        return blob[:800]


__all__ = ["Budget", "allocate", "fit", "ConversationCompressor"]
