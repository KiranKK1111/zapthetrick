"""In-session working memory.

Holds the current conversation's recent Q&As, the active blackboard,
and any short-lived scratch state the agents need. Cleared at session
end — never persisted.

Designed as a small ring buffer over the last `max_turns` interactions
so context injection has predictable bounds.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Turn:
    question: str
    answer: str
    intent: str = "general"
    meta: dict[str, Any] = field(default_factory=dict)


class WorkingMemory:
    def __init__(self, *, max_turns: int = 20) -> None:
        self._turns: deque[Turn] = deque(maxlen=max_turns)
        self._scratch: dict[str, Any] = {}

    # ---- turns -------------------------------------------------------
    def add_turn(self, turn: Turn) -> None:
        self._turns.append(turn)

    def recent(self, n: int | None = None) -> list[Turn]:
        if n is None:
            return list(self._turns)
        return list(self._turns)[-n:]

    # ---- scratch -----------------------------------------------------
    def put(self, key: str, value: Any) -> None:
        self._scratch[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._scratch.get(key, default)

    def clear(self) -> None:
        self._turns.clear()
        self._scratch.clear()
