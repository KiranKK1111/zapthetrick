"""Shared async typed state for the multi-agent mesh.

Agents do not call each other directly. They read from and write to a
typed [Blackboard]; the [Scheduler] runs each agent when its inputs are
present and its deadline hasn't elapsed.
"""
from .board import Blackboard, BlackboardEvent
from .schema import (
    SessionState,
    Intent,
    Plan,
    Evidence,
    MemoryHits,
    Drafts,
    Critiques,
    Grounding,
    Meta,
    Suggestions,
)
from .scheduler import PriorityScheduler

__all__ = [
    "Blackboard",
    "BlackboardEvent",
    "SessionState",
    "Intent",
    "Plan",
    "Evidence",
    "MemoryHits",
    "Drafts",
    "Critiques",
    "Grounding",
    "Suggestions",
    "Meta",
    "PriorityScheduler",
]
