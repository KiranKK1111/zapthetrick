"""Unified semantic understanding (the 'brain').

One embedding of the turn drives every routing signal — intent, difficulty,
topic-shift, task category, required capabilities, output complexity, ambiguity —
so downstream stages (clarifier, persona, model router) read ONE coherent object
instead of recomputing from scattered keyword rules. See `pass.py`.
"""
from app.understanding.understanding_pass import (
    Understanding,
    enabled,
    last_embedding,
    remember_embedding,
    remember_turn_meta,
    turn_meta,
    understand,
)

__all__ = [
    "Understanding", "understand", "enabled",
    "last_embedding", "remember_embedding",
    "remember_turn_meta", "turn_meta",
]
