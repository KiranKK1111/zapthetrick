"""Semantic gates — exemplar-embedding decisions for orchestration.

Every orchestration gate that used to be a hardcoded cue/keyword list is
expressed here as DATA (exemplar phrasings) scored by embedding similarity.
See `gates.py`.
"""
from app.semantics import gates  # noqa: F401

__all__ = ["gates"]
