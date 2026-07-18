"""Typed slots stored on the [Blackboard] for one session.

Each agent declares the slots it reads (subscribes to) and writes
(publishes). The schema is intentionally a flat namespace of dataclasses
so the supervisor can introspect what's available at any moment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# Slot keys — used by both [Blackboard] and agents' reads/writes sets.
KEY_INTENT = "intent"
KEY_PLAN = "plan"
KEY_EVIDENCE = "evidence"
KEY_MEMORY_HITS = "memory_hits"
KEY_DRAFTS = "drafts"
KEY_CRITIQUES = "critiques"
KEY_GROUNDING = "grounding"
KEY_META = "meta"
KEY_SUGGESTIONS = "suggestions"
KEY_QUESTION = "question"


@dataclass
class Intent:
    """Classifier output from the Planner agent."""
    type: str = "general"          # behavioral | coding | concept | general
    topic: str = ""
    urgency: str = "normal"        # low | normal | high
    needs_clarification: bool = False


@dataclass
class Plan:
    """Execution plan produced by the Planner agent."""
    steps: list[str] = field(default_factory=list)
    priorities: list[int] = field(default_factory=list)
    deadlines_ms: list[int] = field(default_factory=list)
    parallel: bool = True


@dataclass
class EvidenceChunk:
    text: str
    source: str
    score: float
    parent_id: str | None = None


@dataclass
class Evidence:
    chunks: list[EvidenceChunk] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)


@dataclass
class MemoryHits:
    episodes: list[dict] = field(default_factory=list)
    skills: list[dict] = field(default_factory=list)


@dataclass
class Drafts:
    current: str = ""
    history: list[str] = field(default_factory=list)


@dataclass
class Critiques:
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class Grounding:
    verified_claims: list[str] = field(default_factory=list)
    unverified: list[str] = field(default_factory=list)


@dataclass
class Meta:
    latency_budget_ms: int = 8000
    tokens_used: int = 0
    agents_active: list[str] = field(default_factory=list)
    started_at_ms: int = 0


@dataclass
class Suggestions:
    proactive: list[str] = field(default_factory=list)


@dataclass
class SessionState:
    """Aggregate view used for introspection / UI events.

    The [Blackboard] holds these as discrete keys; this dataclass mirrors
    them so callers (and tests) can take a snapshot.
    """
    question: str = ""
    intent: Intent = field(default_factory=Intent)
    plan: Plan = field(default_factory=Plan)
    evidence: Evidence = field(default_factory=Evidence)
    memory_hits: MemoryHits = field(default_factory=MemoryHits)
    drafts: Drafts = field(default_factory=Drafts)
    critiques: Critiques = field(default_factory=Critiques)
    grounding: Grounding = field(default_factory=Grounding)
    meta: Meta = field(default_factory=Meta)
    suggestions: Suggestions = field(default_factory=Suggestions)
    extras: dict[str, Any] = field(default_factory=dict)
