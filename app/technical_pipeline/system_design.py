"""System-design pipeline — architecture-grade structured output.

Domain sections (functional/non-functional/architecture/data/scaling/failure)
plus the cross-cutting Assumptions / Pattern / Trade-offs / Governance sections
from `structured`, with a checklist verifier flagging anything missing.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from .structured import DomainSpec, run_structured_domain

_SPEC = DomainSpec(
    domain="system_design",
    role="staff systems architect",
    sections=[
        "Functional Requirements",
        "Non-functional Requirements / Scale Estimates",
        "High-level Architecture",
        "Data Model",
        "Scaling Strategy & Bottlenecks",
        "Failure Modes",
    ],
    checklist=[
        ("functional requirements", ["functional", "requirement"]),
        ("non-functional / scale", ["scale", "qps", "latency", "throughput"]),
        ("high-level architecture", ["architecture", "diagram", "component"]),
        ("data model", ["schema", "data model", "entity", "table"]),
        ("scaling strategy", ["shard", "replication", "cache", "partition"]),
        ("failure modes", ["failure", "fallback", "circuit breaker", "retry"]),
    ],
)


async def run(question: str) -> AsyncIterator[dict]:
    async for evt in run_structured_domain(question, _SPEC):
        yield evt
