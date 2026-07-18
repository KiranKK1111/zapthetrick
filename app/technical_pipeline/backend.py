"""Backend-domain pipeline — architecture-grade structured output.

API/service design questions: contracts, persistence, core logic, resilience,
and performance, plus the cross-cutting architecture sections.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from .structured import DomainSpec, run_structured_domain

_SPEC = DomainSpec(
    domain="backend",
    role="senior backend engineer",
    sections=[
        "API Design & Contracts",
        "Data & Persistence",
        "Core Logic & Services",
        "Error Handling & Resilience",
        "Performance & Scaling",
    ],
    checklist=[
        ("api design", ["endpoint", "rest", "grpc", "graphql", "api",
                        "contract", "request", "response", "status code"]),
        ("persistence", ["database", "sql", "orm", "cache", "storage",
                         "repository", "model"]),
        ("core logic", ["service", "handler", "business logic", "validation",
                        "workflow"]),
        ("resilience", ["retry", "timeout", "circuit", "idempoten", "error",
                        "graceful", "backoff"]),
        ("performance", ["latency", "throughput", "pagination", "n+1", "index",
                         "connection pool", "rate limit"]),
    ],
)


async def run(question: str) -> AsyncIterator[dict]:
    async for evt in run_structured_domain(question, _SPEC):
        yield evt
