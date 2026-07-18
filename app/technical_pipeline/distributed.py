"""Distributed-systems & reliability pipeline (Phase 13, report #42/#43).

Architecture-grade structured output for distributed / reliability questions
(consensus, replication, partitioning, failure handling, idempotency, back-
pressure), reusing the shared `structured` framework + checklist verifier.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from .structured import DomainSpec, run_structured_domain

_SPEC = DomainSpec(
    domain="distributed",
    role="staff distributed-systems engineer",
    sections=[
        "Consistency & Consensus",
        "Partitioning & Replication",
        "Failure Handling & Reliability",
        "Coordination & Messaging",
        "Back-pressure & Flow Control",
    ],
    checklist=[
        ("consistency/consensus", ["consistency", "consensus", "quorum",
                                   "raft", "paxos", "lineariz", "cap"]),
        ("partitioning/replication", ["partition", "shard", "replication",
                                      "leader", "follower", "rebalanc"]),
        ("failure handling", ["retry", "timeout", "circuit breaker",
                              "idempoten", "fallback", "failover", "fencing"]),
        ("coordination/messaging", ["queue", "kafka", "event", "saga",
                                    "outbox", "exactly-once", "at-least-once",
                                    "lease", "lock"]),
        ("back-pressure", ["back-pressure", "backpressure", "rate limit",
                           "bulkhead", "load shed", "buffer", "throttl"]),
    ],
)


async def run(question: str) -> AsyncIterator[dict]:
    async for evt in run_structured_domain(question, _SPEC):
        yield evt
