"""Database-domain pipeline — architecture-grade structured output + SQL lint.

Keeps the dangerous-SQL sanity check (DELETE/UPDATE without WHERE, DROP TABLE,
SELECT *) on top of the structured architecture sections.
"""
from __future__ import annotations

import re
from collections.abc import AsyncIterator

from .structured import DomainSpec, run_structured_domain

_DANGEROUS_PATTERNS = [
    (re.compile(r"\bDELETE\s+FROM\s+\w+\s*;", re.I), "DELETE without WHERE clause"),
    (re.compile(r"\bUPDATE\s+\w+\s+SET[^;]*?(?<!WHERE)[^;]*;", re.I),
     "UPDATE possibly without WHERE clause"),
    (re.compile(r"\bDROP\s+TABLE\b", re.I), "DROP TABLE — confirm intent"),
    (re.compile(r"\bSELECT\s+\*\s+FROM\s+\w+\s*;", re.I),
     "SELECT * — name columns explicitly"),
]


def _lint_sql(text: str) -> list[str]:
    out: list[str] = []
    for pat, msg in _DANGEROUS_PATTERNS:
        if pat.search(text or "") and msg not in out:
            out.append(msg)
    return out


_SPEC = DomainSpec(
    domain="databases",
    role="senior database engineer",
    sections=[
        "Schema & Data Model",
        "Indexing Strategy",
        "Query Design & Plan",
        "Transactions & Consistency",
        "Scaling (Partitioning / Replication)",
    ],
    checklist=[
        ("schema / data model", ["schema", "table", "column", "entity", "data model"]),
        ("indexing", ["index", "btree", "composite", "covering"]),
        ("query plan", ["query", "explain", "plan", "join"]),
        ("transactions", ["transaction", "isolation", "acid", "lock", "mvcc"]),
        ("scaling", ["shard", "partition", "replication", "read replica"]),
    ],
    lint=_lint_sql,
)


async def run(question: str) -> AsyncIterator[dict]:
    async for evt in run_structured_domain(question, _SPEC):
        yield evt
