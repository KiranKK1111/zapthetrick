"""Pick a specialised pipeline for a technical question.

Classification is heuristic: keyword scan over the question. The
result picks a sub-pipeline module from `app.technical_pipeline.*`.
If no domain fires confidently we fall back to `generic`.
"""
from __future__ import annotations

import re
from collections.abc import AsyncIterator


DOMAINS = (
    "system_design",
    "databases",
    "backend",
    "security",
    "devops",
    "cloud",
    "frontend",
    "distributed",
    "generic",
    # TODO: ml, mobile, embedded, networking, browsers, performance,
    # observability, accessibility, testing, build_systems, package_managers,
    # language_runtimes, crypto, data_pipelines, oss_governance, hiring.
)


_DOMAIN_HEURISTICS: list[tuple[str, re.Pattern]] = [
    ("distributed", re.compile(
        r"\b(distributed\s+system|consensus|raft|paxos|quorum|lineariz|"
        r"eventual\s+consistency|cap\s+theorem|two[- ]phase\s+commit|2pc|"
        r"saga|exactly[- ]once|at[- ]least[- ]once|leader\s+election|"
        r"circuit\s+breaker|back[- ]?pressure|idempoten|split[- ]brain|"
        r"gossip|vector\s+clock|crdt)\b", re.I)),
    ("security", re.compile(
        r"\b(security|secure|threat\s+model|owasp|vulnerab|xss|csrf|sql\s+"
        r"injection|injection|auth(?:entication|orization)?|oauth|jwt|rbac|"
        r"encrypt|tls|ssl|hash(?:ing)?|secret|pentest|penetration|exploit|"
        r"mfa|csp|sast|dast)\b", re.I)),
    ("system_design", re.compile(
        r"\b(system\s+design|scalab|sharding|load\s*balanc|cap\s+theorem|"
        r"event\s+sourcing|microservic|high[- ]availability|message\s+queue|"
        r"design\s+(?:a\s+|the\s+)?(?:system|architecture|platform|"
        r"distributed))\b", re.I)),
    ("databases", re.compile(
        r"\b(indexes?|isolation\s+level|MVCC|btree|hash\s+index|sharding|"
        r"replication|wal\s+|materialized\s+view|query\s+plan|EXPLAIN|"
        r"transaction|deadlock|schema\s+design|normaliz|postgres|mysql|"
        r"mongodb)\b", re.I)),
    ("devops", re.compile(
        r"\b(ci/cd|ci\s*cd|pipeline|jenkins|github\s+actions|gitlab\s+ci|"
        r"helm|kubernetes|k8s|docker\s*compose|terraform|ansible|deployment|"
        r"blue[- ]green|canary|rollback)\b", re.I)),
    ("cloud", re.compile(
        r"\b(aws|gcp|azure|s3|ec2|lambda|cloudwatch|iam|vpc|cloudfront|"
        r"cdn|fargate|cloud\s+run|app\s+service|managed\s+service)\b", re.I)),
    ("frontend", re.compile(
        r"\b(react|vue|svelte|next\.?js|nuxt|hydration|csr|ssr|ssg|"
        r"web\s+vitals|reflow|repaint|virtual\s+dom|css\s+grid|flex(box)?|"
        r"a11y|accessibility|component\s+library|design\s+system)\b", re.I)),
    ("backend", re.compile(
        r"\b(rest\s+api|graphql|grpc|api\s+(?:design|endpoint|gateway)|"
        r"endpoint|middleware|orm\b|webhook|rate\s+limit|idempoten|"
        r"backend|server[- ]side|business\s+logic|pagination)\b", re.I)),
]


def classify_domain(question: str) -> str:
    """Pick the best-matching domain. Falls back to `generic`."""
    text = question or ""
    for name, pat in _DOMAIN_HEURISTICS:
        if pat.search(text):
            return name
    return "generic"


async def dispatch(question: str, *, hint: str | None = None) -> AsyncIterator[dict]:
    """Run the matching domain pipeline. Yields the same event shape
    as `app.dsa.solve()` so callers can fan into one renderer."""
    domain = hint or classify_domain(question)
    pipeline = _resolve_pipeline(domain)
    async for evt in pipeline(question):
        yield evt


def _resolve_pipeline(domain: str):
    """Lazy import — don't pay the cost of loading every pipeline at
    process start."""
    import importlib

    mod_name = f"app.technical_pipeline.{domain}"
    try:
        mod = importlib.import_module(mod_name)
    except ImportError:
        mod = importlib.import_module("app.technical_pipeline.generic")
    return mod.run


__all__ = ["dispatch", "classify_domain", "DOMAINS"]
