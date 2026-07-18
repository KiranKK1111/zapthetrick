"""DevOps / CI-CD pipeline — architecture-grade structured output + artifacts.

Uses the ARTIFACT_SET shape so multi-file deployment answers (Dockerfile +
compose + k8s manifest) render as a tabbed artifact card, while still carrying
the cross-cutting Assumptions / Pattern / Trade-offs / Governance sections.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from app.response_arch import Shape

from .structured import DomainSpec, run_structured_domain

_SPEC = DomainSpec(
    domain="devops",
    role="senior DevOps / platform engineer",
    sections=[
        "Pipeline Stages",
        "Environments & Promotion",
        "Infrastructure as Code",
        "Rollout & Rollback Strategy",
        "Monitoring & Alerting",
    ],
    checklist=[
        ("pipeline stages", ["build", "test", "deploy", "stage", "pipeline"]),
        ("environments", ["environment", "staging", "production", "promote"]),
        ("infra as code", ["terraform", "ansible", "helm", "manifest",
                            "dockerfile", "compose", "iac"]),
        ("rollout/rollback", ["rollout", "rollback", "blue", "green", "canary"]),
        ("monitoring", ["monitor", "alert", "metric", "log", "observability"]),
    ],
    shape=Shape.ARTIFACT_SET,
    emit_artifacts=True,
)


async def run(question: str) -> AsyncIterator[dict]:
    async for evt in run_structured_domain(question, _SPEC):
        yield evt
