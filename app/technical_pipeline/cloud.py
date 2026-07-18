"""Cloud-domain pipeline — architecture-grade structured output.

Provider-agnostic; the classifier hint (aws/gcp/azure) flows through the
question so the model picks the right managed services.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from .structured import DomainSpec, run_structured_domain

_SPEC = DomainSpec(
    domain="cloud",
    role="senior cloud architect",
    sections=[
        "Service Selection",
        "Networking & Access (IAM / VPC)",
        "Scaling & Availability",
        "Cost Optimization",
        "Operational Concerns",
    ],
    checklist=[
        ("service selection", ["service", "managed", "lambda", "s3", "compute",
                               "bucket", "queue", "function"]),
        ("networking/access", ["iam", "vpc", "role", "policy", "network",
                               "security group", "subnet"]),
        ("scaling/availability", ["auto", "scale", "availability", "region",
                                  "multi-az", "failover"]),
        ("cost", ["cost", "pricing", "spot", "reserved", "budget", "tier"]),
        ("operations", ["monitor", "cloudwatch", "logging", "backup", "alarm"]),
    ],
)


async def run(question: str) -> AsyncIterator[dict]:
    async for evt in run_structured_domain(question, _SPEC):
        yield evt
