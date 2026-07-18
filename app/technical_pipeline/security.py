"""Security-domain pipeline — architecture-grade structured output.

Threat-model-first answers: STRIDE-style threats, authn/authz, data
protection, input hardening, and detection/response — plus the cross-cutting
architecture sections. Flags an answer that omits a threat model.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from .structured import DomainSpec, run_structured_domain

_SPEC = DomainSpec(
    domain="security",
    role="senior application security engineer",
    sections=[
        "Threat Model",
        "Authentication & Authorization",
        "Data Protection",
        "Input Validation & Hardening",
        "Detection & Response",
    ],
    checklist=[
        ("threat model", ["threat", "attacker", "stride", "attack surface",
                          "risk", "adversary"]),
        ("authn/authz", ["authentication", "authorization", "oauth", "jwt",
                         "rbac", "session", "mfa", "least privilege"]),
        ("data protection", ["encryption", "tls", "secret", "hash", "at rest",
                             "in transit", "key management"]),
        ("input validation", ["validation", "sanitiz", "injection", "xss",
                              "csrf", "escaping", "parameterized"]),
        ("detection/response", ["logging", "audit", "monitor", "alert",
                                "incident", "rate limit"]),
    ],
)


async def run(question: str) -> AsyncIterator[dict]:
    async for evt in run_structured_domain(question, _SPEC):
        yield evt
