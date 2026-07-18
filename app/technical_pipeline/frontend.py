"""Frontend-domain pipeline — architecture-grade structured output.

Carries an a11y / web-vitals / state-management checklist alongside the
cross-cutting architecture sections.
"""
from __future__ import annotations

from collections.abc import AsyncIterator

from .structured import DomainSpec, run_structured_domain

_SPEC = DomainSpec(
    domain="frontend",
    role="senior frontend architect",
    sections=[
        "Component Architecture",
        "State Management",
        "Rendering Strategy (CSR / SSR / SSG)",
        "Performance (Web Vitals)",
        "Accessibility (a11y)",
    ],
    checklist=[
        ("component architecture", ["component", "composition", "props",
                                    "module", "design system"]),
        ("state management", ["state", "store", "context", "redux", "signal",
                              "query cache"]),
        ("rendering strategy", ["csr", "ssr", "ssg", "hydration", "render"]),
        ("performance (web vitals)", ["lcp", "fcp", "cls", "inp", "bundle",
                                      "lazy", "web vital"]),
        ("accessibility", ["aria", "role=", "semantic", "alt=", "a11y",
                           "keyboard", "contrast"]),
    ],
)


async def run(question: str) -> AsyncIterator[dict]:
    async for evt in run_structured_domain(question, _SPEC):
        yield evt
