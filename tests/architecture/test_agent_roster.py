"""Guardrail rule #1 — no "agent explosion".

The design is one Supervisor + a fixed, small cast of specialist agents over a
shared Blackboard — NOT 100 bespoke agents. A new agent module must be added to
the allowlist deliberately (the git diff is the justification).
"""
from __future__ import annotations

from . import _scan
from ._allowlists import ALLOWED_AGENTS

# `base` is the shared base class, not a specialist agent — excluded by design.
_INFRA = {"base"}


def test_no_unlisted_agents():
    current = _scan.package_modules("agents") - _INFRA
    new = current - ALLOWED_AGENTS
    assert not new, (
        f"New agent module(s) {sorted(new)} under app/agents/ are not allowlisted. "
        f"Rule #1 (no agent explosion): prefer extending an existing specialist or "
        f"the capability registry. If a genuinely new specialist is warranted, add "
        f"it to governance_allowlist.json['agents'] with intent."
    )


def test_agent_roster_stays_small():
    # A soft ceiling: the whole point is a *small* cast. If this trips, it's a
    # signal to consolidate, not to raise the number thoughtlessly.
    current = _scan.package_modules("agents") - _INFRA
    assert len(current) <= 20, (
        f"{len(current)} specialist agents — the roster is meant to stay small "
        f"(rule #1). Consolidate before adding more."
    )
