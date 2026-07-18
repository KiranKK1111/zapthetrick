"""Guardrail rule #15 — no "opaque policies".

Decisions are declarative policy rules (id/action/priority) and every decision is
recorded (a PolicyDecision). This enforces the structural contract so the
"every decision is inspectable / explainable" invariant can't silently erode.
"""
from __future__ import annotations

from . import _scan

_ENGINE = _scan.APP_ROOT / "policy" / "engine.py"
_REQUIRED_RULE_FIELDS = {"id", "action", "priority"}


def test_policy_rule_is_declarative():
    fields = _scan.dataclass_fields(_ENGINE, "PolicyRule")
    assert fields, "app/policy/engine.py must define a `PolicyRule`."
    missing = _REQUIRED_RULE_FIELDS - fields
    assert not missing, (
        f"`PolicyRule` missing {sorted(missing)}. A policy must carry "
        f"id/action/priority so decisions are declarative and auditable (rule #15)."
    )


def test_policy_decision_is_recorded():
    assert _scan.has_class(_ENGINE, "PolicyDecision"), (
        "app/policy/engine.py must define a `PolicyDecision` record — every "
        "decision returns which rules fired and why (rule #15: no opaque policies)."
    )
