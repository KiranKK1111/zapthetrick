"""Self-documenting manifest of the 20 architecture guardrails.

Maps every anti-pattern rule from ImplementationRoadmap.md ("🚫→✅ Anti-Patterns"
/ "🛡️ Guardrail Enforcement Plan") to how it is enforced, and asserts that each
rule marked `ci` in this Phase-0 batch actually has a live test file. This keeps
the suite honest: coverage is visible, and a `ci` rule can't be silently dropped.

Enforcement kinds:
  ci        - a static fitness test in THIS directory enforces it now (Phase 0)
  runtime   - already enforced live in app code (documented, not re-tested here)
  checklist - needs human judgment; a PR-review item (structural half may be ci)
  planned   - ci enforcement scheduled for a later phase (not yet built)
"""
from __future__ import annotations

import pathlib

_HERE = pathlib.Path(__file__).resolve().parent

# rule_no: (short name, enforcement kind, test file if ci/None)
GUARDRAILS: dict[int, tuple[str, str, str | None]] = {
    1:  ("agent explosion",            "ci",        "test_agent_roster.py"),
    2:  ("prompt explosion",           "planned",   None),
    3:  ("memory explosion",           "ci",        "test_memory_stores.py"),
    4:  ("tool explosion",             "ci",        "test_tool_registry.py"),
    5:  ("hardcoded anything",         "planned",   None),
    6:  ("translation multilingual",   "planned",   None),
    7:  ("separate speaker pipelines", "planned",   None),
    8:  ("instant everything",         "planned",   None),
    9:  ("narrow negotiation engine",  "checklist", None),
    10: ("blind retries",              "planned",   None),
    11: ("unbounded planning",         "planned",   None),
    12: ("single-model coupling",      "planned",   None),
    13: ("hidden coupling",            "ci",        "test_import_boundaries.py"),
    14: ("plain-text history",         "checklist", None),
    15: ("opaque policies",            "ci",        "test_policy_transparency.py"),
    16: ("prompt->response only",      "planned",   None),
    17: ("raw token streaming",        "planned",   None),
    18: ("re-streaming answers",       "planned",   None),
    19: ("one streaming impl",         "planned",   None),
    20: ("module #121",                "ci",        "test_package_governance.py"),
}


def test_all_twenty_rules_present():
    assert set(GUARDRAILS) == set(range(1, 21)), (
        "The manifest must cover exactly the 20 roadmap anti-pattern rules."
    )


def test_ci_rules_have_live_test_files():
    for rule_no, (name, kind, test_file) in GUARDRAILS.items():
        if kind == "ci":
            assert test_file, f"Rule #{rule_no} ({name}) is 'ci' but names no test file."
            assert (_HERE / test_file).exists(), (
                f"Rule #{rule_no} ({name}) claims enforcement by {test_file}, "
                f"which is missing. A 'ci' rule must have a live test."
            )


def test_phase0_batch_coverage():
    ci = {n for n, (_, k, _) in GUARDRAILS.items() if k == "ci"}
    # Phase 0 delivers the six structural/static guardrails; the rest are staged
    # (runtime-documented, checklist, or planned for later phases).
    assert ci == {1, 3, 4, 13, 15, 20}, (
        f"Phase 0 CI guardrail set changed: {sorted(ci)}. Update this assertion "
        f"and the roadmap's enforcement table together so they stay in sync."
    )
