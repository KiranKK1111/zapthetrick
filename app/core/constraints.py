"""Constraint Solver (roadmap Phase 4 #7).

Extracts the OUTPUT constraints a request imposes (must include tests, ≤ N lines,
valid JSON, avoid X…) and checks a produced output against them — a deterministic
gate that complements the sandbox (which proves it *runs*; this proves it meets
the *stated requirements*). Distinct from `clarify/requirement_matrix.py`, which
tracks INPUT slot-filling.

Conservative by design: only checks what can be verified deterministically, and
reports the rest as `unchecked` rather than guessing a verdict. Fail-open.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# constraint kinds
MUST_INCLUDE = "must_include"
MUST_AVOID = "must_avoid"
FORMAT = "format"
MAX_LINES = "max_lines"


@dataclass(frozen=True)
class Constraint:
    kind: str
    key: str          # the target (e.g. "tests", "json", "100", "recursion")
    text: str         # human-readable description
    required: bool = True


@dataclass
class ConstraintReport:
    satisfied: bool
    violations: list[str] = field(default_factory=list)
    checked: int = 0
    unchecked: list[str] = field(default_factory=list)


# extraction cues → constraints
_INCLUDE_CUES = {
    "tests": ("with tests", "include tests", "add tests", "unit test", "test cases"),
    "readme": ("readme", "documentation", "docs"),
    "error handling": ("error handling", "handle errors", "try/except", "try catch"),
    "comments": ("with comments", "commented", "add comments"),
    "types": ("type hints", "typed", "type annotations"),
}
_AVOID_CUES = {
    "recursion": ("no recursion", "without recursion", "avoid recursion", "iteratively"),
    "external dependencies": ("no external dependencies", "no third-party", "stdlib only",
                              "standard library only", "no dependencies"),
    "globals": ("no globals", "avoid global"),
}
_MAX_LINES_RE = re.compile(r"(?:under|less than|at most|max(?:imum)?|no more than)\s+(\d{1,4})\s*lines",
                           re.IGNORECASE)
_JSON_CUE = re.compile(r"\bas json\b|\bvalid json\b|\breturn json\b|\bjson format\b", re.IGNORECASE)


def extract_constraints(task: str) -> list[Constraint]:
    """Best-effort extraction of output constraints from a request."""
    out: list[Constraint] = []
    try:
        t = (task or "").lower()
        for key, cues in _INCLUDE_CUES.items():
            if any(c in t for c in cues):
                out.append(Constraint(MUST_INCLUDE, key, f"must include {key}"))
        for key, cues in _AVOID_CUES.items():
            if any(c in t for c in cues):
                out.append(Constraint(MUST_AVOID, key, f"must avoid {key}"))
        m = _MAX_LINES_RE.search(t)
        if m:
            out.append(Constraint(MAX_LINES, m.group(1), f"at most {m.group(1)} lines"))
        if _JSON_CUE.search(t):
            out.append(Constraint(FORMAT, "json", "output must be valid JSON"))
    except Exception:  # noqa: BLE001
        pass
    return out


def _check_one(output: str, c: Constraint) -> tuple[str, bool | None]:
    """Return (label, satisfied|None-if-unchecked)."""
    low = (output or "").lower()
    if c.kind == MUST_INCLUDE:
        # keyword presence (conservative — 'tests' looks for test markers)
        markers = {
            "tests": ("test", "def test", "@test", "assert", "expect("),
            "readme": ("readme", "# ", "## "),
            "error handling": ("try", "except", "catch", "error"),
            "comments": ("#", "//", "/*"),
            "types": (":", "->", "type"),
        }.get(c.key, (c.key,))
        return (c.text, any(m in low for m in markers))
    if c.kind == MUST_AVOID:
        markers = {
            "recursion": (),           # can't verify absence reliably → unchecked
            "external dependencies": ("import requests", "import numpy", "from third",
                                      "require("),
            "globals": ("global ",),
        }.get(c.key, (c.key,))
        if not markers:
            return (c.text, None)      # unchecked
        return (c.text, not any(m in low for m in markers))
    if c.kind == MAX_LINES:
        try:
            limit = int(c.key)
            return (c.text, len((output or "").splitlines()) <= limit)
        except Exception:  # noqa: BLE001
            return (c.text, None)
    if c.kind == FORMAT and c.key == "json":
        try:
            json.loads(output)
            return (c.text, True)
        except Exception:  # noqa: BLE001
            return (c.text, False)
    return (c.text, None)


def check(output: str, constraints: list[Constraint]) -> ConstraintReport:
    """Validate an output against constraints. Unverifiable ones are reported as
    `unchecked`, never counted as violations."""
    violations: list[str] = []
    unchecked: list[str] = []
    checked = 0
    try:
        for c in constraints or []:
            label, ok = _check_one(output or "", c)
            if ok is None:
                unchecked.append(label)
            else:
                checked += 1
                if not ok and c.required:
                    violations.append(label)
    except Exception:  # noqa: BLE001
        pass
    return ConstraintReport(satisfied=not violations, violations=violations,
                            checked=checked, unchecked=unchecked)


__all__ = [
    "Constraint", "ConstraintReport", "MUST_INCLUDE", "MUST_AVOID", "FORMAT",
    "MAX_LINES", "extract_constraints", "check",
]
