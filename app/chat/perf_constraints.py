"""Explicit performance/complexity constraint extraction for code requests.

"give me the code … which can execute within 500 milliseconds", "solution in
O(n log n)", "worst-case linear time", "constant space" — when the user states
a performance requirement, the answering model must design for EXACTLY that
requirement (and say so), not silently hand back the generic solution. This
module detects those constraints deterministically and produces the directive
folded into the answer prompt. Empty string = no explicit constraint → the
model defaults to the optimal (best-known) solution as usual.
"""
from __future__ import annotations

import re

# Time bounds: "within 500 ms", "under 2 seconds", "in less than 100ms",
# "between 500 and 1000 milliseconds".
_TIME_RANGE_RE = re.compile(
    r"between\s+(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|seconds?)?\s*"
    r"(?:and|to|-)\s*(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|seconds?)",
    re.IGNORECASE)
_TIME_BOUND_RE = re.compile(
    r"(?:within|under|below|less\s+than|at\s+most|max(?:imum)?\s+(?:of\s+)?|"
    r"faster\s+than|no\s+more\s+than)\s*"
    r"(\d+(?:\.\d+)?)\s*(ms|milliseconds?|s|seconds?)\b",
    re.IGNORECASE)

# Asymptotic bounds: O(n), O(n log n), O(1), O(n^2)…
_BIG_O_RE = re.compile(r"\bo\(\s*[^)]{1,24}\)", re.IGNORECASE)

# Named cases: "worst case", "average case", "best case", "amortized".
_CASE_RE = re.compile(
    r"\b(worst|average|best|amortized)[\s-]*case\b|\bamortized\b",
    re.IGNORECASE)

# Space requirements: "constant space", "in-place", "O(1) space",
# "space complexity", "without extra memory".
_SPACE_RE = re.compile(
    r"\b(constant\s+space|in[\s-]?place|extra\s+(?:space|memory)|"
    r"space\s+complexity|memory\s+(?:usage|limit|budget))\b",
    re.IGNORECASE)


def extract_performance_constraints(text: str) -> str:
    """Return the answer directive for explicit performance constraints in
    `text`, or "" when none are stated. Deterministic; never raises."""
    try:
        t = text or ""
        found: list[str] = []
        m = _TIME_RANGE_RE.search(t)
        if m:
            unit_a = (m.group(2) or m.group(4) or "ms").lower()
            unit_b = (m.group(4) or "ms").lower()
            found.append(
                f"execution time between {m.group(1)}{_u(unit_a)} and "
                f"{m.group(3)}{_u(unit_b)}")
        else:
            m = _TIME_BOUND_RE.search(t)
            if m:
                found.append(
                    f"execution time within {m.group(1)}{_u(m.group(2))}")
        for om in _BIG_O_RE.finditer(t):
            found.append(f"asymptotic bound {om.group(0)}")
        cm = _CASE_RE.search(t)
        if cm:
            case = (cm.group(1) or "amortized").lower()
            found.append(f"the {case}-case complexity is what matters")
        sm = _SPACE_RE.search(t)
        if sm:
            found.append(f"space requirement: {sm.group(0).lower()}")
        if not found:
            return ""
        bullets = "\n".join(f"- {f}" for f in found)
        return (
            "EXPLICIT PERFORMANCE REQUIREMENTS (stated by the user — these "
            "are part of the task, not suggestions):\n"
            f"{bullets}\n"
            "Design the solution to satisfy EXACTLY these requirements. "
            "State the solution's time and space complexity and briefly "
            "justify how it meets each stated bound (for a wall-clock budget, "
            "reason about the input size the complexity class can handle "
            "within it). If a stated bound is provably impossible for this "
            "problem, say so plainly and give the closest achievable "
            "alternative — never silently ignore the requirement."
        )
    except Exception:  # noqa: BLE001 — constraint extraction is best-effort
        return ""


def _u(unit: str | None) -> str:
    u = (unit or "ms").lower()
    return "s" if u.startswith("s") else "ms"
