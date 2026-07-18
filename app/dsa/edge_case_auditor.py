"""Stage 8 — Edge-case auditor.

Returns a short, focused list of edge cases the user should consider.
Heuristic-first: the spec's standard list ("empty / one / max size /
negative / overflow / duplicates / sorted / reverse-sorted / unicode /
cycles") is applied based on the problem's pattern + code signals.
No LLM call — fast, deterministic, and good enough for an interview
checklist.
"""
from __future__ import annotations

import re

from .types import PatternMatch, ProblemSpec, SolutionApproach


_PATTERN_EDGE_CASES: dict[str, list[str]] = {
    "arrays/two-pointer": [
        "Empty array",
        "Single-element array",
        "All elements identical",
        "Already sorted vs reverse-sorted",
    ],
    "sliding-window": [
        "Window larger than the array",
        "All identical elements",
        "All distinct elements",
        "Negative numbers (if relevant)",
    ],
    "binary-search": [
        "Target smaller than every element",
        "Target larger than every element",
        "Target equal to first / last / middle",
        "Empty array",
        "Duplicates of the target",
    ],
    "linked-list": [
        "Empty list (head == null)",
        "Single node",
        "Two nodes",
        "Cycle in the list",
    ],
    "trees": [
        "Empty tree",
        "Single node",
        "Completely skewed tree (linked-list shape)",
        "Perfectly balanced tree",
    ],
    "graphs": [
        "Disconnected graph",
        "Self-loops",
        "Multi-edges",
        "Cycles (for DFS / topo sort)",
        "Single-node graph",
    ],
    "dp": [
        "Base case (n == 0 or 1)",
        "Overflow on large n (use modular arithmetic or BigInt)",
        "Negative inputs (if domain allows)",
    ],
    "strings": [
        "Empty string",
        "Single character",
        "Unicode / non-ASCII",
        "All same character",
        "Already a palindrome (if relevant)",
    ],
    "hashing": [
        "All duplicates",
        "Single element",
        "Negative numbers",
        "Hash collisions don't matter — but `dict` ordering can",
    ],
    "heap/top-k": [
        "k larger than the input size",
        "k == 0",
        "Duplicates",
        "Already-sorted input",
    ],
    "backtracking": [
        "No valid solution exists",
        "Multiple equally-valid solutions",
        "Max depth — beware recursion limits",
    ],
    "greedy": [
        "Counter-example where greedy fails (if any)",
        "Tie-breaking rules",
        "Empty / single-element input",
    ],
}


_CODE_HINTS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(int\(|abs\(|sum\()"), "Integer overflow on very large inputs"),
    (re.compile(r"\b(while\s+\w+\s*!?=)\b"), "Infinite-loop guard — what makes the loop terminate?"),
    (re.compile(r"\b(left|right|lo|hi|low|high)\b"), "Off-by-one at the boundary"),
    (re.compile(r"\b(divmod|//|%)"), "Division by zero"),
]


def audit(
    problem: ProblemSpec,
    pattern: PatternMatch,
    approach: SolutionApproach | None,
) -> list[str]:
    """Return a deduped list of edge cases, ordered: pattern-based
    first, then code-signal hints. Capped at 8 entries to keep the
    answer focused (interview attention span)."""
    out: list[str] = list(_PATTERN_EDGE_CASES.get(pattern.family, []))

    if approach and approach.code:
        seen = set(out)
        for regex, suggestion in _CODE_HINTS:
            if regex.search(approach.code) and suggestion not in seen:
                out.append(suggestion)
                seen.add(suggestion)

    # Generic fallback when the pattern wasn't recognised — still useful
    # for an interview checklist.
    if not out:
        out = [
            "Empty input",
            "Single-element input",
            "Maximum-size input (per constraints)",
            "Duplicates / all-same",
            "Negative or zero values (if domain allows)",
        ]

    return out[:8]
