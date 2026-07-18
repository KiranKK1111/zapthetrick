"""Stage 10 — Beautifier.

Composes the final markdown answer from every prior stage's output.
The Flutter renderer already handles GitHub-flavoured markdown +
fenced code, so the beautifier just has to be *consistent*: same
section order, same heading levels, same anchor names.

Layout (mirrors the spec's "what users expect from an interview answer"):

    # <Title>

    ## Problem
    <statement>

    ## Pattern
    **Family:** <family> &nbsp;·&nbsp; **Confidence:** 0.xx
    <rationale>

    ## Approaches
    ### Brute
    <summary>
    <reasoning>
    ```{lang} ... ```

    ### Optimized
    ...

    ### Optimal
    ...

    ## Complexity
    | Metric | Value |
    | --- | --- |
    | Time  | O(...) |
    | Space | O(...) |
    > <proof>

    ## Examples
    - **Small** — input → output
    - **Edge** — input → output
    - **Large** — input → output

    ## Edge cases
    - bullet 1
    - bullet 2

    ## Verification
    Passed 8/9 · 1 failed (RESULT 4: FAIL ...)

    ## Follow-ups
    - "Can you trace it on input X?"
"""
from __future__ import annotations

from .types import (
    ComplexityProof,
    DsaResponse,
    PatternMatch,
    ProblemSpec,
    SolutionApproach,
    VerifyResult,
)


def compose(resp: DsaResponse) -> str:
    """Render the full response. Always returns at least the problem +
    pattern sections, even when downstream stages produced nothing."""
    parts: list[str] = []

    title = resp.problem.title.strip() or "Solution"
    parts.append(f"# {title}\n")

    if resp.problem.statement.strip():
        parts.append("## Problem\n")
        parts.append(resp.problem.statement.strip())
        parts.append("")

    parts.append(_pattern_block(resp.pattern))

    if resp.approaches:
        parts.append(_approaches_block(resp.approaches))

    if resp.complexity.time or resp.complexity.space:
        parts.append(_complexity_block(resp.complexity))

    if resp.examples:
        parts.append(_examples_block(resp.examples))

    if resp.edge_cases:
        parts.append(_edge_cases_block(resp.edge_cases))

    if not resp.verify.skipped:
        parts.append(_verify_block(resp.verify))

    if resp.follow_ups:
        parts.append(_follow_ups_block(resp.follow_ups))

    return "\n".join(p for p in parts if p).rstrip() + "\n"


# ---- section helpers ----------------------------------------------------
def _pattern_block(p: PatternMatch) -> str:
    lines = ["## Pattern", ""]
    lines.append(
        f"**Family:** `{p.family}` &nbsp;·&nbsp; **Confidence:** {p.confidence:.2f}"
    )
    if p.rationale:
        lines.append("")
        lines.append(f"_{p.rationale}_")
    lines.append("")
    return "\n".join(lines)


def _approaches_block(approaches: list[SolutionApproach]) -> str:
    # Order: brute → optimized → optimal so the user sees the
    # progression. Anything outside that triad falls to the end.
    order = {"brute": 0, "optimized": 1, "optimal": 2}
    ordered = sorted(approaches, key=lambda a: order.get(a.level, 99))

    lines = ["## Approaches", ""]
    for a in ordered:
        heading = a.level.capitalize() if a.level else "Approach"
        lines.append(f"### {heading}")
        if a.summary:
            lines.append("")
            lines.append(a.summary.strip())
        if a.reasoning:
            lines.append("")
            lines.append(a.reasoning.strip())
        if a.code.strip():
            lines.append("")
            lines.append(f"```{a.language or 'python'}")
            lines.append(a.code.rstrip())
            lines.append("```")
        lines.append("")
    return "\n".join(lines)


def _complexity_block(c: ComplexityProof) -> str:
    lines = ["## Complexity", "", "| Metric | Value |", "| --- | --- |"]
    lines.append(f"| Time  | {c.time or '—'} |")
    lines.append(f"| Space | {c.space or '—'} |")
    if c.proof:
        lines.append("")
        lines.append(f"> {c.proof.strip()}")
    lines.append("")
    return "\n".join(lines)


def _examples_block(examples: list[dict]) -> str:
    lines = ["## Examples", ""]
    for ex in examples:
        label = ex.get("label") or "case"
        raw_in = ex.get("input", "").strip()
        raw_out = ex.get("expected_output") or ex.get("output", "")
        raw_out = raw_out.strip() if isinstance(raw_out, str) else str(raw_out)
        note = ex.get("note", "").strip() if isinstance(ex.get("note"), str) else ""
        line = f"- **{label.capitalize()}** — `{raw_in}` → `{raw_out}`"
        if note:
            line += f" _({note})_"
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


def _edge_cases_block(edges: list[str]) -> str:
    lines = ["## Edge cases", ""]
    for e in edges:
        lines.append(f"- {e}")
    lines.append("")
    return "\n".join(lines)


def _verify_block(v: VerifyResult) -> str:
    total = v.passed + v.failed
    if total == 0:
        return ""
    lines = ["## Verification", ""]
    summary = f"Passed **{v.passed}/{total}**"
    if v.failed:
        summary += f" · {v.failed} failed"
    if v.repair_attempts:
        summary += f" · {v.repair_attempts} repair attempt(s)"
    lines.append(summary)
    if v.errors:
        lines.append("")
        for err in v.errors[:6]:
            lines.append(f"- `{err}`")
    lines.append("")
    return "\n".join(lines)


def _follow_ups_block(items: list[str]) -> str:
    lines = ["## Follow-ups", ""]
    for f in items:
        lines.append(f"- {f}")
    lines.append("")
    return "\n".join(lines)


def make_follow_ups(problem: ProblemSpec, pattern: PatternMatch) -> list[str]:
    """A short canned list — keeps the chat going past the answer.

    The supervisor's Suggester agent generates better personalised
    follow-ups when wired up; this is the no-LLM default.
    """
    base = [
        "Walk me through the trace on the EDGE example.",
        "What changes if the input can have negative numbers?",
        "How would you adapt this if the array were streamed?",
    ]
    if pattern.family == "trees":
        return [
            "What if the tree is a BST — can you do better?",
            "How would you handle a skewed tree without blowing the stack?",
            "Can you do this iteratively without recursion?",
        ]
    if pattern.family == "graphs":
        return [
            "What if the graph has negative edges?",
            "How does this generalise to weighted graphs?",
            "Can you detect a cycle while you traverse?",
        ]
    if pattern.family == "dp":
        return [
            "Can you reduce the space to O(1) row?",
            "What's the top-down memoised version?",
            "How would you reconstruct the actual answer, not just its value?",
        ]
    return base
