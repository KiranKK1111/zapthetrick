"""Confidence band + provenance (Phase 8, report #57/#58).

Deterministic, offline. Turns the signals a run already produces (did the goal
pass? did build/tests pass? how many repair rounds? any errors? red-team risks?)
into a single CONFIDENCE BAND the UI shows as a dot, plus a human-readable
PROVENANCE list ("what this answer is based on") so the user can judge trust.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ConfidenceSignals:
    goal_passed: bool | None = None      # run_goal verdict (None = N/A)
    verify_attempted: bool = False       # a build system was found + run
    verify_ok: bool | None = None        # build/tests passed
    rounds: int = 1                      # repair rounds used
    had_error: bool = False              # an error event occurred
    unverified_claims: int = 0           # grounder hallucination count
    critic_issues: int = 0               # critic-flagged issues
    missing_sections: int = 0            # structured-output gaps
    high_risks: int = 0                  # red-team high-severity findings
    cross_verify_disagree: bool = False  # a different model judged it wrong (B1/B2)
    tests_added: int = 0                 # new tests written for the change (P2-5)
    untested_changes: int = 0            # added/changed symbols with no test (P2-5)


@dataclass
class ConfidenceResult:
    band: str                            # "high" | "medium" | "low"
    score: float                         # 0..1
    reasons: list[str] = field(default_factory=list)


def confidence_band(sig: ConfidenceSignals) -> ConfidenceResult:
    """Blend the signals into a 0..1 score and a band. Starts neutral-high and
    subtracts for each risk signal; verified success adds confidence back."""
    score = 0.7
    reasons: list[str] = []

    if sig.verify_attempted and sig.verify_ok:
        score += 0.2
        reasons.append("build & tests passed")
    elif sig.verify_attempted and sig.verify_ok is False:
        score -= 0.3
        reasons.append("build or tests still failing")

    if sig.goal_passed is True:
        score += 0.1
        reasons.append("completion criteria met")
    elif sig.goal_passed is False:
        score -= 0.2
        reasons.append("completion criteria not fully met")

    if sig.rounds >= 4:
        score -= 0.1
        reasons.append(f"took {sig.rounds} repair rounds")
    if sig.had_error:
        score -= 0.15
        reasons.append("errors occurred during the run")
    if sig.unverified_claims:
        score -= min(0.3, 0.1 * sig.unverified_claims)
        reasons.append(f"{sig.unverified_claims} unverified claim(s)")
    if sig.critic_issues:
        score -= min(0.2, 0.05 * sig.critic_issues)
        reasons.append(f"{sig.critic_issues} reviewer issue(s)")
    if sig.missing_sections:
        score -= min(0.15, 0.05 * sig.missing_sections)
        reasons.append(f"{sig.missing_sections} missing section(s)")
    if sig.high_risks:
        score -= min(0.3, 0.15 * sig.high_risks)
        reasons.append(f"{sig.high_risks} high-severity risk(s) flagged")
    if sig.cross_verify_disagree:
        score -= 0.2
        reasons.append("a second model disagreed on correctness")
    if sig.tests_added:
        score += min(0.12, 0.04 * sig.tests_added)
        reasons.append(f"added {sig.tests_added} test(s) for the change")
    if sig.untested_changes:
        score -= min(0.2, 0.05 * sig.untested_changes)
        reasons.append(f"{sig.untested_changes} change(s) without a test")

    score = max(0.0, min(1.0, score))
    band = "high" if score >= 0.75 else "medium" if score >= 0.45 else "low"
    if not reasons:
        reasons.append("no risk signals detected")
    return ConfidenceResult(band=band, score=round(score, 2), reasons=reasons)


def build_provenance(
    *,
    context_files: list[str] | None = None,
    changed_files: int | None = None,
    verify_summary: str | None = None,
    rag_sources: list[str] | None = None,
    web_sources: list[str] | None = None,
) -> list[str]:
    """A short 'based on' list for the answer (most-specific first)."""
    out: list[str] = []
    cf = context_files or []
    if cf:
        head = ", ".join(cf[:5])
        more = f" (+{len(cf) - 5} more)" if len(cf) > 5 else ""
        out.append(f"Read {len(cf)} project file(s): {head}{more}")
    if changed_files:
        out.append(f"Edited {changed_files} file(s) in your project")
    if verify_summary:
        out.append(f"Verification: {verify_summary.splitlines()[0][:120]}")
    for s in (rag_sources or [])[:5]:
        out.append(f"Knowledge: {s}")
    for s in (web_sources or [])[:5]:
        out.append(f"Web: {s}")
    return out


__all__ = ["ConfidenceSignals", "ConfidenceResult", "confidence_band",
           "build_provenance"]
