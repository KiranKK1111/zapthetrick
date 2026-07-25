"""Deep research / subagent isolation (vNext §8.6, Stage 11 Component E).

Hard questions need parallel depth without polluting the main context. §8.6 runs a
research fan-out under an ISOLATION contract:

  * a **planner** decomposes the question into 2–4 worker subtasks (sharing an L0
    prefix, so the cache is warm);
  * each **worker** runs ISOLATED — a fresh context and a large PRIVATE budget —
    and returns a DISTILLED result (≤ ~1k tokens) WITH citations, so the main
    context only ever sees the distilled findings, never the workers' scratch;
  * a **synthesizer** merges the distilled results (reasoning tier) with at most
    ONE follow-up wave — bounded, cited, fail-soft.

The same shape powers code-builder module analysis, the visual-QA critic, and
council judges. This module owns the deterministic orchestration — the isolation
contract, the plan/synthesize bounds, the one-follow-up-wave gate; the actual
worker LLM calls are injected seams. Pure + fail-open. Flag-gated
(`agents.deep_research`, default OFF → today's single-agent answer).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.agents, "deep_research", False))
    except Exception:  # noqa: BLE001
        return False


def _max_workers() -> int:
    try:
        from app.core.config_loader import cfg
        return max(2, min(4, int(getattr(cfg.agents, "deep_research_max_workers", 4) or 4)))
    except Exception:  # noqa: BLE001
        return 4


def _result_cap() -> int:
    try:
        from app.core.config_loader import cfg
        return int(getattr(cfg.agents, "deep_research_result_cap", 1000) or 1000)
    except Exception:  # noqa: BLE001
        return 1000


# --------------------------------------------------------------------------- #
# Isolation contract
# --------------------------------------------------------------------------- #
@dataclass
class IsolationContract:
    """The envelope every worker runs under."""
    fresh_context: bool = True          # workers never see the main context
    private_budget_tokens: int = 20000  # large private budget
    result_cap_tokens: int = 1000       # the distilled result the main sees
    require_citations: bool = True

    def to_dict(self) -> dict:
        return {"fresh_context": self.fresh_context,
                "private_budget_tokens": self.private_budget_tokens,
                "result_cap_tokens": self.result_cap_tokens,
                "require_citations": self.require_citations}


@dataclass
class DistilledResult:
    summary: str
    citations: list = field(default_factory=list)
    truncated: bool = False
    dropped: bool = False              # dropped for missing citations

    def to_dict(self) -> dict:
        return {"summary": self.summary, "citations": list(self.citations),
                "truncated": self.truncated, "dropped": self.dropped}


def _approx_tokens(text: str) -> int:
    return max(1, len(text or "") // 4)


def distill(summary: str, citations, *, contract: IsolationContract | None = None
            ) -> DistilledResult:
    """Enforce the isolation RESULT contract on a worker's output: truncate to
    the ≤1k-token cap and REQUIRE citations (an uncited result is dropped — the
    main context only accepts grounded findings). Never raises."""
    try:
        c = contract or IsolationContract(result_cap_tokens=_result_cap())
        cites = [str(x).strip() for x in (citations or []) if str(x).strip()]
        text = summary or ""
        if c.require_citations and not cites:
            return DistilledResult("", [], False, dropped=True)
        cap_chars = c.result_cap_tokens * 4
        truncated = len(text) > cap_chars
        if truncated:
            text = text[:cap_chars].rsplit(" ", 1)[0].rstrip() + " …"
        return DistilledResult(text.strip(), cites, truncated, dropped=False)
    except Exception:  # noqa: BLE001
        return DistilledResult(summary or "", list(citations or []), False, False)


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
@dataclass
class ResearchPlan:
    question: str
    workers: list = field(default_factory=list)   # subtask strings
    shared_prefix: str = ""                        # L0 prefix (warm cache)

    def to_dict(self) -> dict:
        return {"question": self.question, "workers": list(self.workers),
                "shared_prefix": self.shared_prefix}


def plan_research(question: str, *, planner_fn=None, max_workers: int | None = None
                  ) -> ResearchPlan:
    """Decompose a research question into 2–4 worker subtasks (sharing an L0
    prefix). `planner_fn(question) -> [subtasks]` is INJECTED (the planner LLM);
    a template fallback splits into distinct angles. Clamped to [2, max_workers].
    Never raises → a single-worker plan on error."""
    try:
        q = (question or "").strip()
        cap = _max_workers() if max_workers is None else max(2, min(4, max_workers))
        subtasks: list[str] = []
        if planner_fn is not None:
            try:
                subtasks = [str(s).strip() for s in (planner_fn(q) or []) if str(s).strip()]
            except Exception:  # noqa: BLE001
                subtasks = []
        if not subtasks:
            angles = ("current state and definitions", "key tradeoffs and alternatives",
                      "concrete examples and evidence", "risks and open questions")
            subtasks = [f"{q} — {a}" for a in angles]
        subtasks = subtasks[:cap]
        if len(subtasks) < 2:
            subtasks = (subtasks + [f"{q} — supporting evidence"])[:2]
        return ResearchPlan(question=q, workers=subtasks,
                            shared_prefix=f"Research task: {q}")
    except Exception:  # noqa: BLE001
        return ResearchPlan(question=question or "",
                            workers=[question or "", (question or "") + " — evidence"],
                            shared_prefix="")


# --------------------------------------------------------------------------- #
# Synthesis + one-follow-up-wave gate
# --------------------------------------------------------------------------- #
@dataclass
class Synthesis:
    answer: str
    citations: list = field(default_factory=list)
    followup_needed: bool = False
    followups: list = field(default_factory=list)
    workers_used: int = 0

    def to_dict(self) -> dict:
        return {"answer": self.answer, "citations": list(self.citations),
                "followup_needed": self.followup_needed,
                "followups": list(self.followups), "workers_used": self.workers_used}


def needs_followup(results, *, min_workers: int = 2,
                   followups_done: int = 0) -> "tuple[bool, list]":
    """Whether a SINGLE follow-up wave is warranted: too few grounded results, or
    a worker flagged a gap. Capped at ONE wave (followups_done ≥ 1 → never).
    Returns (needed, gap_questions). Never raises."""
    try:
        if followups_done >= 1:
            return False, []                    # one wave max
        grounded = [r for r in results or () if r and not getattr(r, "dropped", False)
                    and getattr(r, "citations", None)]
        gaps: list[str] = []
        for r in results or ():
            for g in (getattr(r, "gaps", None) or []):
                if str(g).strip():
                    gaps.append(str(g).strip())
        needed = len(grounded) < min_workers or bool(gaps)
        return needed, gaps[:2]
    except Exception:  # noqa: BLE001
        return False, []


def synthesize(question: str, results, *, synth_fn=None,
               followups_done: int = 0) -> Synthesis:
    """Merge the DISTILLED worker results into one cited answer (reasoning tier)
    with at most ONE follow-up wave. Drops uncited/dropped results; unions their
    citations. `synth_fn(question, results) -> str` is INJECTED; a deterministic
    concatenation is the fallback. Disabled → an empty synthesis. Never raises."""
    try:
        if not enabled():
            return Synthesis("", [], False, [], 0)
        good = [r for r in results or () if r and not getattr(r, "dropped", False)
                and (getattr(r, "summary", "") or "").strip()]
        cites: list = []
        for r in good:
            for c in (getattr(r, "citations", None) or []):
                if c not in cites:
                    cites.append(c)
        if synth_fn is not None:
            try:
                answer = str(synth_fn(question, good) or "")
            except Exception:  # noqa: BLE001
                answer = ""
        else:
            answer = "\n\n".join((getattr(r, "summary", "") or "") for r in good).strip()
        need, gaps = needs_followup(good, followups_done=followups_done)
        return Synthesis(answer=answer, citations=cites, followup_needed=need,
                         followups=gaps, workers_used=len(good))
    except Exception:  # noqa: BLE001
        return Synthesis("", [], False, [], 0)


__all__ = ["enabled", "IsolationContract", "DistilledResult", "distill",
           "ResearchPlan", "plan_research", "Synthesis", "needs_followup",
           "synthesize"]
