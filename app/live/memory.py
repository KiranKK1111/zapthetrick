"""
Multi-level memory + rolling session summary
(live-conversational-intelligence R6).

Layers over the existing per-session `context_tracker` ring buffer:

  - **L1** — the last few utterances (recent detail).
  - **L2** — the current-topic turns (topic-scoped recall).
  - **L3** — a compact whole-interview Session_Summary, refreshed in the
             background so it never blocks the answer path.

`context_for` assembles candidate context lines; the actual token-budget /
compression is **deferred to the perceived-speed Context_Budget** (R6.3). The
summary refresh is **deterministic** (no LLM call), so it is safe to run in a
background task. Stored ON the tracker (in-process; no new registry/DB).
Disabled or on error the live path uses today's recent-Q+A context.
"""
from __future__ import annotations

_L1_TURNS = 4
_MAX_TOPICS = 12


class MultiLevelMemory:
    """L1/L2/L3 memory views over a per-session context tracker."""

    def __init__(self, tracker) -> None:
        self._t = tracker
        self._summary: str = ""

    def _turns(self) -> list:
        return list(getattr(self._t, "_turns", []))

    def l1(self, n: int = _L1_TURNS) -> list[str]:
        """Most recent N questions (oldest first)."""
        return [t.question for t in self._turns()[-n:] if getattr(t, "question", "")]

    def l2(self, topic: str) -> list:
        """Turns whose topic matches (or is related to) `topic`."""
        name = (topic or "").strip().lower()
        if not name:
            return []
        out = []
        for t in self._turns():
            tt = (getattr(t, "topic", "") or "").lower()
            if tt and (tt == name or tt in name or name in tt):
                out.append(t)
        return out

    def l3(self) -> str:
        return self._summary

    def set_summary(self, summary: str) -> None:
        self._summary = (summary or "").strip()

    def context_for(self, question: str, topic: str) -> list[str]:
        """Assemble context lines from L2 (current topic) + L1 (recent) with the
        L3 summary as a floor. Token budget is deferred to perceived-speed."""
        ctx: list[str] = []
        for t in self.l2(topic):
            q = getattr(t, "question", "")
            if q and q not in ctx:
                ctx.append(q)
        for q in self.l1():
            if q not in ctx:
                ctx.append(q)
        if self._summary:
            ctx.append("Summary: " + self._summary)
        return ctx


def for_tracker(tracker) -> MultiLevelMemory:
    """Return the MultiLevelMemory attached to a per-session context tracker,
    creating it lazily (stored on the tracker object — in-process, no DB)."""
    m = getattr(tracker, "_live_memory", None)
    if m is None:
        m = MultiLevelMemory(tracker)
        try:
            setattr(tracker, "_live_memory", m)
        except Exception:  # noqa: BLE001
            pass
    return m


def refresh_summary(memory: MultiLevelMemory) -> str:
    """Deterministic, non-blocking Session_Summary refresh (NO LLM call):
    topics covered + recent questions. Safe to call in a background task. Never
    raises — returns the prior summary on any error."""
    try:
        turns = memory._turns()  # noqa: SLF001 — same package
        topics: list[str] = []
        for tr in turns:
            tp = (getattr(tr, "topic", "") or "").strip()
            if tp and tp not in topics:
                topics.append(tp)
        recent_qs = [tr.question for tr in turns[-3:] if getattr(tr, "question", "")]
        parts: list[str] = []
        if topics:
            parts.append("Topics covered: " + ", ".join(topics[:_MAX_TOPICS]))
        if recent_qs:
            parts.append("Recent questions: " + "; ".join(recent_qs))
        summary = ". ".join(parts)
        memory.set_summary(summary)
        return summary
    except Exception:  # noqa: BLE001
        return memory.l3()
