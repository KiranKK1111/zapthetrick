"""Per-run agent metrics (Phase 9, report #12/#29).

`RunMetrics` accumulates the observable facts of one chat agent-run as its
events stream by (tool calls, repair rounds, errors, verification, latency,
output tokens), then serializes to a dict for the SSE `metrics` frame + the
`agent_runs.output_summary` ledger column. `aggregate_runs` rolls a
conversation's ledger rows into totals for the metrics view.

Deterministic + dependency-free so it's trivially testable.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field


def est_tokens(text: str) -> int:
    """Rough token estimate (chars/4), matching the engine's estimator."""
    return max(0, len(text or "") // 4)


@dataclass
class RunMetrics:
    kind: str = "edit"               # build | edit
    tool_calls: int = 0
    rounds: int = 1
    errors: int = 0
    verify_ok: bool | None = None
    verify_attempted: bool = False
    goal_passed: bool | None = None
    confidence: str | None = None    # high | medium | low
    model: str | None = None
    out_tokens: int = 0              # estimated output tokens (final + results)
    duration_ms: int = 0
    success: bool = False
    todos_total: int = 0
    todos_completed: int = 0
    _t0: float = field(default_factory=time.monotonic, repr=False)

    def on_event(self, evt: dict) -> None:
        """Fold one streamed agent event into the running metrics."""
        et = evt.get("type")
        if et == "tool_call":
            self.tool_calls += 1
        elif et == "tool_result":
            self.out_tokens += est_tokens(str(evt.get("result") or ""))
        elif et == "goal_round":
            self.rounds = max(self.rounds, int(evt.get("round") or self.rounds))
        elif et == "goal_done":
            self.goal_passed = bool(evt.get("passed"))
            self.rounds = int(evt.get("rounds") or self.rounds)
        elif et == "goal_eval" and evt.get("verify"):
            self.verify_attempted = True
            self.verify_ok = bool(evt.get("passed"))
        elif et == "error":
            self.errors += 1
        elif et == "final":
            self.out_tokens += est_tokens(str(evt.get("message") or ""))
            self.success = True
        elif et == "model" and evt.get("model"):
            self.model = str(evt.get("model"))
        elif et == "todo":
            # Latest checklist state wins (the agent re-sends the full list).
            self.todos_total = int(evt.get("total") or len(evt.get("todos") or []))
            self.todos_completed = int(evt.get("done") or sum(
                1 for t in (evt.get("todos") or [])
                if isinstance(t, dict) and t.get("status") == "completed"))

    def finalize(self, *, confidence: str | None = None) -> "RunMetrics":
        self.duration_ms = int((time.monotonic() - self._t0) * 1000)
        if confidence:
            self.confidence = confidence
        if self.errors and self.goal_passed is not True:
            self.success = self.success and self.errors == 0
        return self

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "tool_calls": self.tool_calls,
            "rounds": self.rounds,
            "errors": self.errors,
            "verify_ok": self.verify_ok,
            "verify_attempted": self.verify_attempted,
            "goal_passed": self.goal_passed,
            "confidence": self.confidence,
            "model": self.model,
            "out_tokens": self.out_tokens,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "todos_total": self.todos_total,
            "todos_completed": self.todos_completed,
        }


def aggregate_runs(runs: list[dict]) -> dict:
    """Roll a conversation's agent-run ledger rows (each an `output_summary`
    dict + top-level tokens/status/duration) into totals for the metrics view."""
    n = len(runs)
    if not n:
        return {"runs": 0}
    total_tokens = sum(int(r.get("tokens") or 0) for r in runs)
    total_ms = sum(int((r.get("output_summary") or {}).get("duration_ms") or 0)
                   for r in runs)
    total_tools = sum(int((r.get("output_summary") or {}).get("tool_calls") or 0)
                      for r in runs)
    successes = sum(1 for r in runs if (r.get("status") == "ok"))
    return {
        "runs": n,
        "successes": successes,
        "success_rate": round(successes / n, 2),
        "total_tokens": total_tokens,
        "total_duration_ms": total_ms,
        "avg_duration_ms": total_ms // n if n else 0,
        "total_tool_calls": total_tools,
    }


# ── Health-dashboard counters (Phase 1 #6) ─────────────────────────────────
# Lightweight, thread-safe, bounded in-process counters that feed the three
# fields the roadmap's health dashboard was missing: router cost, retrieval
# relevance, and verifier failure rate. Each recorder is fail-open (metrics must
# never break a turn); each snapshot also folds in an already-wired source
# (quota_manager for cost, decision_metrics for verification) so the fields are
# genuinely populated from live traffic even before a direct feed exists.
import threading as _threading

_HLOCK = _threading.Lock()
_router = {"calls": 0, "tokens": 0, "cost_usd": 0.0}
_retr: list[float] = []           # ring of recent retrieval relevance scores
_RETR_CAP = 500
_verify = {"ok": 0, "fail": 0}


def record_router_cost(*, tokens: int = 0, cost_usd: float = 0.0) -> None:
    """One routed LLM call's estimated cost (tokens and/or $)."""
    try:
        with _HLOCK:
            _router["calls"] += 1
            _router["tokens"] += max(0, int(tokens or 0))
            _router["cost_usd"] += max(0.0, float(cost_usd or 0.0))
    except Exception:  # noqa: BLE001
        pass


def record_retrieval_relevance(score: float) -> None:
    """One retrieval's mean top-k relevance score (0..1)."""
    try:
        s = float(score)
        with _HLOCK:
            _retr.append(s)
            if len(_retr) > _RETR_CAP:
                del _retr[0:len(_retr) - _RETR_CAP]
    except Exception:  # noqa: BLE001
        pass


def record_verify(ok: bool) -> None:
    """One verifier verdict (True = passed, False = failed/regenerated)."""
    try:
        with _HLOCK:
            _verify["ok" if ok else "fail"] += 1
    except Exception:  # noqa: BLE001
        pass


def router_cost_snapshot() -> dict:
    try:
        with _HLOCK:
            out = {"calls": _router["calls"], "est_tokens": _router["tokens"],
                   "est_cost_usd": round(_router["cost_usd"], 6)}
        # Fold in provider request usage (already tracked on every dispatch).
        try:
            from app.llm.quota_manager import quota_manager
            prov = quota_manager().snapshot()
            out["provider_requests_used"] = sum(
                int(p.get("used", 0) or 0) for p in prov if isinstance(p, dict))
            out["providers_tracked"] = len(prov)
        except Exception:  # noqa: BLE001
            pass
        return out
    except Exception:  # noqa: BLE001
        return {}


def retrieval_relevance_snapshot() -> dict:
    try:
        with _HLOCK:
            rows = list(_retr)
        n = len(rows)
        return {
            "samples": n,
            "avg_relevance": round(sum(rows) / n, 4) if n else None,
            "min_relevance": round(min(rows), 4) if n else None,
            "max_relevance": round(max(rows), 4) if n else None,
        }
    except Exception:  # noqa: BLE001
        return {}


def verifier_snapshot() -> dict:
    try:
        with _HLOCK:
            ok, fail = _verify["ok"], _verify["fail"]
        total = ok + fail
        out = {"verified": total, "failures": fail,
               "failure_rate": round(fail / total, 4) if total else 0.0}
        # Fold in artifact-validation outcomes (already fed on every doc turn).
        try:
            from app.obs.decision_metrics import snapshot as _dsnap
            arts = _dsnap().get("artifacts", {}) or {}
            av_total = sum(int(arts.get(k, 0) or 0)
                           for k in ("validated", "repaired", "degraded", "failed"))
            av_fail = int(arts.get("failed", 0) or 0) + int(arts.get("degraded", 0) or 0)
            if av_total:
                out["artifact_total"] = av_total
                out["artifact_failures"] = av_fail
                out["artifact_failure_rate"] = round(av_fail / av_total, 4)
        except Exception:  # noqa: BLE001
            pass
        return out
    except Exception:  # noqa: BLE001
        return {}


def reset_health_counters() -> None:
    with _HLOCK:
        _router["calls"] = _router["tokens"] = 0
        _router["cost_usd"] = 0.0
        _retr.clear()
        _verify["ok"] = _verify["fail"] = 0


__all__ = [
    "RunMetrics", "aggregate_runs", "est_tokens",
    "record_router_cost", "record_retrieval_relevance", "record_verify",
    "router_cost_snapshot", "retrieval_relevance_snapshot", "verifier_snapshot",
    "reset_health_counters",
]
