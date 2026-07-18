"""Unified benchmark leaderboard (roadmap Phase 1 #1).

The per-subsystem benchmark suites already exist (behavior, semantic-intent,
live scenarios, synthetic scenarios). This ties them into ONE call that re-runs
every suite and reports comparable 0..1 scores plus an aggregate — the
"leaderboard every change re-runs" the roadmap asks for. Fail-open per suite: a
suite that needs a resource it doesn't have (embeddings, a corpus file) is
recorded with its error and skipped from the aggregate rather than failing the
whole board.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _extract_score(result: Any) -> float | None:
    """Find a normalized 0..1 score in a suite's result dict."""
    if not isinstance(result, dict):
        return None
    for k in ("accuracy", "pass_rate", "score"):
        v = result.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    ov = result.get("overall")
    if isinstance(ov, dict):
        for k in ("pass_rate", "accuracy", "score"):
            v = ov.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return float(v)
    return None


def run_all(*, embed_fn: Callable | None = None) -> dict:
    """Run every registered suite and aggregate. Returns
    ``{suites: {name: {score, detail|error}}, overall, ran, scored}``."""
    suites: dict[str, dict] = {}

    def _run(name: str, fn: Callable[[], Any]) -> None:
        try:
            res = fn()
            suites[name] = {"score": _extract_score(res), "detail": res}
        except Exception as exc:  # noqa: BLE001 — one bad suite never sinks the board
            suites[name] = {"score": None, "error": str(exc)[:160]}

    from app.eval import live_scenarios, live_synth
    _run("live_scenarios", live_scenarios.live_metrics)
    _run("live_synth", live_synth.synth_metrics)

    def _intent() -> Any:
        from app.eval import semantic_intent_bench
        return semantic_intent_bench.run(embed_fn)
    _run("semantic_intent", _intent)

    def _behavior() -> Any:
        from app.eval import behavior_bench
        return behavior_bench.run_corpus()
    _run("behavior", _behavior)

    scored = [s["score"] for s in suites.values()
              if isinstance(s.get("score"), (int, float))]
    overall = round(sum(scored) / len(scored), 4) if scored else None
    return {
        "suites": suites,
        "overall": overall,
        "ran": len(suites),
        "scored": len(scored),
    }


__all__ = ["run_all"]
