"""Per-turn trace (Architecture.md §14 — Observability).

A compact, structured record of what a turn actually did: model, difficulty,
latency, which tools/agents ran, which graphs were consulted, which suggestion
sources fired, and any degraded subsystems. Surfaced in the response envelope
(`meta.trace`, so it's persisted with the message) and emitted as one structured
log line per turn — so "fast" and "accurate" become measurable, not asserted.

Pure + fail-open: `build_trace` never raises; empty/absent facts are omitted.
"""
from __future__ import annotations

from collections import Counter


def build_trace(
    *,
    trace_id: str,
    model: str | None = None,
    difficulty: str | None = None,
    latency_ms: int | None = None,
    tools=None,
    memory_recalled: int = 0,
    kg_neighbors: int = 0,
    episodes: int = 0,
    suggestions=None,
    degraded=None,
    stages=None,
    capture: bool = True,
) -> dict:
    """Assemble the per-turn trace dict. `suggestions` are envelope suggestion
    objects (counted by `source`); `stages` is a {stage: ms} latency map.

    `capture` feeds the runtime flight recorder (Phase 1 #4) with this turn's
    (inputs → trace) so it can be replayed + diffed later; replay re-invokes
    `build_trace` with `capture=False` to avoid re-recording."""
    try:
        by_source = Counter(
            str(s.get("source")) for s in (suggestions or [])
            if isinstance(s, dict) and s.get("source"))
        graphs = {k: v for k, v in {
            "memory_recalled": int(memory_recalled or 0),
            "kg_neighbors": int(kg_neighbors or 0),
            "episodes": int(episodes or 0),
        }.items() if v}
        trace = {
            "id": trace_id,
            "model": model,
            "difficulty": difficulty,
            "latency_ms": latency_ms,
            "tools": list(tools or []),
            "graphs": graphs,
            "suggestions": dict(by_source),
            "degraded": list(degraded or []),
            "stages": {k: round(float(v), 1) for k, v in (stages or {}).items()},
        }
        # Drop empty/None fields for a compact record.
        result = {k: v for k, v in trace.items() if v not in (None, [], {}, {})}
        if capture:
            # Flight-recorder hook: store this turn's (inputs → trace) so it can
            # be replayed and diffed. Fail-open; never touches the returned trace.
            try:
                from app.obs import replay as _replay
                _replay.capture("trace", {
                    "trace_id": trace_id, "model": model,
                    "difficulty": difficulty, "latency_ms": latency_ms,
                    "tools": list(tools or []),
                    "memory_recalled": int(memory_recalled or 0),
                    "kg_neighbors": int(kg_neighbors or 0),
                    "episodes": int(episodes or 0),
                    "suggestions": [s for s in (suggestions or [])
                                    if isinstance(s, dict)],
                    "degraded": list(degraded or []),
                    "stages": dict(stages or {}),
                }, result)
            except Exception:  # noqa: BLE001
                pass
        return result
    except Exception:  # noqa: BLE001 — a trace must never break the turn
        return {"id": trace_id}


def replay_trace(inputs: dict) -> dict:
    """Replay handler for the flight recorder: re-runs `build_trace` on a
    recorded input set WITHOUT re-capturing, so record → replay → diff is a
    pure function of the recorded inputs (Phase 1 #4)."""
    data = dict(inputs or {})
    data.pop("capture", None)
    return build_trace(capture=False, **data)


__all__ = ["build_trace", "replay_trace"]
