"""Analytics / audit view (personalization-and-governance R5).

`summary()` aggregates EXISTING telemetry — the `perceived-speed`
Latency_Observatory, the router penalties/health, and the
`evaluation-and-reliability` degradation events — into a read-only view. It is
aggregation only (no parallel measurement, R5.2), dev-build only, and has no
runtime effect (R5.3, Property 5). Never raises.
"""
from __future__ import annotations


def summary() -> dict:
    """A read-only snapshot of latency / routing health / degradation. Each
    source is best-effort; an unavailable source is simply omitted."""
    out: dict = {"latency": {}, "routing": {}, "degradation": {}}

    # 1) Latency observatory (perceived-speed).
    try:
        from app.perceived.observatory import observatory
        reqs = getattr(observatory, "_reqs", {})
        ttfts = [r.get("ttft_ms") for r in reqs.values()
                 if isinstance(r, dict) and r.get("ttft_ms")]
        out["latency"] = {
            "requests_tracked": len(reqs),
            "avg_ttft_ms": round(sum(ttfts) / len(ttfts), 1) if ttfts else None,
        }
    except Exception:  # noqa: BLE001
        pass

    # 2) Router penalties (intelligent-model-routing / freellmapi router).
    try:
        from app.llm.router import get_all_penalties
        pens = get_all_penalties()
        out["routing"] = {
            "penalized_models": len(pens),
            "top": pens[:5],
        }
    except Exception:  # noqa: BLE001
        pass

    # 3) Degradation events (evaluation-and-reliability).
    try:
        from app.quality.degrade import recent_events
        ev = recent_events(50)
        by_sub: dict[str, int] = {}
        for e in ev:
            s = e.get("subsystem", "?")
            by_sub[s] = by_sub.get(s, 0) + 1
        out["degradation"] = {"recent": len(ev), "by_subsystem": by_sub}
    except Exception:  # noqa: BLE001
        pass

    return out


__all__ = ["summary"]
