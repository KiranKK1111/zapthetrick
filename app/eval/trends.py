"""Self-benchmarking trend tracking (roadmap Phase 7 #10).

The leaderboard (`eval/leaderboard.run_all`) and the regression baseline
(`eval/baseline`) each produce a point-in-time score. Nothing persisted those
scores over time, so "is the system getting better or worse?" had no answer.

This appends each benchmark run to a small JSONL trend log and reports the
direction of travel (latest vs previous, latest vs first, rolling delta). Pure,
file-persisted, bounded, and fail-open — a trend-log error never breaks the run
that produced the score. Meant to be driven nightly by the maintenance loop, and
readable on demand via `GET /api/eval/trends`.
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import threading
import time

log = logging.getLogger("eval.trends")

_LOCK = threading.Lock()
_MAX_POINTS = 500          # ring size of the persisted trend log


def _store_path() -> pathlib.Path:
    override = os.environ.get("ZAPTHETRICK_TRENDS")
    if override:
        return pathlib.Path(override)
    return pathlib.Path.home() / ".zapthetrick" / "trends.jsonl"


def _read_points(path: pathlib.Path) -> list[dict]:
    try:
        if not path.exists():
            return []
        pts: list[dict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    pts.append(json.loads(line))
                except Exception:  # noqa: BLE001 — skip a corrupt line
                    continue
        return pts
    except Exception:  # noqa: BLE001
        return []


def _write_points(path: pathlib.Path, points: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(p, default=str) for p in points) + "\n",
        encoding="utf-8")


def record_point(metric: str, score: float | None, *,
                 detail: dict | None = None,
                 path: str | pathlib.Path | None = None) -> dict:
    """Append one benchmark point `{ts, metric, score, detail}` to the trend log
    (bounded). Returns the recorded point. Never raises."""
    point = {
        "ts": time.time(),
        "metric": str(metric),
        "score": (round(float(score), 4) if isinstance(score, (int, float))
                  and not isinstance(score, bool) else None),
        "detail": detail or {},
    }
    try:
        p = pathlib.Path(path) if path else _store_path()
        with _LOCK:
            points = _read_points(p)
            points.append(point)
            if len(points) > _MAX_POINTS:
                points = points[-_MAX_POINTS:]
            _write_points(p, points)
    except Exception as exc:  # noqa: BLE001
        log.info("trend record failed (%s)", exc)
    return point


def trend_report(metric: str | None = None,
                 path: str | pathlib.Path | None = None) -> dict:
    """Direction-of-travel report over the persisted points (optionally for one
    metric): latest, first, previous, and deltas. Fail-open."""
    try:
        p = pathlib.Path(path) if path else _store_path()
        with _LOCK:
            points = _read_points(p)
        if metric:
            points = [pt for pt in points if pt.get("metric") == metric]
        scored = [pt for pt in points
                  if isinstance(pt.get("score"), (int, float))]
        if not scored:
            return {"metric": metric, "points": len(points), "scored": 0}
        latest = scored[-1]
        first = scored[0]
        prev = scored[-2] if len(scored) >= 2 else None
        d_prev = (round(latest["score"] - prev["score"], 4) if prev else None)
        d_first = round(latest["score"] - first["score"], 4)
        if d_prev is None:
            direction = "flat"
        elif d_prev > 1e-9:
            direction = "improving"
        elif d_prev < -1e-9:
            direction = "regressing"
        else:
            direction = "flat"
        return {
            "metric": metric,
            "points": len(points),
            "scored": len(scored),
            "latest": latest["score"],
            "first": first["score"],
            "previous": prev["score"] if prev else None,
            "delta_vs_previous": d_prev,
            "delta_vs_first": d_first,
            "direction": direction,
        }
    except Exception as exc:  # noqa: BLE001
        return {"metric": metric, "error": str(exc)[:160]}


def run_and_record(*, path: str | pathlib.Path | None = None) -> dict:
    """Self-benchmark now and persist the point (Phase 7 #10). Runs the unified
    leaderboard, records its `overall` score, and returns the fresh report. This
    is what the nightly maintenance loop calls. Fail-open."""
    overall = None
    detail: dict = {}
    try:
        from app.eval.leaderboard import run_all
        board = run_all()
        overall = board.get("overall")
        detail = {"ran": board.get("ran"), "scored": board.get("scored")}
    except Exception as exc:  # noqa: BLE001
        detail = {"error": str(exc)[:160]}
    record_point("leaderboard.overall", overall, detail=detail, path=path)
    return trend_report("leaderboard.overall", path=path)


def reset(path: str | pathlib.Path | None = None) -> None:
    try:
        p = pathlib.Path(path) if path else _store_path()
        if p.exists():
            p.unlink()
    except Exception:  # noqa: BLE001
        pass


__all__ = ["record_point", "trend_report", "run_and_record", "reset"]
