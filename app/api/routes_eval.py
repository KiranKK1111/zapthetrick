"""Eval / parity-measurement API (P2-2).

GET  /api/eval/offline   -> run the offline regression suite (no model keys)
GET  /api/eval/tasks     -> list the graded IT/CS task bank (+ optional filters)
POST /api/eval/run       -> grade the live routed model on the task bank,
                            scored by objective gates → a per-category /
                            per-difficulty scoreboard

This is the surface that turns Claude-parity from an estimate into a number:
the offline suite is a fast capability regression guard, and `/run` measures the
configured free/strong models on the same rubric so you can see exactly where
they trail (and pick the best free models). No auth (single-user; lock the VPS
port at the firewall — see report_2 §P2-12).
"""
from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/eval")


class EvalRunBody(BaseModel):
    categories: list[str] | None = None
    difficulties: list[str] | None = None
    label: str = "routed"
    concurrency: int = 4
    include_tasks: bool = True
    pass_threshold: float = 0.7


@router.get("/offline")
async def eval_offline() -> dict:
    """Run the offline deterministic regression suite (no provider keys)."""
    from app.eval.harness import default_suite, run_suite

    return run_suite(default_suite()).to_dict()


@router.get("/analytics")
async def eval_analytics() -> dict:
    """Read-only analytics/audit view (personalization-and-governance R5):
    aggregates latency / routing health / degradation from existing telemetry.
    Dev-only; no runtime effect."""
    from app.personalization.analytics import summary
    return summary()


@router.get("/scenarios")
async def eval_scenarios(tolerance: float = 0.02) -> dict:
    """Run the scenario coverage matrix (evaluation-and-reliability R1/R2) and
    compare against the committed baseline. No provider keys (R8.3)."""
    from app.eval.harness import run_suite
    from app.eval.scenarios import scenario_suite, category_metrics
    from app.eval.baseline import BaselineStore

    report = run_suite(scenario_suite())
    metrics = category_metrics(report)
    regression = BaselineStore().compare(metrics, tolerance).to_dict()
    return {
        "metrics": metrics,
        "regression": regression,
        "report": report.to_dict(),
    }


@router.get("/live")
async def eval_live(tolerance: float = 0.02, synthetic: int = 0) -> dict:
    """Run the live decision-matrix (live-conversational-intelligence R15) and
    compare against the committed live baseline. Deterministic — no audio, no
    provider keys; dev/CI-only with no runtime effect on the live path.

    `synthetic>0` (R27) also runs that many auto-annotated synthetic scenarios
    and reports a label-free metric-proxy sample alongside the labeled metrics
    (falls back to the hand-annotated run when unavailable)."""
    from app.eval.live_scenarios import live_metrics
    from app.eval.live_baseline import LiveBaselineStore

    metrics = live_metrics()
    regression = LiveBaselineStore().compare(metrics, tolerance).to_dict()
    out: dict = {"metrics": metrics, "regression": regression}
    if synthetic and synthetic > 0:
        try:
            from app.eval.live_synth import synth_metrics
            from app.eval.live_proxies import proxies_over
            out["synthetic"] = synth_metrics(synthetic)
            out["proxies"] = proxies_over([
                {"question": "What is Kafka?",
                 "answer": "Kafka is a distributed log that splits topics into partitions.",
                 "context": "Kafka is a distributed event-streaming log with partitions."},
            ])
        except Exception as exc:  # noqa: BLE001 — fall back to the labeled run
            out["synthetic_error"] = str(exc)
    return out


@router.get("/tasks")
async def eval_tasks(category: str | None = None,
                     difficulty: str | None = None) -> dict:
    """List the graded task bank, optionally filtered."""
    from app.eval.task_bank import categories as cats, task_bank

    tasks = task_bank(
        categories=[category] if category else None,
        difficulties=[difficulty] if difficulty else None,
    )
    return {
        "count": len(tasks),
        "categories": cats(),
        "tasks": [
            {
                "id": t.id,
                "category": t.category,
                "difficulty": t.difficulty,
                "prompt_preview": t.prompt[:160],
                "gates": [g.name for g in t.gates],
            }
            for t in tasks
        ],
    }


@router.post("/prompt-optimize")
async def prompt_optimize() -> dict:
    """Autonomous prompt optimization (P7 #5). Runs the built-in optimization
    suite through the champion-promotion loop (benchmark → pick best → gated
    promote) and returns the decision. Offline/deterministic — no provider keys.
    This is the reachable invoker the roadmap said 'autonomous' was missing."""
    from app.eval.prompt_optimizer import run_default_optimization
    return run_default_optimization()


@router.post("/shadow")
async def shadow_run(min_improvement: float = 0.01,
                     allow_regressions: int = 0) -> dict:
    """Shadow execution / A-B promotion (P7 #7). Runs a baseline vs candidate
    over the offline cases and returns the promotion decision (promote only when
    it measurably beats the baseline and regresses nothing). No provider keys."""
    from app.eval.shadow import run_default_shadow
    return run_default_shadow(min_improvement=min_improvement,
                              allow_regressions=allow_regressions)


@router.get("/trends")
async def eval_trends(metric: str = "leaderboard.overall") -> dict:
    """Self-benchmark trend report (P7 #10): direction of travel over the
    persisted benchmark points (latest vs previous/first)."""
    from app.eval.trends import trend_report
    return trend_report(metric or None)


@router.post("/trends/run")
async def eval_trends_run() -> dict:
    """Run the unified leaderboard now, persist the point, and return the fresh
    trend report (P7 #10). This is what the nightly maintenance loop calls."""
    from app.eval.trends import run_and_record
    return run_and_record()


@router.post("/run")
async def eval_run(body: EvalRunBody) -> dict:
    """Grade the live routed model on the task bank (objective gates)."""
    from app.eval.model_eval import make_llm_runner, run_task_bank
    from app.eval.task_bank import task_bank

    tasks = task_bank(categories=body.categories,
                      difficulties=body.difficulties)
    if not tasks:
        return {"error": "no tasks match the given filters", "total": 0}
    runner = make_llm_runner()
    board = await run_task_bank(
        tasks, runner, model=body.label,
        pass_threshold=body.pass_threshold,
        concurrency=max(1, min(body.concurrency, 8)),
    )
    return board.to_dict(include_tasks=body.include_tasks)


__all__ = ["router"]
