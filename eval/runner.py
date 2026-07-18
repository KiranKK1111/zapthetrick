"""Eval runner — load cases, run them, grade, write a report.

Usage:
    python -m eval.runner                 # run every case under eval/datasets
    python -m eval.runner --filter dsa    # only the dsa category
    python -m eval.runner --baseline last # diff against the previous run

The runner is intentionally library-agnostic — each case declares
which "system under test" to call (chat / agents / dsa). New SUTs
get added to `_SUTS` below.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .graders import grade_case
from .regression_detector import detect_regression


log = logging.getLogger(__name__)


@dataclass
class CaseRun:
    case_id: str
    category: str
    passed: bool
    score: float
    latency_ms: int
    detail: str = ""
    grader_components: dict = field(default_factory=dict)


@dataclass
class RunReport:
    started_at_ms: int
    finished_at_ms: int = 0
    cases: list[CaseRun] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


_DATASET_DIR = Path(__file__).resolve().parent / "datasets"
_REPORT_DIR = Path(__file__).resolve().parent / "reports"


async def _run_case(case: dict) -> CaseRun:
    """Execute one case against the named SUT and grade the output."""
    t0 = time.monotonic()
    sut_name = (case.get("sut") or "dsa").lower()
    sut = _SUTS.get(sut_name)
    if sut is None:
        return CaseRun(
            case_id=str(case.get("case_id", "?")),
            category=case.get("category", "unknown"),
            passed=False,
            score=0.0,
            latency_ms=0,
            detail=f"unknown sut: {sut_name}",
        )
    try:
        output = await sut(case)
    except Exception as exc:  # noqa: BLE001
        return CaseRun(
            case_id=str(case.get("case_id", "?")),
            category=case.get("category", "unknown"),
            passed=False,
            score=0.0,
            latency_ms=int((time.monotonic() - t0) * 1000),
            detail=f"sut raised: {exc}",
        )

    grade = grade_case(case, output)
    return CaseRun(
        case_id=str(case.get("case_id", "?")),
        category=case.get("category", "unknown"),
        passed=grade.passed,
        score=grade.score,
        latency_ms=int((time.monotonic() - t0) * 1000),
        detail=grade.detail,
        grader_components=grade.components,
    )


# ---- SUT adapters ------------------------------------------------------
async def _sut_dsa(case: dict) -> str:
    """Run the DSA pipeline against the case's question."""
    from app.dsa import solve

    question = (case.get("input") or {}).get("question", "")
    collected: list[str] = []
    async for evt in solve(question):
        if evt.get("kind") == "markdown":
            collected.append(str(evt.get("text") or ""))
    return "".join(collected)


async def _sut_chat(case: dict) -> str:
    """Tiny chat SUT — calls the LLM directly with the case's prompt."""
    from app.core.llm_client import llm

    question = (case.get("input") or {}).get("question", "")
    msgs = [{"role": "user", "content": question}]
    chunks: list[str] = []
    try:
        async for c in llm.stream_chat(msgs):
            chunks.append(c)
    except Exception:
        return ""
    return "".join(chunks)


_SUTS = {
    "dsa": _sut_dsa,
    "chat": _sut_chat,
}


def load_cases(filter_category: str | None = None) -> list[dict]:
    """Walk `eval/datasets/` and return every parseable case."""
    out: list[dict] = []
    if not _DATASET_DIR.exists():
        return out
    for path in sorted(_DATASET_DIR.rglob("*.yaml")):
        try:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as exc:  # noqa: BLE001
            log.warning("eval: failed to load %s: %s", path, exc)
            continue
        if not isinstance(data, dict):
            continue
        if filter_category and data.get("category") != filter_category:
            continue
        out.append(data)
    return out


async def run(filter_category: str | None = None) -> RunReport:
    started_at_ms = int(time.time() * 1000)
    rep = RunReport(started_at_ms=started_at_ms)
    cases = load_cases(filter_category)
    for case in cases:
        run = await _run_case(case)
        rep.cases.append(run)
    rep.finished_at_ms = int(time.time() * 1000)
    rep.summary = _summarize(rep.cases)
    _write_report(rep)
    return rep


def _summarize(cases: list[CaseRun]) -> dict:
    by_cat: dict[str, list[CaseRun]] = {}
    for c in cases:
        by_cat.setdefault(c.category, []).append(c)
    out = {"total": len(cases), "passed": sum(1 for c in cases if c.passed)}
    out["per_category"] = {
        cat: {
            "total": len(items),
            "passed": sum(1 for c in items if c.passed),
            "avg_score": sum(c.score for c in items) / max(len(items), 1),
        }
        for cat, items in by_cat.items()
    }
    return out


def _write_report(rep: RunReport) -> None:
    _REPORT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"run-{rep.started_at_ms}.json"
    (
        _REPORT_DIR / name
    ).write_text(
        json.dumps(
            {
                "started_at_ms": rep.started_at_ms,
                "finished_at_ms": rep.finished_at_ms,
                "summary": rep.summary,
                "cases": [asdict(c) for c in rep.cases],
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _cli() -> int:
    parser = argparse.ArgumentParser(prog="eval.runner")
    parser.add_argument("--filter", default=None, help="restrict to one category")
    parser.add_argument("--baseline", default=None,
                        help="compare against this report path (or 'last')")
    args = parser.parse_args()

    rep = asyncio.run(run(args.filter))
    print(json.dumps(rep.summary, indent=2))

    if args.baseline:
        regression = detect_regression(rep, args.baseline)
        if regression:
            print(f"REGRESSION: {regression}", flush=True)
            return 2
    # Non-zero exit when zero cases passed — CI can gate on this.
    return 0 if rep.summary.get("passed", 0) > 0 else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
