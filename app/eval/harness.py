"""Evaluation harness (Phase 9, report #15).

A tiny benchmark runner: each `EvalCase` calls a (sync) function and checks the
result. `run_suite` executes them and returns an `EvalReport` (pass/fail +
per-case detail). `default_suite()` ships an OFFLINE suite over the app's
deterministic decision components (intent/domain classification, doc + agentic
intent detection, context ranking, confidence band, red-team parsing) so it
runs in CI with no provider keys and catches capability regressions.

Extend with LLM-backed cases by passing a `runner` whose `fn` calls a model
(mock it in tests). The harness itself stays provider-agnostic.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalCase:
    name: str
    fn: Callable[[], Any]                 # produces the value to check
    check: Callable[[Any], bool]          # True = pass
    category: str = "general"


@dataclass
class EvalResult:
    name: str
    category: str
    passed: bool
    detail: str = ""
    duration_ms: int = 0


@dataclass
class EvalReport:
    results: list[EvalResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return round(self.passed / self.total, 3) if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "results": [
                {"name": r.name, "category": r.category, "passed": r.passed,
                 "detail": r.detail, "duration_ms": r.duration_ms}
                for r in self.results
            ],
        }


def run_suite(cases: list[EvalCase]) -> EvalReport:
    """Run every case, capturing pass/fail + timing. Exceptions = fail."""
    report = EvalReport()
    for c in cases:
        t0 = time.monotonic()
        try:
            value = c.fn()
            ok = bool(c.check(value))
            detail = "" if ok else f"got: {value!r}"
        except Exception as exc:  # noqa: BLE001
            ok = False
            detail = f"raised {type(exc).__name__}: {exc}"
        report.results.append(EvalResult(
            name=c.name, category=c.category, passed=ok, detail=detail,
            duration_ms=int((time.monotonic() - t0) * 1000),
        ))
    return report


def default_suite() -> list[EvalCase]:
    """Offline benchmark over the deterministic decision components."""
    from app.chat.context_builder import rank_files
    from app.chat.trust import ConfidenceSignals, confidence_band
    from app.chat import redteam as rt
    from app.documents.detect import (
        detect_agentic_intent,
        explicit_doc_request,
    )
    from app.technical_pipeline.dispatcher import classify_domain

    cases: list[EvalCase] = [
        # ── domain classification ──
        EvalCase("domain:security",
                 lambda: classify_domain("prevent SQL injection and XSS"),
                 lambda v: v == "security", "classification"),
        EvalCase("domain:backend",
                 lambda: classify_domain("design a REST api with pagination"),
                 lambda v: v == "backend", "classification"),
        EvalCase("domain:system_design",
                 lambda: classify_domain("design a scalable system with sharding"),
                 lambda v: v == "system_design", "classification"),
        EvalCase("domain:generic",
                 lambda: classify_domain("tell me a joke"),
                 lambda v: v == "generic", "classification"),
        # ── agentic intent ──
        EvalCase("agentic:edit",
                 lambda: detect_agentic_intent("fix the login bug",
                                               has_archive=True),
                 lambda v: v["agentic"] and v["kind"] == "edit", "intent"),
        EvalCase("agentic:build",
                 lambda: detect_agentic_intent("build an app based on this",
                                               has_spec_doc=True),
                 lambda v: v["agentic"] and v["kind"] == "build", "intent"),
        EvalCase("agentic:readonly_qa",
                 lambda: detect_agentic_intent("explain this code",
                                               has_archive=True),
                 lambda v: not v["agentic"], "intent"),
        # ── document intent ──
        EvalCase("doc:pdf",
                 lambda: explicit_doc_request("give me a pdf document"),
                 lambda v: v[0] and v[1] == "pdf", "doc-intent"),
        EvalCase("doc:none",
                 lambda: explicit_doc_request("how do I parse a pdf in python"),
                 lambda v: not v[0], "doc-intent"),
        # ── context ranking ──
        EvalCase("context:rank_login",
                 lambda: rank_files(
                     [("src/auth/login.py", "def login(): pass"),
                      ("src/utils/helpers.py", "def slug(): pass")],
                     "fix the login bug"),
                 lambda v: v and v[0].path == "src/auth/login.py", "context"),
        # ── confidence band ──
        EvalCase("confidence:high",
                 lambda: confidence_band(ConfidenceSignals(
                     goal_passed=True, verify_attempted=True, verify_ok=True)),
                 lambda v: v.band == "high", "trust"),
        EvalCase("confidence:low",
                 lambda: confidence_band(ConfidenceSignals(
                     goal_passed=False, verify_attempted=True, verify_ok=False,
                     had_error=True, high_risks=2)),
                 lambda v: v.band == "low", "trust"),
        # ── red-team parsing ──
        EvalCase("redteam:parse",
                 lambda: rt._parse(
                     '{"risks":[{"severity":"high","area":"sec","issue":"x"}]}'),
                 lambda v: len(v) == 1 and v[0].severity == "high", "trust"),
    ]
    return cases


__all__ = ["EvalCase", "EvalResult", "EvalReport", "run_suite", "default_suite"]
