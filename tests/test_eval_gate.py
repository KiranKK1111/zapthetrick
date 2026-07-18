"""CI accuracy regression gate (Architecture §14 / gap G12.1).

The eval runner (`eval/runner.py`) has a `--baseline` regression check that
`eval/regression_detector.detect_regression` implements, and an
exit-on-regression path — but nothing called it, so a prompt/router change could
silently regress accuracy.

The gate is now covered two ways:

  * **Deterministic** (always runs, no provider): the tests below drive
    `detect_regression` with synthetic run reports + a temp baseline, verifying
    it flags a pass→fail regression and a category pass-rate drop, and stays
    silent when stable. This is the actual gate logic.
  * **Live** (opt-in): run the real golden set end-to-end with
    ``RUN_EVAL_GATE=1 python -m eval.runner --baseline last`` (needs a provider
    key). Kept as a CLI invocation rather than a pytest test so the suite stays
    fast + provider-free (no perpetual skip).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from eval.regression_detector import detect_regression


@dataclass
class _Case:
    case_id: str
    passed: bool


class _Report:
    """Minimal stand-in for eval.runner.RunReport (what detect_regression reads)."""
    def __init__(self, cases, per_category):
        self.cases = cases
        self.summary = {"per_category": per_category}


def _write_baseline(tmp_path, *, cases, per_category):
    p = tmp_path / "run-baseline.json"
    p.write_text(json.dumps({
        "cases": cases,
        "summary": {"per_category": per_category},
    }), encoding="utf-8")
    return str(p)


def test_no_baseline_means_no_regression():
    report = _Report([_Case("a", True)], {"code": {"passed": 1, "total": 1}})
    # a non-existent path resolves to None → empty (can't regress vs nothing)
    assert detect_regression(report, "/no/such/baseline.json") == ""


def test_flags_case_that_went_pass_to_fail(tmp_path):
    baseline = _write_baseline(
        tmp_path,
        cases=[{"case_id": "a", "passed": True}],
        per_category={"code": {"passed": 1, "total": 1, "avg_score": 1.0}},
    )
    report = _Report([_Case("a", False)],
                     {"code": {"passed": 0, "total": 1, "avg_score": 0.0}})
    msg = detect_regression(report, baseline)
    assert "a regressed" in msg


def test_flags_category_pass_rate_drop(tmp_path):
    baseline = _write_baseline(
        tmp_path,
        cases=[],
        per_category={"code": {"passed": 10, "total": 10, "avg_score": 1.0}},
    )
    report = _Report([], {"code": {"passed": 5, "total": 10, "avg_score": 1.0}})
    msg = detect_regression(report, baseline)
    assert "pass rate dropped" in msg


def test_flags_avg_score_drop(tmp_path):
    baseline = _write_baseline(
        tmp_path,
        cases=[],
        per_category={"code": {"passed": 10, "total": 10, "avg_score": 0.9}},
    )
    report = _Report([], {"code": {"passed": 10, "total": 10, "avg_score": 0.7}})
    assert "avg score dropped" in detect_regression(report, baseline)


def test_stable_run_has_no_regression(tmp_path):
    baseline = _write_baseline(
        tmp_path,
        cases=[{"case_id": "a", "passed": True}],
        per_category={"code": {"passed": 1, "total": 1, "avg_score": 0.95}},
    )
    report = _Report([_Case("a", True)],
                     {"code": {"passed": 1, "total": 1, "avg_score": 0.95}})
    assert detect_regression(report, baseline) == ""


def test_new_case_absent_from_baseline_is_ignored(tmp_path):
    baseline = _write_baseline(
        tmp_path,
        cases=[{"case_id": "a", "passed": True}],
        per_category={"code": {"passed": 1, "total": 1, "avg_score": 1.0}},
    )
    # 'b' is new (not in baseline) — it must NOT be reported as a case-level
    # regression (it has no prior to compare against).
    report = _Report([_Case("a", True), _Case("b", True)],
                     {"code": {"passed": 2, "total": 2, "avg_score": 1.0}})
    msg = detect_regression(report, baseline)
    assert "b regressed" not in msg and msg == ""
