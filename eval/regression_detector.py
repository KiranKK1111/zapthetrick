"""Detect regressions across two run reports.

`baseline` is either:
  - a path to a previous `run-*.json` report, or
  - the literal string "last" (use the second-newest report under
    `eval/reports/`).

A regression is declared when:
  - any case that passed in baseline fails in the current run, OR
  - per-category pass rate drops by more than 5%, OR
  - average score for a category drops by more than 0.1.

Returns a short human-readable message (or "" if no regression).

Note: `current` is a `RunReport` from `eval.runner`, but we type-
annotate it as `Any` to avoid a circular import (runner imports
from this module).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_REPORT_DIR = Path(__file__).resolve().parent / "reports"


def detect_regression(current: Any, baseline: str) -> str:
    base_path = _resolve_baseline(baseline)
    if base_path is None:
        return ""
    try:
        base = json.loads(base_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return ""

    base_cases = {c["case_id"]: c for c in base.get("cases") or []}
    failures: list[str] = []

    for c in current.cases:
        prior = base_cases.get(c.case_id)
        if not prior:
            continue
        if prior.get("passed") and not c.passed:
            failures.append(f"{c.case_id} regressed (was pass, now fail)")

    # Category-level drift.
    cur_pc = current.summary.get("per_category") or {}
    base_pc = (base.get("summary") or {}).get("per_category") or {}
    for cat, cur in cur_pc.items():
        b = base_pc.get(cat) or {}
        if not b:
            continue
        cur_rate = cur["passed"] / max(cur["total"], 1)
        base_rate = (b.get("passed") or 0) / max(b.get("total") or 1, 1)
        if base_rate - cur_rate > 0.05:
            failures.append(
                f"{cat} pass rate dropped {base_rate:.2f} → {cur_rate:.2f}"
            )
        if (b.get("avg_score") or 0) - (cur.get("avg_score") or 0) > 0.1:
            failures.append(f"{cat} avg score dropped > 0.1")

    return "; ".join(failures)


def _resolve_baseline(spec: str) -> Path | None:
    if spec == "last":
        reports = sorted(_REPORT_DIR.glob("run-*.json"))
        return reports[-2] if len(reports) >= 2 else None
    p = Path(spec)
    return p if p.exists() else None


__all__ = ["detect_regression"]
