"""Baseline + regression detection (evaluation-and-reliability R2/R8).

Persists a known-good snapshot of the scenario-matrix metrics to a committed
JSON file (`app/eval/baseline.json`) and compares a current run against it.
A per-category or overall pass-rate drop beyond `tolerance` is a Regression
(R2.2/R2.3); an absent baseline is report-only (R2.4). No DB, no provider keys.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULT_PATH = Path(__file__).resolve().parent / "baseline.json"
DEFAULT_TOLERANCE = 0.02       # allow 2% noise before flagging a regression


@dataclass
class RegressionReport:
    has_baseline: bool
    regressed: bool
    tolerance: float
    drops: list[dict] = field(default_factory=list)   # [{scope, baseline, current, delta}]

    def to_dict(self) -> dict:
        return {
            "has_baseline": self.has_baseline,
            "regressed": self.regressed,
            "tolerance": self.tolerance,
            "drops": self.drops,
        }


class BaselineStore:
    """Load/save/compare the scenario-matrix baseline (a committed JSON file)."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else _DEFAULT_PATH

    def load(self) -> dict | None:
        try:
            if not self.path.exists():
                return None
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — corrupt baseline → report-only (R2.4)
            return None

    def save(self, metrics: dict) -> None:
        """Persist the current metrics as the new baseline (done deliberately)."""
        self.path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    def compare(self, current: dict,
                tolerance: float = DEFAULT_TOLERANCE) -> RegressionReport:
        """Flag any overall/per-category pass-rate drop beyond `tolerance`."""
        base = self.load()
        if not base:
            return RegressionReport(has_baseline=False, regressed=False,
                                    tolerance=tolerance)
        drops: list[dict] = []

        # Overall pass rate.
        b_overall = float(base.get("overall", {}).get("pass_rate", 0.0))
        c_overall = float(current.get("overall", {}).get("pass_rate", 0.0))
        if c_overall < b_overall - tolerance:
            drops.append({"scope": "overall", "baseline": b_overall,
                          "current": c_overall, "delta": round(c_overall - b_overall, 3)})

        # Per-category pass rates.
        b_cats = base.get("per_category", {})
        c_cats = current.get("per_category", {})
        for cat, b in b_cats.items():
            bp = float(b.get("pass_rate", 0.0))
            cp = float(c_cats.get(cat, {}).get("pass_rate", 0.0))
            if cp < bp - tolerance:
                drops.append({"scope": f"category:{cat}", "baseline": bp,
                              "current": cp, "delta": round(cp - bp, 3)})

        # Error-rate climbs (false-ask / misroute getting worse).
        for key in ("false_ask_rate", "misroute_rate"):
            b_rate = float(base.get(key, 0.0))
            c_rate = float(current.get(key, 0.0))
            if c_rate > b_rate + tolerance:
                drops.append({"scope": key, "baseline": b_rate,
                              "current": c_rate, "delta": round(c_rate - b_rate, 3)})

        return RegressionReport(has_baseline=True, regressed=bool(drops),
                                tolerance=tolerance, drops=drops)


def run_matrix() -> dict:
    """Run the scenario matrix and return its category metrics (no keys)."""
    from app.eval.harness import run_suite
    from app.eval.scenarios import scenario_suite, category_metrics
    report = run_suite(scenario_suite())
    return category_metrics(report)


def check_regression(tolerance: float = DEFAULT_TOLERANCE,
                     path: str | Path | None = None) -> tuple[dict, RegressionReport]:
    """Run the matrix + compare to baseline. Returns (metrics, regression)."""
    metrics = run_matrix()
    rep = BaselineStore(path).compare(metrics, tolerance)
    return metrics, rep


def _cli() -> int:
    """Standalone CI entrypoint: non-zero exit on a detected regression (R8.2).

    Usage:
      python -m app.eval.baseline            # check against baseline
      python -m app.eval.baseline --update   # (re)establish the baseline
    """
    import sys
    update = "--update" in sys.argv[1:]
    store = BaselineStore()
    metrics = run_matrix()
    if update:
        store.save(metrics)
        print("baseline updated:", json.dumps(metrics["overall"]))
        return 0
    rep = store.compare(metrics)
    if not rep.has_baseline:
        print("no baseline — report only. overall:",
              json.dumps(metrics["overall"]))
        print("run with --update to establish the baseline.")
        return 0
    if rep.regressed:
        print("REGRESSION detected:")
        for d in rep.drops:
            print(f"  {d['scope']}: {d['baseline']} -> {d['current']} "
                  f"(delta {d['delta']})")
        return 1
    print("no regression. overall:", json.dumps(metrics["overall"]))
    return 0


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(_cli())


__all__ = ["BaselineStore", "RegressionReport", "run_matrix",
           "check_regression", "DEFAULT_TOLERANCE"]
