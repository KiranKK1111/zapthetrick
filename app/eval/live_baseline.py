"""
Live-evaluation baseline + regression (live-conversational-intelligence R15).

Persists a known-good snapshot of the live decision-matrix metrics to a
committed JSON file (`app/eval/live_baseline.json`) and compares a current run
against it: a per-category or overall accuracy drop beyond `tolerance`, or a
false-answer-rate climb, is a regression. Mirrors the
`evaluation-and-reliability` `BaselineStore` pattern. No DB, no provider keys,
dev/CI-only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from app.eval.live_scenarios import live_metrics

_DEFAULT_PATH = Path(__file__).resolve().parent / "live_baseline.json"
DEFAULT_TOLERANCE = 0.02


@dataclass
class LiveRegressionReport:
    has_baseline: bool
    regressed: bool
    tolerance: float
    drops: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "has_baseline": self.has_baseline,
            "regressed": self.regressed,
            "tolerance": self.tolerance,
            "drops": self.drops,
        }


class LiveBaselineStore:
    """Load/save/compare the live decision-matrix baseline (committed JSON)."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path else _DEFAULT_PATH

    def load(self) -> dict | None:
        try:
            if not self.path.exists():
                return None
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            return None

    def save(self, metrics: dict) -> None:
        self.path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    def compare(self, current: dict,
                tolerance: float = DEFAULT_TOLERANCE) -> LiveRegressionReport:
        base = self.load()
        if not base:
            return LiveRegressionReport(has_baseline=False, regressed=False,
                                        tolerance=tolerance)
        drops: list[dict] = []

        b_overall = float(base.get("overall", {}).get("pass_rate", 0.0))
        c_overall = float(current.get("overall", {}).get("pass_rate", 0.0))
        if c_overall < b_overall - tolerance:
            drops.append({"scope": "overall", "baseline": b_overall,
                          "current": c_overall, "delta": round(c_overall - b_overall, 3)})

        b_cats = base.get("per_category", {})
        c_cats = current.get("per_category", {})
        for cat, b in b_cats.items():
            bp = float(b.get("pass_rate", 0.0))
            cp = float(c_cats.get(cat, {}).get("pass_rate", 0.0))
            if cp < bp - tolerance:
                drops.append({"scope": f"category:{cat}", "baseline": bp,
                              "current": cp, "delta": round(cp - bp, 3)})

        b_fa = float(base.get("false_answer_rate", 0.0))
        c_fa = float(current.get("false_answer_rate", 0.0))
        if c_fa > b_fa + tolerance:
            drops.append({"scope": "false_answer_rate", "baseline": b_fa,
                          "current": c_fa, "delta": round(c_fa - b_fa, 3)})

        return LiveRegressionReport(has_baseline=True, regressed=bool(drops),
                                    tolerance=tolerance, drops=drops)


def check_regression(tolerance: float = DEFAULT_TOLERANCE,
                     path: str | Path | None = None) -> tuple[dict, LiveRegressionReport]:
    metrics = live_metrics()
    rep = LiveBaselineStore(path).compare(metrics, tolerance)
    return metrics, rep


def _cli() -> int:
    """Standalone CI entrypoint: non-zero exit on a regression.

      python -m app.eval.live_baseline            # check against baseline
      python -m app.eval.live_baseline --update   # (re)establish the baseline
    """
    import sys
    update = "--update" in sys.argv[1:]
    store = LiveBaselineStore()
    metrics = live_metrics()
    if update:
        store.save(metrics)
        print("live baseline updated:", json.dumps(metrics["overall"]))
        return 0
    rep = store.compare(metrics)
    if not rep.has_baseline:
        print("no live baseline — report only. overall:", json.dumps(metrics["overall"]))
        print("run with --update to establish the baseline.")
        return 0
    if rep.regressed:
        print("LIVE REGRESSION detected:")
        for d in rep.drops:
            print(f"  {d['scope']}: {d['baseline']} -> {d['current']} (delta {d['delta']})")
        return 1
    print("no live regression. overall:", json.dumps(metrics["overall"]))
    return 0


if __name__ == "__main__":   # pragma: no cover
    raise SystemExit(_cli())


__all__ = ["LiveBaselineStore", "LiveRegressionReport", "check_regression",
           "DEFAULT_TOLERANCE"]
