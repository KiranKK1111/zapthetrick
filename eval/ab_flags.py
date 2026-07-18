"""A/B a feature flag through the eval harness (gap G11).

The new brain flags (understanding / semantic routing / synthesis / …) are
fail-open and default off, but we had no evidence they actually improve accuracy.
This is the measurement: run the golden set once with the flag OFF and once ON,
then `compare_reports` the two run reports for the per-category and overall
pass-rate + avg-score delta and a verdict.

Procedure (needs a provider key):

    # baseline
    RUN_EVAL_GATE=1 python -m eval.runner --out eval/reports/off.json
    # flip the flag in config.yaml (e.g. understanding.enabled: true), then:
    RUN_EVAL_GATE=1 python -m eval.runner --out eval/reports/on.json
    python -c "import json,eval.ab_flags as a; \
        print(a.compare_reports(json.load(open('eval/reports/off.json')), \
                                json.load(open('eval/reports/on.json'))))"

The comparison itself is pure + deterministic (no provider), so it's unit-tested.
"""
from __future__ import annotations

# Deltas smaller than this (pass-rate / avg-score) are treated as noise.
_EPS = 0.02


def _rates(report: dict) -> dict:
    pc = (report.get("summary") or {}).get("per_category") or {}
    out = {}
    for cat, c in pc.items():
        total = max(int(c.get("total") or 0), 1)
        out[cat] = {
            "pass_rate": (c.get("passed") or 0) / total,
            "avg_score": float(c.get("avg_score") or 0.0),
            "total": int(c.get("total") or 0),
        }
    return out


def _overall(report: dict) -> tuple[float, int]:
    pc = (report.get("summary") or {}).get("per_category") or {}
    passed = sum(int(c.get("passed") or 0) for c in pc.values())
    total = sum(int(c.get("total") or 0) for c in pc.values())
    return (passed / total if total else 0.0, total)


def compare_reports(off: dict, on: dict) -> dict:
    """Delta of a flag-ON run vs a flag-OFF baseline. Returns overall +
    per-category pass-rate/avg-score deltas and a verdict. Pure; never raises."""
    try:
        off_rate, n = _overall(off)
        on_rate, _ = _overall(on)
        overall_delta = round(on_rate - off_rate, 4)

        off_cats, on_cats = _rates(off), _rates(on)
        per_category = {}
        regressions = []
        improvements = []
        for cat in sorted(set(off_cats) | set(on_cats)):
            o = off_cats.get(cat, {})
            n_ = on_cats.get(cat, {})
            d_pass = round(n_.get("pass_rate", 0) - o.get("pass_rate", 0), 4)
            d_score = round(n_.get("avg_score", 0) - o.get("avg_score", 0), 4)
            per_category[cat] = {"pass_rate_delta": d_pass,
                                 "avg_score_delta": d_score}
            if d_pass <= -_EPS:
                regressions.append(cat)
            elif d_pass >= _EPS:
                improvements.append(cat)

        if overall_delta >= _EPS and not regressions:
            verdict = "improved"
        elif overall_delta <= -_EPS or regressions:
            verdict = "regressed"
        else:
            verdict = "neutral"
        return {
            "verdict": verdict,
            "overall_pass_rate_delta": overall_delta,
            "samples": n,
            "improved_categories": improvements,
            "regressed_categories": regressions,
            "per_category": per_category,
        }
    except Exception as exc:  # noqa: BLE001
        return {"verdict": "error", "error": str(exc)}


__all__ = ["compare_reports"]
