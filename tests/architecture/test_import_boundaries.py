"""Guardrail rule #13 — no "hidden coupling" (the highest-value guardrail).

The cross-package import graph between top-level app/ packages is frozen as a
baseline. A NEW edge (package A newly importing package B) fails the test until
it's added to import_baseline.json — forcing coupling to be a conscious, reviewed
decision rather than something that accretes silently.

Pure static AST scan — no imports of app code.
"""
from __future__ import annotations

from . import _scan
from ._allowlists import BASELINE_EDGES


def _current_edges() -> set[str]:
    return {f"{s} -> {d}" for s, d in _scan.cross_package_edges()}


def test_no_new_cross_package_coupling():
    current = _current_edges()
    new = current - BASELINE_EDGES
    assert not new, (
        f"New cross-package import edge(s) introduced:\n  "
        + "\n  ".join(sorted(new))
        + "\n\nRule #13 (no hidden coupling): packages couple only through "
        "declared, reviewed dependencies. If this coupling is intended, "
        "regenerate/extend tests/architecture/import_baseline.json deliberately "
        "(the diff is the record). Prefer the event bus / blackboard / stable "
        "interfaces over a direct import where possible."
    )


def test_baseline_not_stale():
    # Removed edges are safe, but keep the baseline honest so it stays a true map.
    current = _current_edges()
    gone = BASELINE_EDGES - current
    # Allow drift but cap it — a large gap means the baseline drifted from reality
    # and should be regenerated so the guardrail keeps real teeth.
    assert len(gone) <= 25, (
        f"{len(gone)} baseline import edges no longer exist — the baseline has "
        f"drifted from reality. Regenerate import_baseline.json so new-edge "
        f"detection stays meaningful."
    )
