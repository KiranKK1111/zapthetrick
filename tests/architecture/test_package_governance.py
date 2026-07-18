"""Guardrail rule #20 — Module governance / "no module #121".

A new top-level `app/` package must be a deliberate act: add it to
governance_allowlist.json (the git diff is the ADR seed). This prevents silent
architectural sprawl — the roadmap's "a new subsystem must justify itself" gate.

Pure static check: no imports, no keys, no models.
"""
from __future__ import annotations

from . import _scan
from ._allowlists import ALLOWED_PACKAGES


def test_no_unlisted_top_level_packages():
    current = _scan.top_level_packages()
    new = current - ALLOWED_PACKAGES
    assert not new, (
        f"New top-level app package(s) {sorted(new)} are not in the governance "
        f"allowlist. A new subsystem must justify itself (roadmap rule #20: "
        f"'Module #121'). If intentional, add it to "
        f"tests/architecture/governance_allowlist.json with a reason — that diff "
        f"is the architecture-decision record."
    )


def test_allowlist_has_no_stale_packages():
    # Removing a package is fine, but the allowlist should be kept honest so it
    # stays a true inventory. This flags allowlist entries that no longer exist.
    current = _scan.top_level_packages()
    stale = ALLOWED_PACKAGES - current
    assert not stale, (
        f"governance_allowlist.json lists package(s) {sorted(stale)} that no "
        f"longer exist under app/. Remove them to keep the inventory truthful."
    )
