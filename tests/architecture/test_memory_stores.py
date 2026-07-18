"""Guardrail rule #3 — no "memory explosion".

Durable memory is exactly the working/episodic/semantic hierarchy under
app/memory/ (+ project brain + KG elsewhere) — NOT a new per-mode memory store
bolted on per feature. New modules under app/memory/ must be allowlisted.
"""
from __future__ import annotations

from . import _scan
from ._allowlists import ALLOWED_MEMORY

# The three durable layers the architecture is built on (Architecture Part D).
_REQUIRED_LAYERS = {"working", "episodic", "semantic"}


def test_no_unlisted_memory_modules():
    current = _scan.package_modules("memory")
    new = current - ALLOWED_MEMORY
    assert not new, (
        f"New module(s) {sorted(new)} under app/memory/ are not allowlisted. "
        f"Rule #3 (no memory explosion): one hierarchy, not per-mode stores. "
        f"If durable state genuinely belongs here, add it to "
        f"governance_allowlist.json['memory_modules'] with intent."
    )


def test_core_memory_layers_present():
    current = _scan.package_modules("memory")
    missing = _REQUIRED_LAYERS - current
    assert not missing, (
        f"Core durable memory layer(s) {sorted(missing)} missing from app/memory/. "
        f"The working -> episodic -> semantic hierarchy is load-bearing."
    )
