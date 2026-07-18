"""Loads the frozen guardrail allowlists / baselines from JSON."""
from __future__ import annotations

import json
import pathlib

_HERE = pathlib.Path(__file__).resolve().parent


def _load(name: str) -> dict:
    return json.loads((_HERE / name).read_text(encoding="utf-8"))


GOVERNANCE = _load("governance_allowlist.json")
IMPORT_BASELINE = _load("import_baseline.json")

ALLOWED_PACKAGES: set[str] = set(GOVERNANCE["packages"])
ALLOWED_AGENTS: set[str] = set(GOVERNANCE["agents"])
ALLOWED_MEMORY: set[str] = set(GOVERNANCE["memory_modules"])
BASELINE_EDGES: set[str] = set(IMPORT_BASELINE["edges"])
