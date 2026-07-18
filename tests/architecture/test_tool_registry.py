"""Guardrail rule #4 — no "tool explosion".

Tools enter through the single self-registering registry (a `Tool` dataclass:
name/description/input_schema/handler), not hardcoded ad-hoc dispatch. This
locks the registry contract so the "one dispatch surface" invariant can't drift.
"""
from __future__ import annotations

from . import _scan

_REGISTRY = _scan.APP_ROOT / "tools" / "registry.py"
_REQUIRED_TOOL_FIELDS = {"name", "description", "input_schema", "handler"}
_REQUIRED_FUNCS = {"register", "get", "all_tools"}


def test_tool_dataclass_contract():
    fields = _scan.dataclass_fields(_REGISTRY, "Tool")
    assert fields, "app/tools/registry.py must define a `Tool` dataclass."
    missing = _REQUIRED_TOOL_FIELDS - fields
    assert not missing, (
        f"`Tool` is missing required field(s) {sorted(missing)}. Every tool must "
        f"declare name/description/input_schema/handler so the registry stays the "
        f"uniform surface (rule #4)."
    )


def test_registry_exposes_registration_api():
    funcs = _scan.module_level_functions(_REGISTRY)
    missing = _REQUIRED_FUNCS - funcs
    assert not missing, (
        f"app/tools/registry.py must expose {sorted(_REQUIRED_FUNCS)} — "
        f"missing {sorted(missing)}. Tools register through this single path; "
        f"there is no ad-hoc dispatch (rule #4)."
    )
