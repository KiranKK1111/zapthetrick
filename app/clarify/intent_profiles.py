"""Intent Profile Registry (Architecture §4) — one behavior profile per intent.

Intent selects a *behavior profile*: config data (with code defaults here) that
says which P1 agents run, which graphs are consulted, which tools are allowed,
the response shape, whether the turn may generate a document, and the follow-up
suggestion style. Centralizing these per-intent decisions in one place means
behavior is tuned by editing data — not by touching scattered call sites.

The registry is OFF by default (`cfg.intent_profiles.enabled`). `resolve()`
always returns a profile (so callers can read one), but every wiring site is
gated on `enabled()` and falls back to today's behavior when off — so turning
the flag off is byte-for-byte the pre-registry pipeline. Config entries overlay
the code defaults field-by-field, so a YAML author can tweak one field of one
intent without restating the rest.

Tool names are the registry's real ids (`app/tools/registry.py`): `code_solver`
(the doc's "compute"), `code_search`, `web_search`, `resume_lookup`. Graph names
are `knowledge` (content KG), `memory` (memory graph), `code_kg` (code graph).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

# Canonical graph + tool ids the profiles reference (kept as module constants so
# a typo in a default is caught here, not silently at a call site).
GRAPH_KNOWLEDGE = "knowledge"
GRAPH_MEMORY = "memory"
GRAPH_CODE_KG = "code_kg"

TOOL_COMPUTE = "code_solver"     # Architecture §4 calls this "compute"
TOOL_CODE_SEARCH = "code_search"
TOOL_WEB_SEARCH = "web_search"
TOOL_RESUME_LOOKUP = "resume_lookup"


@dataclass(frozen=True)
class IntentProfile:
    """A resolved behavior profile. Immutable; config overrides produce a copy."""
    intent: str
    # P1 mesh agents to run for this intent (None = the default P1 set).
    agents: tuple[str, ...] | None = None
    # Graphs consulted this turn (subset of knowledge|memory|code_kg).
    graphs: tuple[str, ...] = ()
    # Allowed-tools whitelist (None = today's LLM heuristic; () = no tools).
    tools: tuple[str, ...] | None = None
    exclude_tools: tuple[str, ...] = ()
    max_tools: int = 2
    # Response shape hint (None = auto pick_shape); prose|table|code|steps|…
    response_shape: str | None = None
    # Depth override (None = config/user default); tldr|standard|deeper|exhaustive
    depth: str | None = None
    # Whether this intent may ever produce a downloadable document.
    doc_eligible: bool = True
    # Follow-up suggestion style: deepen|iterate|pivot|verify|extend|generic.
    suggestions: str = "generic"

    def consults(self, graph: str) -> bool:
        return graph in self.graphs


# Code-default registry — mirrors the table in Architecture §4. Keyed by the
# INTENT_* label strings from app/clarify/intent_pipeline.py.
DEFAULTS: dict[str, IntentProfile] = {
    "knowledge": IntentProfile(
        "knowledge", graphs=(GRAPH_KNOWLEDGE, GRAPH_MEMORY),
        response_shape="prose", doc_eligible=False, suggestions="deepen"),
    "code_generation": IntentProfile(
        "code_generation", agents=("persona", "critic"),
        graphs=(GRAPH_CODE_KG, GRAPH_MEMORY),
        tools=(TOOL_COMPUTE, TOOL_CODE_SEARCH),
        response_shape="code", doc_eligible=True, suggestions="iterate"),
    "comparison": IntentProfile(
        "comparison", graphs=(GRAPH_KNOWLEDGE,),
        response_shape="table", suggestions="pivot"),
    "debugging": IntentProfile(
        "debugging", graphs=(GRAPH_CODE_KG,),
        tools=(TOOL_COMPUTE, TOOL_CODE_SEARCH),
        response_shape="code", suggestions="verify"),
    "test_generation": IntentProfile(
        "test_generation", graphs=(GRAPH_CODE_KG, GRAPH_MEMORY),
        tools=(TOOL_CODE_SEARCH,), response_shape="code",
        doc_eligible=True, suggestions="iterate"),
    "documentation": IntentProfile(
        "documentation", graphs=(GRAPH_KNOWLEDGE, GRAPH_MEMORY),
        response_shape="prose", doc_eligible=True, suggestions="deepen"),
    "design": IntentProfile(
        "design", graphs=(GRAPH_MEMORY, GRAPH_KNOWLEDGE),
        response_shape="prose", suggestions="pivot"),
    "project_build": IntentProfile(
        "project_build", graphs=(GRAPH_CODE_KG, GRAPH_MEMORY),
        doc_eligible=True, suggestions="extend"),
    "chitchat": IntentProfile(
        "chitchat", graphs=(), tools=(), response_shape="prose",
        doc_eligible=False, suggestions="generic"),
    "archive": IntentProfile(
        "archive", graphs=(GRAPH_MEMORY,), tools=(TOOL_RESUME_LOOKUP,),
        doc_eligible=False, suggestions="generic"),
}

# Fallback for unknown/general intents — permissive but graph-aware.
FALLBACK = IntentProfile("general", graphs=(GRAPH_KNOWLEDGE, GRAPH_MEMORY))

# Config keys that may overlay a default. Anything else in a YAML profile is
# ignored (so a stray/renamed key can't crash resolution).
_OVERRIDABLE = frozenset({
    "agents", "graphs", "tools", "exclude_tools", "max_tools",
    "response_shape", "depth", "doc_eligible", "suggestions",
})
_SEQ_FIELDS = frozenset({"agents", "graphs", "tools", "exclude_tools"})
# Fields where None is a meaningful value (means "unset / use fallback").
_NULLABLE_SEQ = frozenset({"agents", "tools"})


def enabled() -> bool:
    """Master switch. Off → callers keep today's behavior."""
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.intent_profiles, "enabled", False))
    except Exception:  # noqa: BLE001 — fail-open to "off"
        return False


def _coerce(name: str, value):
    if name in _SEQ_FIELDS:
        if value is None:
            return None if name in _NULLABLE_SEQ else ()
        if isinstance(value, (list, tuple)):
            return tuple(str(x) for x in value)
        return (str(value),)
    if name == "max_tools":
        try:
            return int(value)
        except (TypeError, ValueError):
            return None  # dropped below
    if name == "doc_eligible":
        return bool(value)
    return value


def resolve(intent_type: str | None) -> IntentProfile:
    """The profile for an intent: code default (or FALLBACK) with any config
    overrides overlaid field-by-field. Never raises."""
    key = (intent_type or "").strip().lower()
    base = DEFAULTS.get(key, FALLBACK)
    try:
        from app.core.config_loader import cfg
        overrides = getattr(cfg.intent_profiles, "profiles", None) or {}
        raw = overrides.get(base.intent)
        if raw is None and key:
            raw = overrides.get(key)
        if isinstance(raw, dict) and raw:
            patch = {}
            for k, v in raw.items():
                if k not in _OVERRIDABLE:
                    continue
                cv = _coerce(k, v)
                if k == "max_tools" and cv is None:
                    continue
                patch[k] = cv
            if patch:
                return replace(base, **patch)
    except Exception:  # noqa: BLE001 — config overlay is best-effort
        pass
    return base


__all__ = [
    "IntentProfile", "DEFAULTS", "FALLBACK", "resolve", "enabled",
    "GRAPH_KNOWLEDGE", "GRAPH_MEMORY", "GRAPH_CODE_KG",
    "TOOL_COMPUTE", "TOOL_CODE_SEARCH", "TOOL_WEB_SEARCH", "TOOL_RESUME_LOOKUP",
]
