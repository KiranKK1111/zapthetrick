"""Runtime capability registry (SeveralFeatures.md: Capability Discovery +
Capability Negotiation).

Answers "what can this deployment actually do RIGHT NOW?" before anything is
planned or promised: which document formats can be rendered (import probes),
whether the code sandbox / GPU / embedder / STT / web search are available,
and which tools are registered. Consumers: the clarifier/policy layer (never
promise an artifact we can't produce), the doc-format negotiation
(unavailable format → concrete alternative, no hallucinated deliverable), and
`GET /api/capabilities` for the UI.

Probes are cheap, cached with a short TTL, and individually fail-open — a
probe error reports that capability as unavailable, never raises.
"""
from __future__ import annotations

import importlib.util
import time
from dataclasses import dataclass, field
from typing import Any, Callable

_TTL_S = 30.0
_cache: dict[str, Any] | None = None
_cache_at: float = 0.0


# ── Self-registering capability contracts (SeveralFeatures.md: Capability
#    Registry + Contracts / roadmap Phase 4 #3) ───────────────────────────────
@dataclass(frozen=True)
class CapabilityContract:
    """A first-class, self-declared capability: what it does, what it needs, and
    what it produces — so planning can match a goal to a capability by contract
    instead of ad-hoc probing."""

    name: str
    summary: str = ""
    inputs: tuple = ()          # declared input kinds (e.g. ("text",))
    outputs: tuple = ()         # declared output kinds (e.g. ("pdf",))
    requires: tuple = ()        # capability keys it depends on (probe names)
    tags: tuple = ()

    def to_dict(self) -> dict:
        return {"name": self.name, "summary": self.summary,
                "inputs": list(self.inputs), "outputs": list(self.outputs),
                "requires": list(self.requires), "tags": list(self.tags)}


_CONTRACTS: dict[str, CapabilityContract] = {}


def register_capability(contract: CapabilityContract) -> CapabilityContract:
    """Register (or replace) a capability contract. Idempotent + never raises."""
    try:
        if contract and contract.name:
            _CONTRACTS[contract.name] = contract
    except Exception:  # noqa: BLE001
        pass
    return contract


def capability(name: str, *, summary: str = "", inputs: tuple = (),
               outputs: tuple = (), requires: tuple = (), tags: tuple = ()
               ) -> Callable:
    """Decorator that self-registers the decorated callable's capability
    contract on import — the "self-registering" part of #3."""
    def _wrap(fn: Callable) -> Callable:
        register_capability(CapabilityContract(
            name=name, summary=summary or (fn.__doc__ or "").strip().split("\n")[0],
            inputs=inputs, outputs=outputs, requires=requires, tags=tags))
        return fn
    return _wrap


def capability_contracts() -> list[dict]:
    """All registered contracts as dicts (for /api/capabilities + planning)."""
    return [c.to_dict() for c in _CONTRACTS.values()]


def satisfiable(name: str) -> bool:
    """True if a contract's `requires` are all currently available — pairs the
    self-declared contract with the live capability snapshot."""
    c = _CONTRACTS.get(name)
    if c is None:
        return False
    if not c.requires:
        return True
    snap = capability_snapshot()
    fmts = snap.get("document_formats") or {}
    for req in c.requires:
        if req in fmts:
            if not fmts.get(req):
                return False
        elif req == "sandbox":
            if not (snap.get("sandbox") or {}).get("available"):
                return False
        elif req == "gpu":
            if not (snap.get("gpu") or {}).get("available"):
                return False
        elif req == "web_search":
            if not snap.get("web_search"):
                return False
    return True

# Document format → the python package that renders it (import probe).
_FORMAT_DEPS: dict[str, str | None] = {
    "pdf": "fpdf",
    "docx": "docx",
    "pptx": "pptx",
    "xlsx": "openpyxl",
    "zip": None,           # stdlib zipfile — always available
    "7z": "py7zr",
    "md": None, "txt": None, "csv": None, "json": None,
}
# Preferred fallback when a requested format is unavailable.
_FORMAT_FALLBACK = {
    "pdf": "docx", "docx": "md", "pptx": "md", "xlsx": "csv",
    "7z": "zip",
}


def _has_module(name: str | None) -> bool:
    if name is None:
        return True
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:  # noqa: BLE001
        return False


def _probe_gpu() -> dict:
    """CUDA availability without importing torch when it isn't installed."""
    try:
        if importlib.util.find_spec("torch") is None:
            return {"available": False, "reason": "torch not installed"}
        import torch
        ok = bool(torch.cuda.is_available())
        return {
            "available": ok,
            "device": torch.cuda.get_device_name(0) if ok else None,
        }
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)[:80]}


def _probe_models() -> dict:
    """Local model warm-up states (embedder + STT chain)."""
    try:
        from app import models_warmup
        snap = models_warmup.snapshot()
        return {m.get("key"): m.get("stage") for m in snap.get("models", [])}
    except Exception:  # noqa: BLE001
        return {}


def _probe_tools() -> list[str]:
    try:
        from app.tools.registry import all_tools
        return sorted(t.name for t in all_tools())
    except Exception:  # noqa: BLE001
        return []


def _probe_sandbox() -> dict:
    """The dedicated sandbox + its actual isolation level (namespace on Linux
    with bubblewrap, rlimit on bare POSIX, subprocess on Windows) — so
    planning/negotiation know how strong execution isolation really is."""
    try:
        from app.sandbox import isolation_level
        return {"available": True, "isolation": isolation_level()}
    except Exception:  # noqa: BLE001
        return {"available": _has_module("app.agent_workspace.runner"),
                "isolation": "subprocess"}


def _probe_web_search() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.agents.enabled, "web", False))
    except Exception:  # noqa: BLE001
        return False


def _register_builtins() -> None:
    """Self-register the deployment's core capability contracts (#3). Idempotent."""
    if _CONTRACTS:
        return
    for c in (
        CapabilityContract("document_render", "Render a document to a binary format",
                           inputs=("text", "ir"), outputs=("pdf", "docx", "pptx", "xlsx"),
                           requires=("pdf",), tags=("documents",)),
        CapabilityContract("code_execution", "Run code in the isolated sandbox",
                           inputs=("code",), outputs=("result",),
                           requires=("sandbox",), tags=("execution",)),
        CapabilityContract("doc_transform", "Unified parse→transform→format→validate flow",
                           inputs=("upload", "text"), outputs=("markdown", "code"),
                           tags=("documents", "transform")),
        CapabilityContract("resume_template", "Render a resume into an ATS-safe template",
                           inputs=("markdown",), outputs=("markdown",),
                           tags=("documents", "resume")),
        CapabilityContract("web_search", "Search the web for fresh information",
                           outputs=("text",), requires=("web_search",), tags=("research",)),
    ):
        register_capability(c)


def refresh() -> dict:
    """Recompute the snapshot now (also used by tests)."""
    global _cache, _cache_at
    _register_builtins()
    formats = {fmt: _has_module(dep) for fmt, dep in _FORMAT_DEPS.items()}
    _cache = {
        "document_formats": formats,
        "sandbox": _probe_sandbox(),
        "gpu": _probe_gpu(),
        "models": _probe_models(),
        "tools": _probe_tools(),
        "web_search": _probe_web_search(),
        "contracts": capability_contracts(),
        "computed_at": time.time(),
    }
    _cache_at = time.time()
    return _cache


def capability_snapshot() -> dict:
    """The cached capability view (TTL-refreshed)."""
    if _cache is None or (time.time() - _cache_at) > _TTL_S:
        return refresh()
    return _cache


def available_document_formats() -> set[str]:
    snap = capability_snapshot()
    return {f for f, ok in (snap.get("document_formats") or {}).items() if ok}


def negotiate_format(fmt: str) -> tuple[bool, str | None, str]:
    """Capability negotiation for a requested document/archive format.

    Returns (available, alternative, reason):
      * available=True → proceed (alternative is None);
      * available=False → `alternative` is the closest format we CAN produce
        (following the fallback chain until an available one is found) and
        `reason` explains the substitution — so the caller degrades gracefully
        instead of failing or hallucinating a deliverable.
    """
    f = (fmt or "").strip().lower().lstrip(".")
    avail = available_document_formats()
    if f in avail:
        return True, None, "available"
    alt = _FORMAT_FALLBACK.get(f)
    seen = {f}
    while alt and alt not in avail and alt not in seen:
        seen.add(alt)
        alt = _FORMAT_FALLBACK.get(alt)
    if alt in avail:
        return False, alt, f"{f} renderer not installed; {alt} offered instead"
    return False, "md", f"{f} renderer not installed; markdown offered instead"


__all__ = ["capability_snapshot", "available_document_formats",
           "negotiate_format", "refresh",
           "CapabilityContract", "register_capability", "capability",
           "capability_contracts", "satisfiable"]
