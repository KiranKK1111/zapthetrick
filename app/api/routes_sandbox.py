"""Sandbox capability reporting (SandboxLangPack.md — Capability Matrix).

GET /api/sandbox/languages — which languages the sandbox knows how to run, and
which are actually EXECUTABLE on this host (toolchain installed). The model can
write a solution in any known language; verification only runs for the ones
reported `available`.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter

log = logging.getLogger("zapthetrick.sandbox")
router = APIRouter(prefix="/api/sandbox", tags=["sandbox"])


@router.get("/languages")
async def sandbox_languages() -> dict:
    """The full registry + per-language availability on this machine."""
    try:
        from app.sandbox import lang_registry as lr
        supported = lr.supported_ids()
        available = set(lr.available_languages())
        langs = [
            {"id": cid, "available": cid in available,
             "tool": lr.check_tool(cid)}
            for cid in supported
        ]
        return {
            "supported": supported,
            "available": sorted(available),
            "count": {"supported": len(supported), "available": len(available)},
            "languages": langs,
        }
    except Exception as exc:  # noqa: BLE001
        log.warning("sandbox languages failed: %s", exc)
        return {"supported": [], "available": [], "count": {}, "languages": []}
