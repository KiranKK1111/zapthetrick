"""Background research follow-up (perceived-speed R15).

For a complex request, after the initial answer has streamed we may keep
researching in the background and, ONLY when material additional findings turn
up, present them as a clearly-marked follow-up (R15.2); nothing is appended when
findings are immaterial (R15.3). Because it runs after the answer, it never
delays the first answer (R15.1). Off by default.
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

log = logging.getLogger(__name__)

FOLLOWUP_HEADER = "Additional findings"


def research_enabled() -> bool:
    # P5 #10: enabling default True. Runs strictly AFTER the answer streams, so
    # it never affects TTFT, and only appends when findings are material.
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.perceived, "background_research", True))
    except Exception:  # noqa: BLE001
        return False


async def maybe_followup(
    question: str,
    answer: str,
    *,
    research: Callable[[str, str], Awaitable[Any]],
    is_material: Callable[[Any, str], bool],
    format_findings: Callable[[Any], str] | None = None,
) -> str | None:
    """Run the post-answer research and return a marked follow-up string, or
    None when disabled / immaterial / on error. Caller invokes this AFTER the
    answer stream completes so it never affects TTFT."""
    if not research_enabled():
        return None
    try:
        findings = await research(question, answer)
    except Exception as exc:  # noqa: BLE001 — research is best-effort
        log.info("background research failed (%s) — no follow-up", exc)
        return None
    if not findings:
        return None
    try:
        material = bool(is_material(findings, answer))
    except Exception:  # noqa: BLE001
        material = False
    if not material:
        return None
    body = format_findings(findings) if format_findings else str(findings)
    return f"**{FOLLOWUP_HEADER}**\n\n{body}".strip()


__all__ = ["maybe_followup", "research_enabled", "FOLLOWUP_HEADER"]
