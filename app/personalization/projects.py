"""Project context for a turn (Architecture §17).

When a conversation belongs to a project, the turn gains two things:

  * **project-level instructions** — standing directions for every chat in the
    project, injected as TRUSTED prompt content just below the user's own custom
    instructions (precedence: safety ▷ user instructions ▷ project instructions
    ▷ learned memory ▷ intent defaults), and
  * a **project-scoped knowledge graph** — the content KG accretes at the project
    level (`Project.metadata['kg']`) instead of per-conversation, so entities and
    relations learned in one project chat are available to the others.

Helpers here are small and fail-open: any lookup error yields "no project", so a
turn degrades to today's per-conversation behavior.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_INSTR_CAP = 4000


def frame_project_instructions(text: str | None) -> str:
    """Format a project's instructions as a TRUSTED, labelled prompt block that
    states its precedence. "" when there's nothing to inject."""
    t = (text or "").strip()[:_INSTR_CAP]
    if not t:
        return ""
    return (
        "This conversation belongs to a project with the following standing "
        "instructions. Apply them for every reply in this project UNLESS they "
        "conflict with the safety rules or the user's own instructions above "
        "(which take precedence). Treat them as trusted project context:\n" + t
    )


async def load_project_context(session, conversation_row) -> dict:
    """Return ``{"project_id": str|None, "instructions": str}`` for the project a
    conversation belongs to (empty when ungrouped). Never raises."""
    empty = {"project_id": None, "instructions": ""}
    try:
        pid = getattr(conversation_row, "project_id", None)
        if not pid:
            return empty
        from storage.models import Project
        proj = await session.get(Project, pid)
        if proj is None:
            return empty
        return {"project_id": str(proj.id),
                "instructions": (proj.instructions or "").strip()[:_INSTR_CAP]}
    except Exception as exc:  # noqa: BLE001 — project context is best-effort
        log.debug("project context lookup failed: %s", exc)
        return empty


__all__ = ["frame_project_instructions", "load_project_context"]
