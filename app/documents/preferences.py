"""Document-generation memory — Phase 7 (#25 / third-doc #3).

Remembers the user's document preferences — default audience/persona, branding
(logo, colors, header/footer, confidentiality), preferred format, and whether to
auto-enrich (TOC/glossary) — so they don't restate them each time. Single-user +
local: stored as one JSON row in the generic `llm_settings` key/value table
(no new schema). Fail-open everywhere — missing/broken prefs → empty defaults.
"""
from __future__ import annotations

import json
import logging

log = logging.getLogger(__name__)

_KEY = "doc_prefs"


async def get_preferences() -> dict:
    """The stored document preferences, or ``{}``. Never raises."""
    try:
        from storage.db import get_session_factory
        from storage.models import LLMSetting
        factory = get_session_factory()
        if factory is None:
            return {}
        async with factory() as s:
            row = await s.get(LLMSetting, _KEY)
            if row and row.value:
                data = json.loads(row.value)
                return data if isinstance(data, dict) else {}
    except Exception as exc:  # noqa: BLE001
        log.debug("get_preferences failed (non-fatal): %s", exc)
    return {}


async def save_preferences(prefs: dict) -> bool:
    """Upsert the document preferences. Returns success; never raises."""
    try:
        from storage.db import get_session_factory
        from storage.models import LLMSetting
        factory = get_session_factory()
        if factory is None:
            return False
        payload = json.dumps(prefs if isinstance(prefs, dict) else {})
        async with factory() as s:
            row = await s.get(LLMSetting, _KEY)
            if row is not None:
                row.value = payload
            else:
                s.add(LLMSetting(key=_KEY, value=payload))
            await s.commit()
        return True
    except Exception as exc:  # noqa: BLE001
        log.debug("save_preferences failed (non-fatal): %s", exc)
        return False


def export_settings_from_prefs(prefs: dict):
    """Build an :class:`ExportSettings` from stored branding prefs (empty when
    unset → no branding applied)."""
    from app.documents.model import ExportSettings
    b = (prefs or {}).get("branding", {}) if isinstance(prefs, dict) else {}
    if not isinstance(b, dict):
        b = {}
    return ExportSettings(
        header=str(b.get("header", "")),
        footer=str(b.get("footer", "")),
        logo_url=str(b.get("logo_url", "")),
        primary_color=str(b.get("primary_color", "")),
        confidentiality=str(b.get("confidentiality", "")),
        author=str(b.get("author", "")),
    )


__all__ = ["get_preferences", "save_preferences", "export_settings_from_prefs"]
