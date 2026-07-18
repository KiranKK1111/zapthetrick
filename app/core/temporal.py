"""Temporal / Multi-Horizon intelligence (roadmap Phase 3 #16).

Time as a first-class dimension of the shared cognitive state: classify a
request's planning HORIZON (immediate → long-term), spot relative-time
references ("yesterday", "before the offer"), and flag deadline sensitivity.
Deterministic + fail-open; feeds the unified `TurnState`.
"""
from __future__ import annotations

import re

# Planning horizons, ordered short → long.
IMMEDIATE = "immediate"          # this exact reply, seconds
CONVERSATION = "conversation"    # this thread
SESSION = "session"              # this working session
PROJECT = "project"              # a multi-turn deliverable / project
LONG_TERM = "long_term"          # career / months+
HORIZONS = [IMMEDIATE, CONVERSATION, SESSION, PROJECT, LONG_TERM]
_ORDINAL = {h: i for i, h in enumerate(HORIZONS)}

_IMMEDIATE_CUES = ("right now", "quick question", "just tell me", "asap",
                   "immediately", "one-liner", "tl;dr", "real quick")
_PROJECT_CUES = ("this project", "the whole app", "the codebase", "the system",
                 "build me", "the repo", "end to end", "the application",
                 "across the project", "the entire", "whole project",
                 "the project", "entire project")
_LONG_TERM_CUES = ("my career", "long term", "long-term", "over the next year",
                   "eventually", "five years", "in the future", "roadmap",
                   "strategy", "grow into")

_REL_TIME = re.compile(
    r"\b(yesterday|today|tomorrow|tonight|last (?:week|month|year|night)|"
    r"next (?:week|month|year)|this (?:week|month|year|morning|afternoon)|"
    r"an hour ago|a (?:minute|moment) ago|earlier|later|"
    r"before the (?:offer|interview|call|meeting)|after the (?:offer|interview|call))\b",
    re.IGNORECASE)

_DEADLINE = re.compile(
    r"\b(by (?:today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|end of day|eod|the end of|next week)|due (?:today|tomorrow|by)|"
    r"deadline|before \d{1,2}\s*(?:am|pm)|within (?:the )?(?:hour|day|week))\b",
    re.IGNORECASE)


def classify_horizon(text: str) -> str:
    """Best-effort planning horizon for a request. Defaults to CONVERSATION."""
    try:
        t = (text or "").lower()
        if not t.strip():
            return CONVERSATION
        if any(c in t for c in _LONG_TERM_CUES):
            return LONG_TERM
        if any(c in t for c in _PROJECT_CUES):
            return PROJECT
        if any(c in t for c in _IMMEDIATE_CUES):
            return IMMEDIATE
        return CONVERSATION
    except Exception:  # noqa: BLE001
        return CONVERSATION


def horizon_ordinal(horizon: str) -> int:
    return _ORDINAL.get(horizon, _ORDINAL[CONVERSATION])


def relative_time_refs(text: str) -> list[str]:
    """Distinct relative-time expressions mentioned (lower-cased)."""
    try:
        return list(dict.fromkeys(m.group(0).lower() for m in _REL_TIME.finditer(text or "")))
    except Exception:  # noqa: BLE001
        return []


def has_deadline(text: str) -> bool:
    try:
        return bool(_DEADLINE.search(text or ""))
    except Exception:  # noqa: BLE001
        return False


def temporal_signal(text: str) -> dict:
    """Compact temporal read for the shared state."""
    return {
        "horizon": classify_horizon(text),
        "time_refs": relative_time_refs(text),
        "deadline": has_deadline(text),
    }


__all__ = [
    "IMMEDIATE", "CONVERSATION", "SESSION", "PROJECT", "LONG_TERM", "HORIZONS",
    "classify_horizon", "horizon_ordinal", "relative_time_refs",
    "has_deadline", "temporal_signal",
]
