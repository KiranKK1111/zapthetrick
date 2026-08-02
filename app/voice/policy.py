"""Engine selection and degradation (design §"Engine Selection, Cost …").

Evaluated once at session open, then re-evaluated only on failure::

    voice.engine = "auto" | "realtime" | "staged"

    auto:
      realtime  if credential present
                and a realtime model is configured
                and the endpoint is reachable within the preflight timeout
                and budget.remaining() > session_reserve
      else staged

`staged` is the default, so a fresh checkout, an offline machine or a build
without credentials behaves exactly as it does today — that is Requirement 10.1
("no credential ⇒ staged, without presenting an error").

The empty-`realtime_model` interlock is checked FIRST and cannot be overridden
by `engine: realtime`. Selection is otherwise deliberately boring: every reason
to fall back is named, so a fallback is explainable in a log line instead of
being a mystery.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from app.voice import budget

log = logging.getLogger("zapthetrick.voice.policy")

STAGED = "staged"
REALTIME = "realtime"


@dataclass(frozen=True)
class Selection:
    engine: str
    reason: str = ""
    # Advisory: a realtime request that fell back. Surfaced in telemetry so a
    # silently-degraded deployment is visible rather than merely quiet.
    degraded: bool = False


def _voice_cfg():
    from app.core.config_loader import cfg
    return cfg.voice


def configured_engine() -> str:
    """The requested engine, honouring the superseded `s2s_engine` key for one
    release (Requirement 6.5): "omni" maps to "realtime", "staged" to "staged".

    `voice.engine` wins whenever it is set to something other than its default,
    so a deployment that has migrated is never overridden by a stale key.
    """
    v = _voice_cfg()
    engine = str(getattr(v, "engine", STAGED) or STAGED).strip().lower()
    if engine != STAGED:
        return engine
    legacy = str(getattr(v, "s2s_engine", "") or "").strip().lower()
    if legacy == "omni":
        log.info("voice: legacy s2s_engine=omni mapped to engine=realtime")
        return REALTIME
    return STAGED


def realtime_model() -> str:
    return str(getattr(_voice_cfg(), "realtime_model", "") or "").strip()


def realtime_credential() -> str:
    """The realtime API key: the explicit `voice.realtime_api_key`, else the
    OpenAI key already in the LLM config. Never logged."""
    v = _voice_cfg()
    key = str(getattr(v, "realtime_api_key", "") or "").strip()
    if key:
        return key
    try:
        from app.core.config_loader import cfg
        return str(getattr(cfg.llm, "openai_api_key", "") or "").strip()
    except Exception:  # noqa: BLE001
        return ""


def select(*, reachable: bool | None = None) -> Selection:
    """Choose the engine for a new session.

    `reachable` is the result of the caller's cheap preflight probe. `None`
    means "not probed" and is treated as reachable — the runner performs the
    real probe and hands the answer in; unit tests can skip it.
    """
    requested = configured_engine()
    if requested == STAGED:
        return Selection(STAGED, "configured")

    # Hard interlock, checked before anything else: no model ⇒ never realtime,
    # whatever `engine` says. This is what makes default spend exactly zero.
    if not realtime_model():
        return Selection(STAGED, "no realtime model configured", degraded=True)

    if not realtime_credential():
        # Requirement 10.1 — silent, this is the normal local mode.
        return Selection(STAGED, "no realtime credential", degraded=False)

    ok, why = budget.can_open_session()
    if not ok:
        return Selection(STAGED, why, degraded=True)

    if reachable is False:
        return Selection(STAGED, "realtime endpoint unreachable", degraded=False)

    return Selection(REALTIME, "selected")


def fallback(reason: str) -> Selection:
    """The selection to hand over to after a mid-session realtime failure."""
    return Selection(STAGED, reason, degraded=True)


__all__ = [
    "STAGED", "REALTIME", "Selection", "configured_engine", "realtime_model",
    "realtime_credential", "select", "fallback",
]
