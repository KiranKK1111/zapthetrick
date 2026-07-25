"""Live pre-flight systems board (vNext §4.6, Stage 6 Component A).

Before a Live interview starts, a ~10 s systems board checks every subsystem the
session needs — interviewer audio, the mic channel, an STT round-trip, an LLM
first-token ping, the Stage-5 model plan is pinned, the GPU lanes are ready, the
session context is computed — and REFUSES a broken session with an actionable
fix hint (acceptance: "Pre-flight refuses a broken session"). It also seeds the
§4.7 audio watchdog with the check baseline.

Design:
  * Each check is a typed row `{name, ok: bool|None, detail, hint, blocking}` and
    is ISOLATED + fail-open — a probe that errors reports `ok=None`, never
    crashes the board.
  * The board `ready` is False ONLY when some BLOCKING check is explicitly
    `ok=False`. An unknown/unprobed check (`ok=None`) NEVER refuses a session —
    the board is a safety gate, not a new failure mode (fail-open by construction).
  * Environment-dependent probes (audio / STT / LLM / GPU / context / backend
    readiness) are INJECTED via `PreflightProbes`, so the board is unit-tested
    with no audio device, GPU, or model call. The `model_plan` check reads the
    in-process Stage-5 `live_plan` directly (no injection, no cross-package edge).

Flag-gated (`live.preflight`, default OFF → today's immediate start, no board).
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# A probe returns (ok, detail): ok True = pass, False = fail, None = unknown.
Probe = Callable[[], Awaitable["tuple[bool | None, str]"]]


@dataclass
class PreflightProbes:
    """Injected environment probes (all optional). An absent probe → the check is
    reported `ok=None` ('not probed') and never blocks. The real Live route wires
    these to `/readyz`, the audio capture topology, an STT round-trip, and
    `warm_live_provider`; tests stub them."""
    backend: Probe | None = None            # /readyz subsystems (blocking)
    interviewer_audio: Probe | None = None  # interviewer channel receiving
    mic_channel: Probe | None = None        # candidate mic live
    stt_roundtrip: Probe | None = None      # STT round-trip under SLO (blocking)
    llm_first_token: Probe | None = None    # first-token ping to the pinned model (blocking)
    gpu_lanes: Probe | None = None          # §9.1 GPU lanes ready
    session_context: Probe | None = None    # profile/JD/org brief computed


@dataclass
class PreflightCheck:
    name: str
    ok: bool | None
    detail: str = ""
    hint: str = ""
    blocking: bool = False

    def as_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "detail": self.detail,
                "hint": self.hint, "blocking": self.blocking}


@dataclass
class PreflightBoard:
    ready: bool
    checks: list[PreflightCheck] = field(default_factory=list)

    @property
    def blocking_failures(self) -> list[PreflightCheck]:
        return [c for c in self.checks if c.ok is False and c.blocking]

    def as_dict(self) -> dict:
        return {"ready": self.ready,
                "checks": [c.as_dict() for c in self.checks],
                "blocking_failures": [c.name for c in self.blocking_failures]}


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(getattr(cfg, "live", None), "preflight", False))
    except Exception:  # noqa: BLE001
        return False


# Static definition of the board: (name, blocking, hint-on-failure). The order is
# the display order; the probe for each is looked up on `PreflightProbes`.
_CHECKS: tuple[tuple[str, bool, str], ...] = (
    ("backend", True, "Backend subsystems aren't ready — check /readyz."),
    ("interviewer_audio", False,
     "No interviewer audio yet — check the capture source / share-audio."),
    ("mic_channel", False, "Mic not detected — you can still use push-to-talk."),
    ("stt_roundtrip", True,
     "Speech-to-text isn't responding — restart STT or fail over to cloud."),
    ("llm_first_token", True,
     "The answer model didn't respond — check the provider key / connection."),
    ("model_plan", True,
     "No session model pinned — run pre-flight after the model plan is set."),
    ("gpu_lanes", False, "GPU lanes busy — running on the cloud path."),
    ("session_context", False,
     "Session context still computing — depth will fill in shortly."),
)


async def _run_probe(probe: Probe | None) -> "tuple[bool | None, str]":
    if probe is None:
        return None, "not probed"
    try:
        res = await asyncio.wait_for(probe(), timeout=12.0)
        ok, detail = res
        return ok, str(detail or "")
    except asyncio.TimeoutError:
        return None, "probe timed out"
    except Exception as exc:  # noqa: BLE001 — a probe error is a report, not a crash
        return None, f"probe error: {exc}"


def _model_plan_check(session_id: str, profile: str) -> PreflightCheck:
    """The Stage-5 §2.7 F Live plan must have a pinned primary for this session.
    Read in-process (no injection). When `routing.live_plan` is OFF the check is
    informational (non-blocking) — the ordinary ladder serves every turn."""
    try:
        from app.llm import live_plan as _lp
        if not _lp.enabled():
            return PreflightCheck("model_plan", None,
                                  "live_plan disabled — ordinary ladder",
                                  blocking=False)
        plan = _lp.live_planner().get(session_id)
        if plan is not None and plan.primary is not None:
            standby = "+standby" if plan.standby is not None else "no standby"
            return PreflightCheck(
                "model_plan", True,
                f"pinned {plan.primary.provider} ({standby})", blocking=True)
        return PreflightCheck(
            "model_plan", False, "no plan pinned",
            hint="No session model pinned — run pre-flight after the model plan "
                 "is set.", blocking=True)
    except Exception as exc:  # noqa: BLE001
        return PreflightCheck("model_plan", None, f"check error: {exc}",
                              blocking=False)


async def run_preflight(session_id: str, profile: str, *,
                        probes: PreflightProbes | None = None) -> PreflightBoard:
    """Run the pre-flight systems board for a Live session. Returns a
    `PreflightBoard`; `ready` is False only when a BLOCKING check explicitly
    failed. Never raises — a board error yields a ready board with a note (the
    board must never itself become the reason a session can't start)."""
    probes = probes or PreflightProbes()
    checks: list[PreflightCheck] = []
    try:
        probe_map = {
            "backend": probes.backend,
            "interviewer_audio": probes.interviewer_audio,
            "mic_channel": probes.mic_channel,
            "stt_roundtrip": probes.stt_roundtrip,
            "llm_first_token": probes.llm_first_token,
            "gpu_lanes": probes.gpu_lanes,
            "session_context": probes.session_context,
        }
        for name, blocking, hint in _CHECKS:
            if name == "model_plan":
                checks.append(_model_plan_check(session_id, profile))
                continue
            ok, detail = await _run_probe(probe_map.get(name))
            checks.append(PreflightCheck(
                name, ok, detail,
                hint=(hint if ok is False else ""), blocking=blocking))
        ready = not any(c.ok is False and c.blocking for c in checks)
        _seed_watchdog(session_id, checks)
        return PreflightBoard(ready=ready, checks=checks)
    except Exception as exc:  # noqa: BLE001 — the board never blocks on its own bug
        log.info("preflight board error (session %s): %s", session_id, exc)
        return PreflightBoard(ready=True, checks=checks or [
            PreflightCheck("board", None, f"board error: {exc}")])


def _seed_watchdog(session_id: str, checks: list[PreflightCheck]) -> None:
    """Best-effort: hand the §4.7 audio watchdog the pre-flight baseline (audio
    channels seen at start). No-op until the watchdog (Component C) lands."""
    try:
        from app.live import silence as _sil
        seed = getattr(_sil, "seed_baseline", None)
        if callable(seed):
            audio_ok = next((c.ok for c in checks
                             if c.name == "interviewer_audio"), None)
            seed(session_id, interviewer_audio=audio_ok)
    except Exception:  # noqa: BLE001 — seeding is best-effort
        pass


__all__ = ["Probe", "PreflightProbes", "PreflightCheck", "PreflightBoard",
           "enabled", "run_preflight"]
