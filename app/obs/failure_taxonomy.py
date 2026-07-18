"""Formal failure taxonomy (roadmap Phase 1 #5).

Every failure the platform can hit is classified into a typed `FailureClass`
with a severity and a *recovery strategy* — so recovery is deterministic
(repair / switch-model / cooldown / replan / clarify / degrade), never a blind
retry (anti-pattern rule #10). This is the shared vocabulary the Recovery Planner
(Phase 4 #8), the failure KB (Phase 7), and error UX build on.

Self-contained (no app imports) so it can be used anywhere without coupling.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


class Severity(enum.Enum):
    LOW = "low"            # cosmetic / recoverable silently
    MEDIUM = "medium"      # user-visible degradation, recoverable
    HIGH = "high"          # turn likely fails without recovery
    CRITICAL = "critical"  # session/data at risk


class Recovery(enum.Enum):
    RETRY = "retry"                     # same path, transient blip
    RETRY_DIFFERENT = "retry_different" # different model/key/strategy
    COOLDOWN_WAIT = "cooldown_wait"     # back off, then retry (rate limits)
    REPLAN = "replan"                   # re-plan the task
    REPAIR = "repair"                   # fix the artifact/output, re-verify
    CLARIFY = "clarify"                 # ask the user
    DEGRADE = "degrade"                 # fall back to a lesser capability
    SKIP = "skip"                       # intentionally do nothing (not an error)
    ESCALATE = "escalate"               # surface to the user, can't self-recover
    ABORT = "abort"                     # stop cleanly


@dataclass(frozen=True)
class FailureClass:
    id: str
    title: str
    severity: Severity
    recovery: Recovery
    retryable: bool
    description: str


def _c(id, title, severity, recovery, retryable, description) -> FailureClass:
    return FailureClass(id, title, severity, recovery, retryable, description)


# The registry — grounded in the platform's real failure surface (STT chain,
# LLM router, live decision engine, generation, verification, sandbox, artifacts).
TAXONOMY: dict[str, FailureClass] = {c.id: c for c in [
    # ── input / STT ──
    _c("stt_unavailable", "STT engine unavailable", Severity.HIGH,
       Recovery.DEGRADE, False, "The STT engine failed to load/run; fall back or surface stt_status."),
    _c("stt_empty", "STT heard but produced nothing", Severity.LOW,
       Recovery.SKIP, False, "Audio captured but transcript empty (noise/silence)."),
    _c("endpoint_ambiguous", "Utterance boundary unclear", Severity.LOW,
       Recovery.RETRY, True, "Endpointing couldn't decide; hold/merge and re-evaluate."),

    # ── decision (not errors — classified non-answers) ──
    _c("decision_skip", "Intentional skip (not a question)", Severity.LOW,
       Recovery.SKIP, False, "Decision engine chose SKIP (directive/echo/feedback); a muted row, not a failure."),
    _c("budget_cap", "Concurrency/budget cap hit", Severity.MEDIUM,
       Recovery.DEGRADE, False, "Session budget cap reached; skip with reason (surface a force-answerable row)."),

    # ── retrieval ──
    _c("retrieval_empty", "No relevant context found", Severity.LOW,
       Recovery.DEGRADE, False, "Retriever returned nothing; answer from model knowledge / lower confidence."),
    _c("retrieval_error", "Retrieval subsystem error", Severity.MEDIUM,
       Recovery.DEGRADE, False, "Vector/keyword search failed; degrade to no-context and continue."),

    # ── provider / router ──
    _c("provider_rate_limit", "Provider rate limited (429)", Severity.MEDIUM,
       Recovery.COOLDOWN_WAIT, True, "429/rate window exhausted; apply cooldown and route to another model."),
    _c("provider_transport", "Provider transport error", Severity.MEDIUM,
       Recovery.RETRY_DIFFERENT, True, "Network/transport blip to the provider; retry a different (model,key)."),
    _c("provider_auth", "Provider auth/key failure", Severity.HIGH,
       Recovery.RETRY_DIFFERENT, True, "Bad/expired/undecryptable key; skip that key, try another; re-enter if all fail."),
    _c("provider_exhausted", "No usable provider available", Severity.HIGH,
       Recovery.ESCALATE, False, "Every candidate model is cooling down/unavailable; surface honestly."),

    # ── planning / generation ──
    _c("planning_error", "Planner failed", Severity.HIGH,
       Recovery.REPLAN, True, "Plan construction failed; re-plan or fall back to a direct answer."),
    _c("generation_timeout", "Generation exceeded deadline", Severity.MEDIUM,
       Recovery.RETRY_DIFFERENT, True, "Answer generation passed its wall-clock budget; persist partial, offer Continue."),
    _c("generation_empty", "Model returned nothing", Severity.MEDIUM,
       Recovery.RETRY_DIFFERENT, True, "Empty/garbled completion; retry a different model."),

    # ── verification / sandbox ──
    _c("verification_failed", "Deliverable failed verification", Severity.HIGH,
       Recovery.REPAIR, True, "Compile/test/smoke-run failed; enter bounded repair loop, re-verify."),
    _c("verification_partial", "Deliverable partially verified", Severity.MEDIUM,
       Recovery.REPAIR, True, "Some checks passed; repair the rest or deliver with an honest partial verdict."),
    _c("sandbox_error", "Sandbox execution error", Severity.MEDIUM,
       Recovery.DEGRADE, False, "Sandbox couldn't run (isolation/host issue); degrade to static checks."),
    _c("sandbox_timeout", "Sandbox run timed out", Severity.MEDIUM,
       Recovery.REPAIR, True, "Runaway execution killed by timeout; treat as a failing check and repair."),

    # ── artifact / output ──
    _c("artifact_build_error", "Artifact build failed", Severity.MEDIUM,
       Recovery.REPAIR, True, "Doc/archive generation failed; repair or return the original bytes (fail-open)."),

    # ── transport / infra ──
    _c("network_error", "Network unavailable", Severity.MEDIUM,
       Recovery.DEGRADE, False, "Internet down; degrade to fully-local models/answering."),
    _c("internal_error", "Unclassified internal error", Severity.HIGH,
       Recovery.ESCALATE, False, "Catch-all; log with full trace and surface an honest error."),
]}


def get(failure_id: str) -> FailureClass | None:
    return TAXONOMY.get(failure_id)


def all_classes() -> list[FailureClass]:
    return list(TAXONOMY.values())


# Substring signatures for mapping raw exceptions to a class. Ordered — first
# match wins — so specific signatures precede generic ones.
_SIGNATURES: list[tuple[tuple[str, ...], str]] = [
    (("429", "rate limit", "rate-limit", "ratelimit", "too many requests"), "provider_rate_limit"),
    (("decrypt", "invalid api key", "unauthorized", "401", "403", "forbidden"), "provider_auth"),
    (("timeout", "timed out", "deadline"), "generation_timeout"),
    (("connection", "transport", "connect", "read timed out", "econnreset"), "provider_transport"),
    (("no module named", "import", "compile", "syntaxerror"), "verification_failed"),
    (("network is unreachable", "name resolution", "dns", "offline"), "network_error"),
]


def classify_exception(exc: BaseException) -> FailureClass:
    """Best-effort map a raw exception to a FailureClass (never raises)."""
    try:
        text = f"{type(exc).__name__}: {exc}".lower()
    except Exception:  # noqa: BLE001
        return TAXONOMY["internal_error"]
    if isinstance(exc, TimeoutError):
        return TAXONOMY["generation_timeout"]
    if isinstance(exc, (ConnectionError, OSError)):
        # refine by message below if possible, else transport
        for needles, fid in _SIGNATURES:
            if any(n in text for n in needles):
                return TAXONOMY[fid]
        return TAXONOMY["provider_transport"]
    for needles, fid in _SIGNATURES:
        if any(n in text for n in needles):
            return TAXONOMY[fid]
    return TAXONOMY["internal_error"]


def observe(exc: BaseException, *, where: str = "") -> FailureClass:
    """Classify an exception, log it with its class + recommended recovery, and
    return the FailureClass. Fail-open: never raises, so it's safe to drop into
    any `except` block for structured failure diagnostics.
    """
    fc = classify_exception(exc)
    try:
        log.info("failure[%s] class=%s severity=%s recovery=%s :: %s",
                 where or "?", fc.id, fc.severity.value, fc.recovery.value, exc)
    except Exception:  # noqa: BLE001 — diagnostics must never break a call
        pass
    return fc


__all__ = [
    "Severity", "Recovery", "FailureClass",
    "TAXONOMY", "get", "all_classes", "classify_exception", "observe",
]
