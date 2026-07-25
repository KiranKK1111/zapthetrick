"""Eval flywheel + canary (vNext §8.9, Stage 11 Component A).

The compounding loop: every live FAILURE is a lesson. A thumbs-down, a
forced-answer-after-a-wrong-SKIP, a verify-fail, a schema-retry, a
clarifier-"other" — each is captured (PII-redacted) as a REPLAYABLE case, added
to a golden set, and replayed nightly to catch regressions. And no prompt / gate /
route change ships blind: it rides a **canary** — a small deterministic slice of
traffic — and auto-PROMOTES or auto-ROLLS-BACK on the metric delta.

This module owns the deterministic core:
  * `should_capture` / `capture_case` — turn a failure signal into a redacted,
    replayable `EvalCase` (the redactor is injected — reuses egress redaction);
  * `GoldenSet` — dedup + add;
  * `CanaryController` — a deterministic fraction assignment + a promote /
    rollback / hold verdict from baseline vs canary metrics.

The nightly replay + scoring run on-pod (the harness already exists); this is the
capture + canary decision logic. Pure + fail-open. Flag-gated (`eval.flywheel`,
`eval.canary`, default OFF → no capture, no canary).
"""
from __future__ import annotations

import hashlib
import re as _re
from dataclasses import dataclass, field

# Failure signals worth capturing as cases.
THUMBS_DOWN = "thumbs_down"
FORCED_ANSWER = "forced_answer"        # answer forced after a wrong SKIP
VERIFY_FAIL = "verify_fail"
SCHEMA_RETRY = "schema_retry"
CLARIFIER_OTHER = "clarifier_other"    # user picked "other" / rephrased
_CAPTURE_SIGNALS = frozenset({THUMBS_DOWN, FORCED_ANSWER, VERIFY_FAIL,
                              SCHEMA_RETRY, CLARIFIER_OTHER})

# Canary verdicts.
PROMOTE = "promote"
ROLLBACK = "rollback"
HOLD = "hold"


def flywheel_enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.eval, "flywheel", False))
    except Exception:  # noqa: BLE001
        return False


def canary_enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.eval, "canary", False))
    except Exception:  # noqa: BLE001
        return False


def _canary_cfg() -> "tuple[float, int, float]":
    try:
        from app.core.config_loader import cfg
        return (float(getattr(cfg.eval, "canary_fraction", 0.10) or 0.10),
                int(getattr(cfg.eval, "canary_min_samples", 50) or 50),
                float(getattr(cfg.eval, "canary_margin", 0.02) or 0.02))
    except Exception:  # noqa: BLE001
        return 0.10, 50, 0.02


# --------------------------------------------------------------------------- #
# Case capture
# --------------------------------------------------------------------------- #
@dataclass
class EvalCase:
    id: str
    signal: str
    prompt: str
    response: str = ""
    expected: str = ""
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"id": self.id, "signal": self.signal, "prompt": self.prompt,
                "response": self.response, "expected": self.expected,
                "meta": dict(self.meta)}


def should_capture(signal: str) -> bool:
    """Whether a signal becomes a golden case. Disabled → never. Never raises."""
    try:
        return flywheel_enabled() and signal in _CAPTURE_SIGNALS
    except Exception:  # noqa: BLE001
        return False


def _case_id(signal: str, prompt: str) -> str:
    h = hashlib.sha256(f"{signal}\x00{prompt}".encode("utf-8", "replace"))
    return f"{signal}-{h.hexdigest()[:12]}"


def capture_case(signal: str, *, prompt: str, response: str = "",
                 expected: str = "", meta: dict | None = None,
                 redact_fn=None) -> "EvalCase | None":
    """Build a PII-REDACTED replayable case from a failure signal. `redact_fn(s)
    -> s` is INJECTED (reuses egress redaction); without it a conservative
    built-in scrub runs. Returns None when disabled / not a capture signal /
    empty prompt. Never raises."""
    try:
        if not should_capture(signal) or not (prompt or "").strip():
            return None
        red = redact_fn or _builtin_redact
        return EvalCase(
            id=_case_id(signal, prompt),
            signal=signal,
            prompt=red(prompt),
            response=red(response or ""),
            expected=red(expected or ""),
            meta=dict(meta or {}))
    except Exception:  # noqa: BLE001
        return None


_SECRET_RE = [
    (_re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"), "<email>"),
    (_re.compile(r"\b(?:sk|pk|ghp|xox[baprs])[-_][A-Za-z0-9]{10,}\b"), "<token>"),
    (_re.compile(r"\b\d{12,19}\b"), "<number>"),
    (_re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "<ip>"),
]


def _builtin_redact(text: str) -> str:
    """A conservative fallback PII scrub (the injected egress redactor is
    preferred). Never raises → the input."""
    try:
        t = text or ""
        for rx, repl in _SECRET_RE:
            t = rx.sub(repl, t)
        return t
    except Exception:  # noqa: BLE001
        return text or ""


# --------------------------------------------------------------------------- #
# Golden set
# --------------------------------------------------------------------------- #
@dataclass
class GoldenSet:
    cases: dict = field(default_factory=dict)   # id -> EvalCase

    def add(self, case: "EvalCase | None") -> bool:
        """Add a case (deduped by id). Returns True if newly added. Never raises."""
        try:
            if case is None or not case.id:
                return False
            if case.id in self.cases:
                return False
            self.cases[case.id] = case
            return True
        except Exception:  # noqa: BLE001
            return False

    def __len__(self) -> int:
        return len(self.cases)

    def signals(self) -> dict:
        out: dict = {}
        for c in self.cases.values():
            out[c.signal] = out.get(c.signal, 0) + 1
        return out


# --------------------------------------------------------------------------- #
# Canary controller
# --------------------------------------------------------------------------- #
@dataclass
class CanaryVerdict:
    decision: str                 # promote | rollback | hold
    baseline: float
    canary: float
    samples: int
    reason: str = ""

    def to_dict(self) -> dict:
        return {"decision": self.decision, "baseline": round(self.baseline, 4),
                "canary": round(self.canary, 4), "samples": self.samples,
                "reason": self.reason}


class CanaryController:
    """Assign a deterministic traffic slice to the canary and decide its fate."""

    def assign(self, key: str, *, fraction: float | None = None) -> bool:
        """Whether `key` (a stable id — user/session/request) is in the canary
        slice. Deterministic hash bucketing → the SAME key always lands the same
        way (a user has a stable experience). Disabled → always False (everyone
        on baseline). Never raises."""
        try:
            if not canary_enabled():
                return False
            frac = _canary_cfg()[0] if fraction is None else fraction
            frac = max(0.0, min(1.0, frac))
            if frac <= 0:
                return False
            if frac >= 1:
                return True
            h = hashlib.sha256((key or "").encode("utf-8", "replace")).hexdigest()
            bucket = int(h[:8], 16) / 0xFFFFFFFF
            return bucket < frac
        except Exception:  # noqa: BLE001
            return False

    def evaluate(self, *, baseline_score: float, canary_score: float,
                 samples: int, min_samples: int | None = None,
                 margin: float | None = None) -> CanaryVerdict:
        """Promote / rollback / hold from the metric delta. HIGHER score is
        better. Needs `min_samples` before deciding; then PROMOTE if canary beats
        baseline by ≥ margin, ROLLBACK if it's WORSE by ≥ margin, else HOLD.
        Disabled → hold. Never raises."""
        try:
            if not canary_enabled():
                return CanaryVerdict(HOLD, baseline_score, canary_score, samples,
                                     "canary disabled")
            _, cfg_min, cfg_margin = _canary_cfg()
            need = cfg_min if min_samples is None else min_samples
            mrg = cfg_margin if margin is None else margin
            if samples < need:
                return CanaryVerdict(HOLD, baseline_score, canary_score, samples,
                                     f"need {need} samples, have {samples}")
            delta = canary_score - baseline_score
            if delta >= mrg:
                return CanaryVerdict(PROMOTE, baseline_score, canary_score, samples,
                                     f"canary +{delta:.3f} >= margin {mrg:.3f}")
            if delta <= -mrg:
                return CanaryVerdict(ROLLBACK, baseline_score, canary_score, samples,
                                     f"canary {delta:.3f} <= -margin {mrg:.3f}")
            return CanaryVerdict(HOLD, baseline_score, canary_score, samples,
                                 f"delta {delta:.3f} within +/-{mrg:.3f}")
        except Exception:  # noqa: BLE001
            return CanaryVerdict(HOLD, baseline_score, canary_score, samples, "error")


__all__ = ["THUMBS_DOWN", "FORCED_ANSWER", "VERIFY_FAIL", "SCHEMA_RETRY",
           "CLARIFIER_OTHER", "PROMOTE", "ROLLBACK", "HOLD", "flywheel_enabled",
           "canary_enabled", "EvalCase", "should_capture", "capture_case",
           "GoldenSet", "CanaryVerdict", "CanaryController"]
