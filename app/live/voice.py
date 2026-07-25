"""Voice contract + reasoning-leak guard (vNext §4.10, Stage 6 Component D).

A dictated Live answer must sound like the CANDIDATE talking — first person,
spoken register, no markdown scaffolding, shaped to the seniority band — and it
must never leak the model's internal reasoning or meta-commentary. §4.10 layers
three defenses; this module adds the two that weren't already in place:

  * **Layer 2 — streaming leakage guard** (`live.leak_guard`): the answer HEAD is
    watched for internal-reasoning / meta-discourse ("we need to answer…", "as an
    AI…", "here's my response…"); a hit → HOLD the stream and regenerate with the
    model ROTATED (leaks are a per-weight tic). Reuses the existing
    `verify.looks_like_leaked_reasoning` detector + a meta-discourse pass over the
    first ~2 sentences.
  * **Layer 3 — voice validator** (`live.voice_contract`): first-person? spoken
    register (no bullets / headers / fences / bold)? no scaffolding/meta? band
    shape (length via the session contract)? → a `VoiceVerdict`, feeding a
    per-model **dictatability EWMA** (which model reliably produces dictatable
    prose) that the router (§2.6) can later prefer.

Layer 1 (channel separation — stripping `<think>`/harmony markers) already lives
in `app/response_arch/sanitize.py`; the "never restate the question" style clause
is exposed here as `never_restate_clause()` for the L1 prompt. Deterministic +
fail-open; every check degrades to "looks fine" on error so the guard is never
itself the reason an answer stalls. Flag-gated (both default OFF).
"""
from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


def voice_enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "voice_contract", False))
    except Exception:  # noqa: BLE001
        return False


def leak_guard_enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "leak_guard", False))
    except Exception:  # noqa: BLE001
        return False


# --------------------------------------------------------------------------- #
# Layer 2 — streaming leakage guard
# --------------------------------------------------------------------------- #
# Meta-discourse the answer must NOT open with (in addition to the reasoning
# leaks `verify.looks_like_leaked_reasoning` already catches). These are the
# "assistant preamble" tells — the answer should just BE the answer.
_META_HEAD = (
    "as an ai", "as a language model", "here's my response", "here is my response",
    "here's how i", "here is how i", "sure, here", "sure! here", "certainly,",
    "i'd be happy to", "i would be happy to", "let me answer", "to answer your",
    "in this response", "the following is", "below is my", "my response is",
)


def leaked_head(text: str) -> bool:
    """True when the answer HEAD reads as internal reasoning OR assistant
    meta-discourse rather than the candidate's actual answer. Never raises."""
    try:
        t = (text or "").strip()
        if not t:
            return False
        from app.live.verify import looks_like_leaked_reasoning
        if looks_like_leaked_reasoning(t):
            return True
        # First ~2 sentences only — a leak/preamble is always at the very start.
        head = _first_sentences(t, 2).lower()
        return any(m in head for m in _META_HEAD)
    except Exception:  # noqa: BLE001
        return False


@dataclass
class LeakDecision:
    hold: bool
    reason: str = ""
    rotate_model: bool = False


def should_hold(streamed_text: str) -> LeakDecision:
    """Layer-2 decision on the streamed HEAD: HOLD + regenerate (model rotated)
    when a leak/meta-discourse opener is detected. No-op (hold=False) when the
    guard is off. Fail-open → no hold. The caller performs the regeneration with
    `avoid_model_db_id` set (rotate)."""
    try:
        if not leak_guard_enabled():
            return LeakDecision(hold=False)
        if leaked_head(streamed_text):
            return LeakDecision(
                hold=True,
                reason="answer opened with internal reasoning / meta-discourse",
                rotate_model=True)
        return LeakDecision(hold=False)
    except Exception:  # noqa: BLE001
        return LeakDecision(hold=False)


# --------------------------------------------------------------------------- #
# Layer 3 — voice validator
# --------------------------------------------------------------------------- #
# Markdown scaffolding that must never appear in DICTATED prose.
_SCAFFOLD_RE = re.compile(
    r"(^|\n)\s*(#{1,6}\s|[-*+]\s|\d+[.)]\s|>\s)|```|\*\*|__|\|.*\|")
# First-person markers (the candidate speaking).
_FIRST_PERSON_RE = re.compile(r"\b(i|i'?m|i'?ve|i'?d|i'?ll|my|mine|me|we|our)\b",
                              re.I)
# Third-person "about the candidate" tells (wrong voice).
_THIRD_PERSON_RE = re.compile(
    r"\b(the candidate|the applicant|one should|the interviewee|this person)\b",
    re.I)


@dataclass
class VoiceVerdict:
    first_person: bool = True
    spoken_register: bool = True   # no markdown scaffolding
    no_scaffolding: bool = True    # no assistant meta / preamble
    band_shape_ok: bool = True     # within the contract's spoken length
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (self.first_person and self.spoken_register
                and self.no_scaffolding and self.band_shape_ok)

    def as_dict(self) -> dict:
        return {"ok": self.ok, "first_person": self.first_person,
                "spoken_register": self.spoken_register,
                "no_scaffolding": self.no_scaffolding,
                "band_shape_ok": self.band_shape_ok, "issues": self.issues}


def validate_voice(text: str, *, contract=None) -> VoiceVerdict:
    """Layer-3 voice validation of a (finished) dictated answer. Deterministic;
    never raises → a passing verdict on error (advisory, not a gate)."""
    v = VoiceVerdict()
    try:
        t = (text or "").strip()
        if not t:
            return v
        # Spoken register: markdown scaffolding is disqualifying.
        if _SCAFFOLD_RE.search(t):
            v.spoken_register = False
            v.issues.append("markdown scaffolding in dictated prose")
        # Meta / preamble.
        if leaked_head(t):
            v.no_scaffolding = False
            v.issues.append("assistant meta / reasoning in the opening")
        # First person: present, and not talking ABOUT the candidate.
        if _THIRD_PERSON_RE.search(t) or not _FIRST_PERSON_RE.search(t):
            v.first_person = False
            v.issues.append("not first-person candidate voice")
        # Band shape: reuse the session contract's spoken-length check.
        if contract is not None:
            try:
                from app.live.contract import validate as _cv
                chk = _cv(t, contract)
                if getattr(chk, "ok", True) is False:
                    v.band_shape_ok = False
                    v.issues.append("outside the band's spoken length")
            except Exception:  # noqa: BLE001
                pass
        return v
    except Exception:  # noqa: BLE001
        return VoiceVerdict()


def never_restate_clause() -> str:
    """The §4.10 L1 style clause: answer in the candidate's own voice, don't echo
    the question back. Prepended to the live answer prompt by the caller."""
    return ("Answer in first person as the candidate, in a natural SPOKEN "
            "register — no markdown, headings, bullet points, or code fences, and "
            "never restate the interviewer's question. Just speak the answer.")


# --------------------------------------------------------------------------- #
# Per-model "dictatability" EWMA (§4.10 → §2.6)
# --------------------------------------------------------------------------- #
_ALPHA = 0.2
_LOCK = threading.RLock()
_dictatability: dict[str, float] = {}


def record_dictatability(identity_key: str, ok: bool) -> None:
    """EWMA-update how reliably a model produces DICTATABLE prose (the Layer-3
    verdict). Seeded optimistically at 1.0 so an unseen model isn't punished."""
    if not identity_key:
        return
    try:
        with _LOCK:
            prev = _dictatability.get(identity_key, 1.0)
            _dictatability[identity_key] = round(
                (1 - _ALPHA) * prev + _ALPHA * (1.0 if ok else 0.0), 4)
    except Exception:  # noqa: BLE001
        pass


def dictatability(identity_key: str) -> float:
    """0..1 dictatability rate for a model identity (1.0 = unseen/optimistic)."""
    try:
        with _LOCK:
            return _dictatability.get(identity_key, 1.0)
    except Exception:  # noqa: BLE001
        return 1.0


def clear_dictatability() -> None:
    with _LOCK:
        _dictatability.clear()


# --------------------------------------------------------------------------- #
def _first_sentences(text: str, n: int) -> str:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return " ".join(parts[:max(1, n)])


__all__ = ["voice_enabled", "leak_guard_enabled", "leaked_head", "should_hold",
           "LeakDecision", "VoiceVerdict", "validate_voice",
           "never_restate_clause", "record_dictatability", "dictatability",
           "clear_dictatability"]
