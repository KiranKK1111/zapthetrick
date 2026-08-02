"""Speaker verification gate — Phase 4 (Requirement 12).

The problem this solves is one no realtime model solves for you: a speech-native
model cannot tell whose mouth the audio came from either. In a shared room, a
colleague's voice can take a turn or confirm a barge-in. Addressing that is
where this design can exceed ChatGPT rather than approximate it.

The gate is evaluated at exactly two points — **before turn admission** and
**before interruption confirmation** — because a bystander must be able to do
neither.

Fail-open, and that is load-bearing
-----------------------------------
`app/live/speaker_embed.py` is referenced by the Live module but is **not present
in this checkout**: `routes_ws.py._speaker_embedding()` imports it lazily inside a
try/except and fail-softs to `None`, and its docstring says the embedder lives
only on the pod. So today there is an import site, not an implementation.

Requirement 12.3 is written for exactly that: gate disabled, no enrolment, or no
embedder ⇒ behave EXACTLY as without this feature. Every path here returns
"admit" when it cannot make an informed decision, so shipping this cannot make a
session worse — it can only make it stricter once someone enrols.

The embedder is reached through a wrapper (rule L2): voice-specific logic never
lands in `app/live/`.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger("zapthetrick.voice.speaker")

# Minimum speech to enrol from / verify against. Below this an embedding is
# dominated by noise and its similarity score is meaningless — admitting is
# strictly safer than guessing.
MIN_ENROL_SECONDS = 3.0
MIN_VERIFY_SECONDS = 0.6
SAMPLE_RATE = 16_000


@dataclass
class SpeakerProfile:
    """An enrolled voice. Stored as a plain vector so it rides existing rows
    (`User.preferences`) and needs no schema change (Requirement 11.4)."""

    embedding: tuple[float, ...] = ()
    samples: int = 0
    # Rolling centroid, so enrolment improves with use rather than being frozen
    # to whatever the first three seconds sounded like.
    _sum: list[float] = field(default_factory=list, repr=False)

    @property
    def enrolled(self) -> bool:
        return bool(self.embedding)

    def to_dict(self) -> dict:
        return {"embedding": list(self.embedding), "samples": self.samples}

    @classmethod
    def from_dict(cls, d: dict | None) -> "SpeakerProfile":
        if not isinstance(d, dict):
            return cls()
        vec = d.get("embedding") or []
        try:
            emb = tuple(float(x) for x in vec)
        except (TypeError, ValueError):
            return cls()
        return cls(embedding=emb, samples=int(d.get("samples") or 0),
                   _sum=[float(x) * max(1, int(d.get("samples") or 1))
                         for x in emb])

    def add(self, vec) -> None:
        """Fold another sample into the centroid."""
        v = [float(x) for x in vec]
        if not v:
            return
        if not self._sum or len(self._sum) != len(v):
            self._sum = list(v)
            self.samples = 1
        else:
            self._sum = [a + b for a, b in zip(self._sum, v)]
            self.samples += 1
        n = max(1, self.samples)
        self.embedding = tuple(x / n for x in self._sum)


def _embed(audio) -> tuple[float, ...] | None:
    """Speaker embedding via the on-pod embedder, or None when it is absent.

    A WRAPPER over `app.live.speaker_embed` (rule L2) — this never modifies the
    Live module. None here means "no opinion", which every caller treats as
    admit.
    """
    try:
        if audio is None:
            return None
        from app.live import speaker_embed as _se   # on-pod only
        vec = _se.embed(audio)
        return tuple(float(x) for x in vec) if vec else None
    except Exception:  # noqa: BLE001 — no embedder here ⇒ no opinion
        return None


def available() -> bool:
    """Whether a speaker embedder exists in this deployment."""
    try:
        import app.live.speaker_embed  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _cfg():
    from app.core.config_loader import cfg
    return cfg.voice


def enabled() -> bool:
    return bool(getattr(_cfg(), "speaker_gate", False))


def threshold() -> float:
    return float(getattr(_cfg(), "speaker_threshold", 0.65) or 0.65)


def similarity(a, b) -> float:
    """Cosine similarity in [-1, 1]. 0.0 when either side is empty/degenerate."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na <= 1e-9 or nb <= 1e-9:
        return 0.0
    return float(dot / (na * nb))


def _seconds(audio) -> float:
    try:
        n = int(getattr(audio, "size", 0) or len(audio))
    except (TypeError, ValueError):
        return 0.0
    return n / float(SAMPLE_RATE)


def enrol(profile: SpeakerProfile, audio) -> tuple[SpeakerProfile, bool]:
    """Fold a sample of the user's speech into their profile.

    Returns `(profile, accepted)`. A clip shorter than `MIN_ENROL_SECONDS` is
    REJECTED rather than folded in: a short, noisy embedding poisons the centroid
    permanently and every later verification pays for it.
    """
    if _seconds(audio) < MIN_ENROL_SECONDS:
        return profile, False
    vec = _embed(audio)
    if not vec:
        return profile, False
    profile.add(vec)
    return profile, True


@dataclass(frozen=True)
class Verdict:
    """`admit` is what callers act on. `score` and `reason` exist so a rejected
    turn is explainable — a gate that silently swallows speech is indebuggable."""

    admit: bool
    score: float = 0.0
    reason: str = ""

    @property
    def informed(self) -> bool:
        """Whether a real comparison happened (as opposed to failing open)."""
        return self.reason in ("match", "mismatch")


def verify(profile: SpeakerProfile | None, audio) -> Verdict:
    """Is this the enrolled speaker?

    Admits — deliberately — whenever an informed decision is impossible: gate
    off, nobody enrolled, no embedder, or too little audio. Requirement 12.3
    says an absent embedder means *current behaviour*, not a silent mute.
    """
    if not enabled():
        return Verdict(True, reason="gate_disabled")
    if profile is None or not profile.enrolled:
        return Verdict(True, reason="not_enrolled")
    if _seconds(audio) < MIN_VERIFY_SECONDS:
        # Too short to judge. Rejecting here would drop legitimate one-word
        # answers ("yes", "no"), which are real turns.
        return Verdict(True, reason="too_short")
    vec = _embed(audio)
    if not vec:
        return Verdict(True, reason="no_embedder")
    score = similarity(vec, profile.embedding)
    if score >= threshold():
        return Verdict(True, score=score, reason="match")
    return Verdict(False, score=score, reason="mismatch")


def admits_turn(profile: SpeakerProfile | None, audio) -> bool:
    """Gate BEFORE turn admission (Requirement 12.2)."""
    return verify(profile, audio).admit


def admits_interruption(profile: SpeakerProfile | None, audio) -> bool:
    """Gate BEFORE interruption confirmation.

    Ordering matters: a bystander must be able neither to take a turn nor to cut
    the assistant off mid-answer. Checking only at turn admission would leave the
    second door open.
    """
    return verify(profile, audio).admit


__all__ = [
    "SpeakerProfile", "Verdict", "MIN_ENROL_SECONDS", "MIN_VERIFY_SECONDS",
    "available", "enabled", "threshold", "similarity", "enrol", "verify",
    "admits_turn", "admits_interruption",
]
