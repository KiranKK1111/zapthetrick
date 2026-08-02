"""Speaker verification gate — Phase 4 (Requirement 12).

The property that matters most here is the NEGATIVE one: with the gate off, no
enrolment, or no embedder, behaviour must be **exactly** what it is today. The
embedder genuinely does not exist in this checkout, so a gate that failed closed
would silently mute every user on every non-pod deployment.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.voice import speaker as S
from app.voice.speaker import SpeakerProfile


def _audio(seconds: float) -> np.ndarray:
    return np.zeros(int(S.SAMPLE_RATE * seconds), dtype=np.float32)


@pytest.fixture()
def gate_on(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.voice, "speaker_gate", True, raising=False)
    monkeypatch.setattr(cfg.voice, "speaker_threshold", 0.65, raising=False)


@pytest.fixture()
def fake_embedder(monkeypatch):
    """Install a deterministic embedder. Returns a dict the test writes the
    NEXT vector into, so a session can present different 'speakers'."""
    box = {"vec": (1.0, 0.0, 0.0)}
    monkeypatch.setattr(S, "_embed", lambda audio: box["vec"])
    return box


# ── Requirement 12.3: absent machinery ⇒ current behaviour ──────────────────

def test_gate_disabled_admits_everything():
    assert S.verify(SpeakerProfile(), _audio(2)).admit is True
    assert S.admits_turn(None, _audio(2)) is True


def test_no_enrolment_admits(gate_on):
    v = S.verify(SpeakerProfile(), _audio(2))
    assert v.admit is True and v.reason == "not_enrolled"


def test_absent_embedder_admits(gate_on, monkeypatch):
    """`app/live/speaker_embed.py` is genuinely missing from this checkout. A
    gate that failed CLOSED here would mute every user on every non-pod build."""
    monkeypatch.setattr(S, "_embed", lambda audio: None)
    prof = SpeakerProfile(embedding=(1.0, 0.0), samples=1)
    v = S.verify(prof, _audio(2))
    assert v.admit is True and v.reason == "no_embedder"
    assert v.informed is False


def test_the_real_embedder_really_is_absent_here():
    """Documents the checkout's actual state — if someone adds the module, this
    flips and Phase 4 becomes live rather than inert."""
    assert S.available() is False


# ── Verification ────────────────────────────────────────────────────────────

def test_the_enrolled_speaker_is_admitted(gate_on, fake_embedder):
    prof = SpeakerProfile()
    prof, ok = S.enrol(prof, _audio(4))
    assert ok and prof.enrolled
    v = S.verify(prof, _audio(2))
    assert v.admit is True and v.reason == "match" and v.score > 0.99


def test_a_bystander_is_rejected(gate_on, fake_embedder):
    prof = SpeakerProfile()
    prof, _ = S.enrol(prof, _audio(4))
    fake_embedder["vec"] = (0.0, 1.0, 0.0)      # an orthogonal voice
    v = S.verify(prof, _audio(2))
    assert v.admit is False and v.reason == "mismatch"
    assert v.informed is True


def test_a_bystander_can_neither_take_a_turn_nor_interrupt(gate_on,
                                                           fake_embedder):
    """Both doors. Gating only turn admission would leave the interruption path
    open, and cutting the assistant off is the more disruptive of the two."""
    prof = SpeakerProfile()
    prof, _ = S.enrol(prof, _audio(4))
    fake_embedder["vec"] = (-1.0, 0.0, 0.0)
    assert S.admits_turn(prof, _audio(2)) is False
    assert S.admits_interruption(prof, _audio(2)) is False


def test_threshold_is_configurable(gate_on, fake_embedder, monkeypatch):
    from app.core.config_loader import cfg
    prof = SpeakerProfile()
    prof, _ = S.enrol(prof, _audio(4))
    fake_embedder["vec"] = (0.8, 0.6, 0.0)      # cos ≈ 0.8 against (1,0,0)
    monkeypatch.setattr(cfg.voice, "speaker_threshold", 0.9, raising=False)
    assert S.verify(prof, _audio(2)).admit is False
    monkeypatch.setattr(cfg.voice, "speaker_threshold", 0.7, raising=False)
    assert S.verify(prof, _audio(2)).admit is True


# ── Duration floors ─────────────────────────────────────────────────────────

def test_a_short_utterance_is_admitted_not_judged(gate_on, fake_embedder):
    """A one-word answer ("yes") is a real turn. Rejecting it because there was
    too little audio to verify would drop legitimate speech."""
    prof = SpeakerProfile()
    prof, _ = S.enrol(prof, _audio(4))
    fake_embedder["vec"] = (0.0, 1.0, 0.0)      # would MISMATCH if judged
    v = S.verify(prof, _audio(0.2))
    assert v.admit is True and v.reason == "too_short"


def test_enrolment_rejects_a_clip_that_is_too_short(fake_embedder):
    """A short, noisy embedding poisons the centroid permanently and every later
    verification pays for it."""
    prof, ok = S.enrol(SpeakerProfile(), _audio(0.5))
    assert ok is False and not prof.enrolled


# ── Profile mechanics ───────────────────────────────────────────────────────

def test_enrolment_averages_across_samples(fake_embedder):
    prof = SpeakerProfile()
    prof, _ = S.enrol(prof, _audio(4))
    fake_embedder["vec"] = (0.0, 1.0, 0.0)
    prof, _ = S.enrol(prof, _audio(4))
    assert prof.samples == 2
    assert prof.embedding == pytest.approx((0.5, 0.5, 0.0))


def test_profile_round_trips_through_a_plain_dict():
    """It rides `User.preferences`; Requirement 11.4 forbids new schema."""
    prof = SpeakerProfile(embedding=(0.1, 0.2, 0.3), samples=3)
    back = SpeakerProfile.from_dict(prof.to_dict())
    assert back.embedding == pytest.approx((0.1, 0.2, 0.3))
    assert back.samples == 3 and back.enrolled


@pytest.mark.parametrize("bad", [None, {}, {"embedding": "nope"},
                                 {"embedding": [1, "x"]}])
def test_a_corrupt_stored_profile_reads_as_not_enrolled(bad):
    assert SpeakerProfile.from_dict(bad).enrolled is False


def test_similarity_is_bounded_and_safe():
    assert S.similarity((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert S.similarity((1.0, 0.0), (-1.0, 0.0)) == pytest.approx(-1.0)
    assert S.similarity((), (1.0,)) == 0.0            # empty
    assert S.similarity((0.0, 0.0), (1.0, 0.0)) == 0.0  # degenerate, no div/0
    assert S.similarity((1.0,), (1.0, 0.0)) == 0.0    # length mismatch


# ── Wiring into the engine ──────────────────────────────────────────────────

def test_the_staged_engine_consults_the_gate(monkeypatch):
    """The gate is only real if the engine actually calls it."""
    import asyncio

    from app.voice.engine import TranscriptDelta, VoiceContext
    from app.voice.staged import StagedEngine

    seen = {"turn": 0}

    def _admits(profile, audio):
        seen["turn"] += 1
        return False                      # a bystander

    monkeypatch.setattr("app.voice.staged._speaker.admits_turn", _admits)

    async def body():
        sess = await StagedEngine().open(VoiceContext(conversation_id="c"))
        await sess.on_utterance("what is kafka", None)
        assert seen["turn"] == 1, "the engine never consulted the speaker gate"
        # …and nothing was emitted, so the bystander took no turn.
        try:
            ev = await asyncio.wait_for(sess.events().__anext__(), timeout=0.3)
        except asyncio.TimeoutError:
            ev = None
        assert not isinstance(ev, TranscriptDelta)
        await sess.close()

    asyncio.run(body())
