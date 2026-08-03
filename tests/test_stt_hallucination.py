"""ASR hallucination suppression — from a real deployed session.

The evidence: of 22 partial transcripts in one live interview on the pod, **13
were "Thank you." and one was "Thanks for watching!"** — 59% of everything the
recognizer produced, none of it spoken. Whisper was trained largely on captioned
video, so handed silence it emits the most frequent closers in its training data.

The pipeline already discarded these downstream (`skipped … reason: feedback`),
but only after each had become an utterance, taken a qid and cost a round trip.

The safety property that matters most here is the INVERSE one: a candidate who
genuinely says "thank you" is audible, and dropping their words would be a worse
failure than letting an artifact through. Both directions are pinned below.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.stt.hallucination import (
    SILENCE_RMS, is_hallucination, is_hallucination_phrase, normalise,
    speech_energy,
)

SR = 16000


def silence(seconds: float = 1.0) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype="float32")


def room_tone(seconds: float = 1.0) -> np.ndarray:
    """Not digital silence — a real quiet room, which is what actually triggers
    the hallucination."""
    rng = np.random.default_rng(7)
    return (rng.normal(0, 0.001, int(SR * seconds))).astype("float32")


def speech(seconds: float = 1.0, amp: float = 0.25) -> np.ndarray:
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * 150 * t)).astype("float32")


# ── The reported failure ────────────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "Thank you.", "thank you", "Thank you!", "Thanks.",
    "Thanks for watching!", "Thank you for watching",
    "Please subscribe", "Bye bye", "[Music]", "(Applause)",
])
def test_known_artifacts_on_silence_are_dropped(text):
    assert is_hallucination(text, silence()) is True


def test_room_tone_counts_as_silence():
    """Digital silence is rare in practice; a quiet room is the real input."""
    assert is_hallucination("Thank you.", room_tone()) is True


def test_the_exact_pair_from_the_live_log():
    for text in ("Thank you.", "Thanks for watching!"):
        assert is_hallucination(text, room_tone()) is True


# ── The inverse property: never eat real speech ─────────────────────────────

def test_a_spoken_thank_you_SURVIVES():
    """The failure that would be worse than the bug. Someone who actually says
    it is audible, so the energy test keeps their words."""
    assert is_hallucination("Thank you.", speech()) is False


@pytest.mark.parametrize("text", [
    "What is a hash map?",
    "Thank you, now what is a hash map?",
    "Thanks — could you explain polymorphism",
    "Thank you for explaining, and how would you scale it",
])
def test_real_utterances_are_never_dropped(text):
    """Whole-string match only. An utterance that merely OPENS with the phrase
    carries a genuine question, and dropping it loses the turn."""
    assert is_hallucination(text, silence()) is False


def test_unmeasurable_audio_assumes_speech():
    """Suppressing a real question because energy could not be computed is the
    worse mistake, so unknown fails open."""
    assert speech_energy("not audio") == -1.0
    assert is_hallucination("Thank you.", "not audio") is False


def test_without_audio_the_phrase_test_alone_applies():
    # These phrases are never a meaningful interview turn on their own — which
    # is exactly why the live pipeline already discarded them downstream.
    assert is_hallucination("Thank you.", None) is True
    assert is_hallucination("What is Kafka?", None) is False


# ── Mechanics ───────────────────────────────────────────────────────────────

def test_normalisation_collapses_case_and_punctuation():
    assert normalise("  Thank YOU!! ") == "thank you"
    assert is_hallucination_phrase("THANK YOU.") is True


def test_energy_handles_int16_scaled_input():
    """Some engines hand back int16-scaled arrays; treating those as float
    would read every segment as deafeningly loud and disable the gate."""
    loud = (speech() * 32767).astype("float32")
    assert speech_energy(loud) == pytest.approx(speech_energy(speech()), rel=0.05)


def test_silence_sits_below_the_threshold_and_speech_above():
    assert speech_energy(room_tone()) < SILENCE_RMS
    assert speech_energy(speech()) > SILENCE_RMS


def test_empty_text_is_not_treated_as_an_artifact():
    assert is_hallucination("", silence()) is False


# ── The guard is actually wired in ──────────────────────────────────────────

def test_the_stt_entrypoints_apply_the_guard():
    """A filter nothing calls filters nothing. Both public entrypoints must run
    it — partials especially, since that is what the caption shows."""
    import inspect

    from app.stt import factory
    src = inspect.getsource(factory)
    assert src.count("is_hallucination") >= 2, \
        "the STT entrypoints do not both apply the hallucination guard"
    assert "_transcribe_partial_raw" in src, \
        "transcribe_partial was not wrapped"
    assert "_transcribe_with_confidence_raw" in src, \
        "transcribe_with_confidence was not wrapped"


# ── The VAD gate: the one that actually holds on real hardware ──────────────
#
# An absolute RMS threshold was tried first and does not survive measurement.
# Room tone spans a wide range with mic gain and AGC, and an AGC-boosted room
# (0.028) reads LOUDER than quiet speech (0.061 is only 2x above it) — so no
# single level catches the hallucination without risking real speech. The
# segmenter's voiced-sample count is level-independent, because it is the same
# decision the VAD already made about this audio.

def test_the_voiced_sample_floor_is_shorter_than_any_real_sentence():
    from app.audio.stream import _HALLUCINATION_VOICED_MIN
    seconds = _HALLUCINATION_VOICED_MIN / SR
    assert 0.2 <= seconds <= 0.6, (
        f"{seconds:.2f}s — too long risks discarding a real short answer, "
        "too short lets the artifact through")


def test_the_segmenter_gates_partials_on_the_vad_count():
    """A guard that lives only in the STT factory cannot see the VAD verdict.
    The segmenter is where both facts are available at once."""
    import inspect

    from app.audio import stream
    src = inspect.getsource(stream.AudioStreamSegmenter._maybe_spawn_partial)
    assert "is_hallucination_phrase" in src, \
        "partials are not checked against the artifact list"
    assert "_HALLUCINATION_VOICED_MIN" in src, \
        "the VAD voiced-count gate is not applied"


def test_the_rms_threshold_sits_above_typical_room_tone():
    """Documented as the secondary gate: conservative on purpose, so it misses
    a hallucination in a loud room rather than eating quiet speech."""
    assert speech_energy(room_tone()) < SILENCE_RMS < 0.02
