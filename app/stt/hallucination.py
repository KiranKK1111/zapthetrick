"""Suppress ASR hallucinations on silence.

The problem, straight from a live session log
---------------------------------------------
Of 22 partial transcripts in one real interview, **13 were "Thank you." and one
was "Thanks for watching!"** — 59% of everything the recognizer produced. Nobody
said any of it.

This is a well-known Whisper artifact, not a bug in the audio path. Whisper was
trained largely on captioned video, so when it is handed silence, room tone, a
fan or a breath it emits the most frequent closers in its training data. The
model is doing exactly what it was trained to do; it just has nothing to
transcribe.

Why catching it downstream was not enough
-----------------------------------------
The live pipeline already discarded these — the log shows `skipped … reason:
feedback` ten times, because a bare "Thank you." reads as satisfaction rather
than a question. But by then each one had already become an utterance, taken a
qid, and cost a round trip. Suppressing at the source means the pipeline never
sees them at all.

Why this list is literal, and why that is right here
----------------------------------------------------
These are not intents to be classified — they are specific memorised artifacts
of a specific model's training set, the same closed-set reasoning that makes a
filler list appropriate. A semantic gate would be the wrong tool: "thank you" is
genuinely ambiguous in meaning, and the thing that disambiguates it is not what
it means but **whether any speech energy was present when it was produced**.
That is the real discriminator, and it is acoustic.
"""
from __future__ import annotations

import re

# Verbatim closers Whisper emits on non-speech audio. Normalised (lowercase, no
# punctuation) before comparison, so "Thank you." and "thank you" are one entry.
_HALLUCINATIONS = frozenset({
    "thank you",
    "thanks",
    "thank you very much",
    "thank you so much",
    "thanks for watching",
    "thanks for watching!",
    "thank you for watching",
    "thanks for listening",
    "please subscribe",
    "subscribe to my channel",
    "like and subscribe",
    "see you next time",
    "see you in the next video",
    "bye",
    "bye bye",
    "you",
    "the end",
    "outro",
    "music",
    "silence",
    "applause",
    # Bracketed sound tags the decoder emits for non-speech.
    "[music]", "[silence]", "[applause]", "[blank_audio]", "[inaudible]",
    "(music)", "(silence)", "(applause)",
})

_PUNCT = re.compile(r"[^\w\s\[\]()]+")

# RMS below which a segment is treated as containing no real speech.
#
# MEASURED, and the measurement is why this is only the SECONDARY gate. Room
# tone spans a huge range depending on mic gain and AGC:
#
#     digital silence  0.0000   quiet room   0.0009
#     typical room     0.0046   noisy/fan    0.0122
#     AGC-boosted      0.0276   quiet speech 0.0612
#
# There is no single value that catches an AGC-boosted room without also eating
# quiet speech — the two overlap. So the authoritative gate is the VAD's own
# voiced-sample count in `app/audio/stream.py`, which is level-independent
# because it is the same decision the segmenter already made about this audio.
#
# This threshold remains as a defence for callers with no VAD context, set just
# above typical room tone. It is deliberately conservative: it will MISS a
# hallucination in a loud room rather than risk discarding quiet speech.
SILENCE_RMS = 0.008


def normalise(text: str) -> str:
    return _PUNCT.sub("", (text or "").strip().lower()).strip()


def is_hallucination_phrase(text: str) -> bool:
    """Whether the transcript is ENTIRELY a known artifact.

    Whole-string only, deliberately. "Thank you, now what is a hash map?" is a
    real utterance that happens to open with the phrase, and dropping it would
    lose a genuine question — a far worse failure than letting one artifact
    through.
    """
    return normalise(text) in _HALLUCINATIONS


def speech_energy(audio) -> float:
    """RMS of the segment, or -1.0 when it cannot be measured.

    -1.0 means "unknown", and every caller treats unknown as *speech present* —
    suppressing a real question because energy could not be computed would be
    the worse mistake.
    """
    try:
        import numpy as np
        arr = np.asarray(audio, dtype="float32").reshape(-1)
        if arr.size == 0:
            return 0.0
        # int16-scaled input arrives from some engines; normalise it.
        if float(np.max(np.abs(arr))) > 1.5:
            arr = arr / 32768.0
        return float(np.sqrt(np.mean(arr * arr)))
    except Exception:  # noqa: BLE001
        return -1.0


def is_hallucination(text: str, audio=None,
                     silence_rms: float = SILENCE_RMS) -> bool:
    """Whether this transcript should be discarded as an ASR artifact.

    Two conditions, and BOTH are required:

    1. the text is entirely a known artifact phrase, and
    2. the audio carried no real speech energy.

    Requiring both is what makes this safe. A candidate who genuinely says
    "thank you" is audible, so the energy test passes and their words survive.
    Only a phrase invented out of silence is dropped. Where audio is not
    available the phrase test alone applies — those phrases are never a
    meaningful interview turn on their own, which is exactly why the live
    pipeline was already discarding them downstream.
    """
    if not is_hallucination_phrase(text):
        return False
    if audio is None:
        return True
    rms = speech_energy(audio)
    if rms < 0:
        return False          # unmeasurable ⇒ assume real speech
    return rms < silence_rms


__all__ = ["is_hallucination", "is_hallucination_phrase", "speech_energy",
           "normalise", "SILENCE_RMS"]
