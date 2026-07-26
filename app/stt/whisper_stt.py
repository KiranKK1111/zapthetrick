"""
faster-whisper STT.

CTranslate2-backed Whisper — ~4x faster than the reference openai-whisper
implementation with identical accuracy. Runs on CPU (int8) or GPU
(float16); selected by `cfg.stt.device` and `cfg.stt.compute_type`.

For live use the audio handler chunks incoming audio with VAD, then
calls `transcribe` on each speech segment. True streaming with stable
partials needs an overlapping-window scheduler (whisper_streaming
library) — out of Phase-4 scope but planned.
"""
from __future__ import annotations

from functools import lru_cache

from app.core.config_loader import cfg


class STTError(RuntimeError):
    """Raised when the STT model cannot be loaded or used."""


@lru_cache(maxsize=1)
def _model():
    """Lazy-load the faster-whisper model. ~80MB for base.en."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise STTError(
            "faster-whisper is not installed. Run: pip install faster-whisper"
        ) from exc
    # GPU-or-CPU device resolution (R60) — latency-only, clean CPU fallback.
    try:
        from app.stt.factory import resolve_device
        _device, _compute_type = resolve_device()
    except Exception:  # noqa: BLE001
        _device, _compute_type = cfg.stt.device, cfg.stt.compute_type
    return WhisperModel(
        cfg.stt.model,
        device=_device,
        compute_type=_compute_type,
        cpu_threads=getattr(cfg.stt, "cpu_threads", 4),
    )


def _initial_prompt() -> str | None:
    """Compose Whisper's `initial_prompt` from the vocabulary booster.

    Architecture.md §"Vocabulary boosting" — feeds proper-noun-shaped
    terms from the active resume + COPILOT.md + session terms so the
    decoder is biased toward common interview jargon.

    Returns None when boosting is disabled, the module errors, or
    yields no terms — Whisper accepts `initial_prompt=None` as a no-op.
    """
    if not getattr(cfg.stt, "vocab_boost_enabled", True):
        return None
    try:
        from .vocabulary_boost import build_initial_prompt

        text = build_initial_prompt()
        return text or None
    except Exception:  # noqa: BLE001 — boosting is non-essential
        return None


def _hotwords() -> str | None:
    """A space-separated boost list for faster-whisper's `hotwords`.

    `hotwords` biases the decoder more strongly than `initial_prompt`
    toward these exact terms — the fix for "bean lifecycle" decoding as
    "being lifecycle". Pulls the same ranked pool the initial_prompt uses.
    """
    if not getattr(cfg.stt, "hotwords_enabled", True):
        return None
    try:
        from .vocabulary_boost import build_boost_list

        # Keep the active bias FOCUSED — Whisper conditions on a small prompt
        # budget (~224 tokens) and overloading it dilutes accuracy AND slows
        # decoding (measured: ~220 terms dropped accuracy and doubled time vs
        # a focused set). The candidate's resume/session terms rank first, then
        # the most-mangled names. The acoustic model handles common words on
        # its own; the full 720-term pack is for coverage, not all-at-once bias.
        terms = build_boost_list(limit=64)
        return " ".join(terms) if terms else None
    except Exception:  # noqa: BLE001 — boosting is non-essential
        return None


def _decode_kwargs() -> dict:
    """Shared faster-whisper decode options tuned for accuracy on short
    interview utterances."""
    return {
        "language": cfg.stt.language,
        # VAD is done upstream — let Whisper see all of what the VAD chose.
        "vad_filter": False,
        "beam_size": cfg.stt.beam_size,
        "initial_prompt": _initial_prompt(),
        "hotwords": _hotwords(),
        # Each utterance is independent; conditioning on prior text makes
        # Whisper hallucinate continuations and repeat itself on short clips.
        "condition_on_previous_text": False,
        # Deterministic greedy/beam decode — no temperature sampling drift.
        "temperature": 0.0,
    }


# English + the major Indian languages — the plausible set for this app. Whisper
# auto-detect on a SHORT clip ("Hi") misfires to Japanese/Korean/Chinese; pinning
# detection to this whitelist keeps a short English/Indian utterance from being
# transcribed as a random unrelated language.
_ALLOWED_LANGS_DEFAULT = ["en", "hi", "te", "ta", "kn", "ml", "bn", "mr", "gu",
                          "pa", "ur", "or", "as"]


def _resolve_language(model, audio_np) -> str | None:
    """Pick the transcription language. An explicit `stt.language` wins. When it
    is None (auto), detect the language but CONSTRAIN it to the allowed set so a
    short English clip can't be read as Japanese — pick the highest-probability
    language among the whitelist, with a mild English bias for ambiguous clips
    (this app's speakers are mostly English + Indian languages)."""
    pinned = getattr(cfg.stt, "language", None)
    if pinned:
        return pinned
    allowed = list(getattr(cfg.stt, "allowed_languages", None)
                   or _ALLOWED_LANGS_DEFAULT)
    if not allowed:
        return None  # truly unconstrained auto-detect
    try:
        _lang, _prob, all_probs = model.detect_language(audio_np)
    except Exception:  # noqa: BLE001 — fall back to a safe default
        return "en" if "en" in allowed else allowed[0]
    probs = dict(all_probs or [])
    best = max(allowed, key=lambda code: probs.get(code, 0.0))
    # English bias ONLY for SHORT clips (< ~2s), where detection is noisy and a
    # quick English "Hi" gets misread. For a real sentence — including genuine
    # Hindi/Telugu or a code-switched mix — trust the whitelisted detection so
    # it's transcribed in the RIGHT language/script (pinning English would
    # garble it). Whisper still transcribes any in-utterance language mixing.
    try:
        duration_s = float(len(audio_np)) / 16000.0
    except Exception:  # noqa: BLE001
        duration_s = 99.0
    if (duration_s < 2.0 and "en" in allowed
            and probs.get("en", 0.0) >= probs.get(best, 0.0) - 0.15):
        return "en"
    return best


def transcribe(audio_np) -> str:
    """Transcribe a 16-kHz float32 mono numpy array. Returns plain text."""
    model = _model()
    kwargs = _decode_kwargs()
    kwargs["language"] = _resolve_language(model, audio_np)
    segments, _info = model.transcribe(audio_np, **kwargs)
    return " ".join(s.text.strip() for s in segments if s.text.strip())


def transcribe_with_timings(audio_np) -> list[dict]:
    """Same as `transcribe` but returns per-segment timing for the UI."""
    model = _model()
    segments, _ = model.transcribe(audio_np, **_decode_kwargs())
    return [
        {"start": s.start, "end": s.end, "text": s.text.strip()}
        for s in segments
        if s.text.strip()
    ]
