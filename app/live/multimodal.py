"""
Extensible Multimodal_Input adapter (live-conversational-intelligence R46).

An adapter that normalizes non-audio inputs (shared screen text, pasted code, a
typed question, a chat message) into the SAME utterance shape the live event
pipeline already consumes. When no multimodal source is present the system is
audio-only (no behavior change). New modalities register a normalizer — the
pipeline stays untouched. Deterministic + fail-open.
"""
from __future__ import annotations

from dataclasses import dataclass

# Modality identifiers.
AUDIO = "audio"
SCREEN_TEXT = "screen_text"
PASTED_CODE = "pasted_code"
TYPED = "typed"
CHAT = "chat"


@dataclass
class MultimodalInput:
    modality: str
    text: str
    meta: dict


# modality → normalizer(raw) -> text. Extensible: register more without touching
# the pipeline.
def _norm_text(raw) -> str:
    return str(raw or "").strip()


def _norm_code(raw) -> str:
    t = str(raw or "").strip()
    # Surface code as an utterance the question pipeline can reason about.
    return f"[shared code]\n{t}" if t else ""


_NORMALIZERS = {
    SCREEN_TEXT: _norm_text,
    PASTED_CODE: _norm_code,
    TYPED: _norm_text,
    CHAT: _norm_text,
}


def register_modality(name: str, normalizer) -> None:
    """Register a new modality normalizer (extensibility hook). Never raises."""
    try:
        _NORMALIZERS[str(name)] = normalizer
    except Exception:  # noqa: BLE001
        pass


def to_utterance(modality: str, raw, meta: dict | None = None) -> MultimodalInput | None:
    """Normalize a non-audio input into an utterance. Returns None for empty /
    unknown / audio (audio uses the existing path). Never raises."""
    try:
        m = (modality or "").strip().lower()
        if m == AUDIO or m not in _NORMALIZERS:
            return None
        text = _NORMALIZERS[m](raw)
        if not text:
            return None
        return MultimodalInput(modality=m, text=text, meta=dict(meta or {}))
    except Exception:  # noqa: BLE001
        return None


def supported() -> list[str]:
    return [AUDIO, *sorted(_NORMALIZERS.keys())]
