"""Neural TTS synthesis (§10.5) — natural, human voices for the client.

`tts_lane.to_spoken` produces the speech-ready TEXT; this turns that text into
natural AUDIO. Two engines behind one interface:

  * **Kokoro** (on the GPU pod) — the high-quality on-device neural voice; the
    real production path. Injected/optional; absent on the dev box.
  * **Edge Neural** (`edge-tts`) — Microsoft Edge's online neural voices, FREE
    and key-less, very natural. The dev-box + fallback path.

The ten named references (`aria`… / `atlas`…) map to a neural voice per engine.
Returns MP3 bytes. Pure I/O + fail-open (returns b"" on any error so the client
falls back to its local OS voice). Flag-gated (`voice.tts`, default reads cfg).
"""
from __future__ import annotations

import asyncio

# The ten client references → (edge neural voice, kokoro voice). Female first.
# We use Azure's flagship *Multilingual* neural voices where available — they're
# markedly more natural/expressive (ChatGPT-tier) than the older -Neural voices.
_VOICES: dict[str, tuple[str, str]] = {
    "aria": ("en-US-AvaMultilingualNeural", "af_bella"),
    "nova": ("en-US-EmmaMultilingualNeural", "af_sarah"),
    "luna": ("en-US-JennyNeural", "af_nicole"),
    "ivy": ("en-GB-SoniaNeural", "bf_emma"),
    "sol": ("en-US-AriaNeural", "af_sky"),
    "atlas": ("en-US-AndrewMultilingualNeural", "am_adam"),
    "orion": ("en-US-BrianMultilingualNeural", "am_michael"),
    "cove": ("en-US-GuyNeural", "am_echo"),
    "vale": ("en-GB-RyanNeural", "bm_george"),
    "ezra": ("en-US-ChristopherNeural", "am_onyx"),
}
_DEFAULT = "nova"

# The English references' gender (female first five, then male five).
_FEMALE_PRESETS = {"aria", "nova", "luna", "ivy", "sol"}

# Non-English → (female neural voice, male neural voice). When the ANSWER is in
# one of these languages, we voice it with the matching neural voice for the
# chosen reference's gender (Edge has natural voices for all of these).
_LANG_VOICES: dict[str, tuple[str, str]] = {
    "hi": ("hi-IN-SwaraNeural", "hi-IN-MadhurNeural"),        # Hindi
    "te": ("te-IN-ShrutiNeural", "te-IN-MohanNeural"),        # Telugu
    "ta": ("ta-IN-PallaviNeural", "ta-IN-ValluvarNeural"),    # Tamil
    "bn": ("bn-IN-TanishaaNeural", "bn-IN-BashkarNeural"),    # Bengali
    "kn": ("kn-IN-SapnaNeural", "kn-IN-GaganNeural"),         # Kannada
    "ml": ("ml-IN-SobhanaNeural", "ml-IN-MidhunNeural"),      # Malayalam
    "gu": ("gu-IN-DhwaniNeural", "gu-IN-NiranjanNeural"),     # Gujarati
    "mr": ("mr-IN-AarohiNeural", "mr-IN-ManoharNeural"),      # Marathi
    "pa": ("pa-IN-OjasNeural", "pa-IN-OjasNeural"),           # Punjabi
    "ur": ("ur-IN-GulNeural", "ur-IN-SalmanNeural"),          # Urdu
}

# Unicode blocks → language, so we voice a reply in the script it's written in.
_SCRIPT_RANGES = [
    (0x0C00, 0x0C7F, "te"), (0x0B80, 0x0BFF, "ta"), (0x0980, 0x09FF, "bn"),
    (0x0C80, 0x0CFF, "kn"), (0x0D00, 0x0D7F, "ml"), (0x0A80, 0x0AFF, "gu"),
    (0x0A00, 0x0A7F, "pa"), (0x0600, 0x06FF, "ur"), (0x0900, 0x097F, "hi"),
]


def _detect_lang(text: str) -> str:
    """The language to voice `text` in, from the DOMINANT non-Latin script in it
    (so mixed/code-switched replies — Hinglish, Hindi+Telugu+English, etc. — are
    voiced in the primary Indian language, whose neural voice also handles the
    embedded English words naturally). Pure Latin → 'en'. Never raises."""
    try:
        counts: dict[str, int] = {}
        for ch in text:
            o = ord(ch)
            for lo, hi, lang in _SCRIPT_RANGES:
                if lo <= o <= hi:
                    counts[lang] = counts.get(lang, 0) + 1
                    break
        if not counts:
            return "en"
        lang, n = max(counts.items(), key=lambda kv: kv[1])
        return lang if n >= 2 else "en"
    except Exception:  # noqa: BLE001
        return "en"


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.voice, "tts", False))
    except Exception:  # noqa: BLE001
        return False


def _edge_voice(voice_id: str, text: str = "") -> str:
    """The Edge Neural voice to use. If the answer is in a non-English language,
    switch to that language's neural voice for the chosen reference's GENDER; an
    English answer uses the reference's English voice."""
    vid = (voice_id or "").strip().lower() or _DEFAULT
    lang = _detect_lang(text) if text else "en"
    if lang != "en" and lang in _LANG_VOICES:
        female = vid in _FEMALE_PRESETS
        f, m = _LANG_VOICES[lang]
        return f if female else m
    return _VOICES.get(vid, _VOICES[_DEFAULT])[0]


def _kokoro_voice(voice_id: str) -> str:
    return _VOICES.get((voice_id or "").strip().lower() or _DEFAULT,
                       _VOICES[_DEFAULT])[1]


def _rate_str(speed: float) -> str:
    """A [speedOptions] multiplier → edge-tts `rate` (e.g. 1.25 → "+25%")."""
    try:
        pct = int(round((float(speed) - 1.0) * 100))
    except Exception:  # noqa: BLE001
        pct = 0
    pct = max(-50, min(100, pct))
    return f"{pct:+d}%"


async def _synth_edge(text: str, voice_id: str, *, speed: float) -> bytes:
    """Edge Neural (free, no key). Returns MP3 bytes; b"" on failure."""
    try:
        import edge_tts  # lazy — only when we actually synthesize
        comm = edge_tts.Communicate(
            text, _edge_voice(voice_id, text), rate=_rate_str(speed))
        buf = bytearray()
        async for chunk in comm.stream():
            if chunk.get("type") == "audio" and chunk.get("data"):
                buf += chunk["data"]
        return bytes(buf)
    except Exception:  # noqa: BLE001
        return b""


# Kokoro seam — replaced on the pod with the real GPU synthesizer. Injectable so
# the pod wires a `kokoro_fn(text, voice, speed) -> bytes` without touching this.
_kokoro_fn = None


def set_kokoro(fn) -> None:
    """Register the on-pod Kokoro synthesizer (text, voice, speed) -> bytes."""
    global _kokoro_fn
    _kokoro_fn = fn


async def _synth_kokoro(text: str, voice_id: str, *, speed: float) -> bytes:
    fn = _kokoro_fn
    if fn is None:
        return b""
    try:
        out = fn(text, _kokoro_voice(voice_id), speed)
        if asyncio.iscoroutine(out):
            out = await out
        return out if isinstance(out, (bytes, bytearray)) else b""
    except Exception:  # noqa: BLE001
        return b""


def _engine() -> str:
    """The selected TTS engine ("edge" | "kokoro"). Never raises → "edge"."""
    try:
        from app.core.config_loader import cfg
        e = str(getattr(cfg.voice, "tts_engine", "edge") or "edge").strip().lower()
        return e if e in ("edge", "kokoro") else "edge"
    except Exception:  # noqa: BLE001
        return "edge"


async def synthesize(text: str, voice_id: str = _DEFAULT,
                     *, speed: float = 1.0) -> bytes:
    """Natural audio (MP3) for `text` in the chosen reference voice, using the
    configured engine: "kokoro" (pod) when selected AND registered, else "edge"
    (Edge Neural). Kokoro falls back to edge if the pod engine is absent. Returns
    b"" only when both fail. Never raises."""
    t = (text or "").strip()
    if not t:
        return b""
    if _engine() == "kokoro":
        audio = await _synth_kokoro(t, voice_id, speed=speed)
        if audio:
            return bytes(audio)
        # Pod engine not present on this host — degrade to Edge Neural.
    return await _synth_edge(t, voice_id, speed=speed)


def voice_ids() -> list[str]:
    return list(_VOICES.keys())


__all__ = ["enabled", "synthesize", "set_kokoro", "voice_ids"]
