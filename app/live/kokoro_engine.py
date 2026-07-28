"""
GPU-local Kokoro TTS for the pod (the `set_kokoro` seam in tts_synth.py).

WHY THIS EXISTS: the only other engine is Edge Neural — Microsoft's CLOUD
service. On a pod that means every spoken sentence travels
client → pod → Microsoft → pod → client, i.e. two internet round trips per
sentence, on top of the LLM. That is the dominant reason spoken replies feel
laggy and choppy next to ChatGPT voice, which generates audio co-located with
the model. Kokoro-82M is tiny (~350 MB) and runs on the pod's GPU right next to
the LLM, so synthesis becomes a local call.

Emits **MP3** because the client writes the bytes to a `.mp3` temp file and
hands it to the platform player; matching Edge's format keeps that path
unchanged.

STRICTLY FAIL-OPEN: any missing dependency, model-load failure, or synthesis
error returns b"", and `tts_synth.synthesize()` then degrades to Edge exactly as
it does today. Registering this can therefore never make the voice surface worse
than not registering it.
"""
from __future__ import annotations

import logging
import threading

log = logging.getLogger("zapthetrick.tts.kokoro")

_SAMPLE_RATE = 24_000          # Kokoro's native output rate
_pipeline = None
_lock = threading.Lock()
_failed = False                # latch: don't retry a hopeless import every turn


def available() -> bool:
    """True when the Kokoro runtime + MP3 encoder are importable."""
    try:
        import kokoro  # noqa: F401
        import lameenc  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def _get_pipeline():
    """Load the pipeline once (single-flight — two concurrent turns must not
    both pay the model load). Returns None once a load has failed."""
    global _pipeline, _failed
    if _failed:
        return None
    if _pipeline is not None:
        return _pipeline
    with _lock:
        if _pipeline is None and not _failed:
            try:
                from kokoro import KPipeline
                _pipeline = KPipeline(lang_code="a")   # 'a' = American English
                log.info("kokoro TTS pipeline loaded")
            except Exception as exc:  # noqa: BLE001
                _failed = True
                log.warning("kokoro unavailable (%s) — falling back to Edge", exc)
    return _pipeline


def _to_mp3(pcm_f32) -> bytes:
    """float32 [-1,1] mono @24k → MP3 bytes."""
    import lameenc
    import numpy as np
    pcm = np.clip(np.asarray(pcm_f32, dtype="float32"), -1.0, 1.0)
    pcm16 = (pcm * 32767.0).astype("<i2")
    enc = lameenc.Encoder()
    enc.set_bit_rate(64)                  # speech: 64 kbps mono is transparent
    enc.set_in_sample_rate(_SAMPLE_RATE)
    enc.set_channels(1)
    enc.set_quality(5)                    # 2=best/slow … 7=fast; 5 is the balance
    return bytes(enc.encode(pcm16.tobytes()) + enc.flush())


def synth(text: str, voice: str, speed: float) -> bytes:
    """(text, kokoro voice id, speed) -> MP3 bytes. b"" on ANY failure so the
    caller degrades to Edge. Matches the `set_kokoro` callable contract."""
    t = (text or "").strip()
    if not t:
        return b""
    # This pipeline is ENGLISH (lang_code='a'); Edge Neural has language-matched
    # voices (_edge_voice picks a Hindi/Telugu/… voice from the text). A
    # non-English reply must therefore fall back to Edge — b"" triggers exactly
    # that in tts_synth.synthesize — or the multilingual voice feature would
    # come out mangled through an English G2P.
    try:
        from app.live.language import detect_language
        if detect_language(t) != "en":
            return b""
    except Exception:  # noqa: BLE001 — detector unavailable → assume English
        pass
    try:
        pipe = _get_pipeline()
        if pipe is None:
            return b""
        import numpy as np
        chunks = []
        for item in pipe(t, voice=voice, speed=float(speed or 1.0)):
            # KPipeline yields (graphemes, phonemes, audio); tolerate a bare
            # audio yield from a different minor version.
            audio = item[-1] if isinstance(item, (tuple, list)) else item
            if audio is None:
                continue
            arr = (audio.detach().cpu().numpy()
                   if hasattr(audio, "detach") else np.asarray(audio))
            if arr.size:
                chunks.append(arr.reshape(-1))
        if not chunks:
            return b""
        return _to_mp3(np.concatenate(chunks))
    except Exception as exc:  # noqa: BLE001
        log.info("kokoro synth failed (%s) — Edge fallback", exc)
        return b""


def register_if_available() -> bool:
    """Wire this engine into `tts_synth` when the runtime is present. Returns
    whether it was registered. Safe to call unconditionally at startup."""
    try:
        if not available():
            return False
        from app.live.tts_synth import set_kokoro
        set_kokoro(synth)
        return True
    except Exception:  # noqa: BLE001
        return False


__all__ = ["synth", "available", "register_if_available"]
