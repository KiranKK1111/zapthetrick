"""Cloud STT via Groq's Whisper endpoint — used when `cfg.stt.mode == "cloud"`.

The local chain (Parakeet/Qwen-ASR) stays the default; this is opt-in from the
Settings toggle. It reuses the Groq key already stored (encrypted) in the app's
keystore — no new key to manage. Encodes the numpy audio to a WAV in memory and
POSTs it. Never used unless the toggle is on; raises on failure so the factory
falls back to the local chain.
"""
from __future__ import annotations

import io
import logging

log = logging.getLogger(__name__)

_GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"


async def _groq_key() -> str | None:
    """The first healthy Groq API key from the keystore (decrypted)."""
    try:
        from sqlalchemy import select

        from app.database import get_session_factory
        from app.llm import crypto
        from storage.models import LLMApiKey
        f = get_session_factory()
        if f is None:
            return None
        try:
            await crypto.ensure_initialized()
        except Exception:  # noqa: BLE001
            pass
        async with f() as s:
            rows = (await s.execute(
                select(LLMApiKey).where(
                    LLMApiKey.platform == "groq",
                    LLMApiKey.enabled.is_(True),
                ))).scalars().all()
            for k in rows:
                try:
                    return crypto.decrypt(k.encrypted_key, k.iv, k.auth_tag)
                except Exception:  # noqa: BLE001
                    continue
    except Exception as exc:  # noqa: BLE001
        log.info("cloud STT: key lookup failed (%s)", exc)
    return None


def _to_wav(audio_np, sample_rate: int) -> bytes:
    import numpy as np
    import soundfile as sf
    buf = io.BytesIO()
    sf.write(buf, np.asarray(audio_np, dtype="float32"), sample_rate,
             format="WAV", subtype="PCM_16")
    return buf.getvalue()


async def transcribe(audio_np, prompt: str | None = None,
                     sample_rate: int = 16000) -> str:
    """Transcribe `audio_np` (float32 mono) via Groq Whisper. Raises on any
    failure so the caller falls back to the local chain."""
    key = await _groq_key()
    if not key:
        raise RuntimeError("cloud STT is on but no Groq API key is configured "
                           "(add one in Settings → Providers)")
    import httpx

    from app.core.config_loader import cfg
    model = str(getattr(cfg.stt, "cloud_model", "whisper-large-v3-turbo")
                or "whisper-large-v3-turbo")
    wav = _to_wav(audio_np, sample_rate)
    data = {"model": model, "response_format": "json"}
    if prompt:
        data["prompt"] = prompt[:500]
    lang = getattr(cfg.stt, "language", None)
    if lang and str(lang).lower() not in ("", "auto"):
        data["language"] = str(lang)
    async with httpx.AsyncClient(timeout=60.0) as c:
        r = await c.post(
            _GROQ_URL,
            headers={"Authorization": f"Bearer {key}"},
            files={"file": ("audio.wav", wav, "audio/wav")},
            data=data)
        r.raise_for_status()
        return (r.json().get("text") or "").strip()
