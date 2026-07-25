"""Neural text-to-speech (§10.5) — natural audio for the client.

`POST /api/tts` turns speech-ready text into natural MP3 audio using the neural
engine (Kokoro on the pod, Edge Neural on the dev box — see `app/live/
tts_synth.py`). The client plays the returned audio for read-aloud + voice mode.

Flag-gated (`voice.tts`, default on) → 503 when off. Never leaks a stack trace.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app.live import tts_synth

log = logging.getLogger("routes_tts")
router = APIRouter(prefix="/api/tts", tags=["tts"])


class TtsRequest(BaseModel):
    text: str = Field(..., description="Speech-ready text to voice.")
    voice: str = Field("nova", description="A voice reference id "
                                           "(aria/nova/…/atlas/…).")
    speed: float = Field(1.0, ge=0.5, le=2.0,
                         description="Playback speed multiplier.")


@router.post("")
async def synthesize(body: TtsRequest) -> Response:
    if not tts_synth.enabled():
        raise HTTPException(503, detail="Voice output isn't enabled.")
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, detail="Nothing to speak.")
    # Bound a single request so a giant paste can't tie up synthesis.
    if len(text) > 6000:
        text = text[:6000]
    try:
        audio = await tts_synth.synthesize(text, body.voice, speed=body.speed)
    except Exception:  # noqa: BLE001 — never leak a trace to the client
        log.warning("tts synthesis failed", exc_info=True)
        audio = b""
    if not audio:
        # The neural engine was unreachable — the client keeps its text answer.
        raise HTTPException(503, detail="The voice engine is unavailable.")
    return Response(content=audio, media_type="audio/mpeg")
