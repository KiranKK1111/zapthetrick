"""End-to-end live-module test against a locally running backend.

Simulates exactly what the Flutter app does:
  1. POST /api/live/sessions            -> session id
  2. WS   /ws/live?session_id=<id>      -> handshake ('ready')
  3. stream int16 PCM frames (500ms)    -> VAD segments -> LOCAL STT
  4. collect events: transcript / meta / token / done

Run (backend must be up on :8000):
    .venv/Scripts/python test_live_e2e.py
"""
from __future__ import annotations

import asyncio
import json
import time
import wave

import httpx
import numpy as np
import websockets

import os as _os
BASE = _os.environ.get("DTT_BASE", "http://127.0.0.1:8000")
WAV = (
    r"C:\Users\kiran\AppData\Local\Temp\claude\d--DTT-Backup"
    r"\7e4a13aa-8c39-4a98-aaf4-ed245da675e5\scratchpad\test_q1.wav"
)
# fixed path typo guard — resolved at runtime below
WAV = WAV.replace("7e4a13aa-8c39-4a98-aaf4-ed245da675e5",
                  "7e4a13aa-8c39-4a98-aaf4-ed245da075e5")

CHUNK_MS = 500
SR = 16000


def pcm_frames() -> list[bytes]:
    with wave.open(WAV, "rb") as w:
        assert w.getframerate() == SR
        pcm = w.readframes(w.getnframes())
    # Append 1.2s of silence so the segmenter's endpoint (400ms) fires.
    pcm += b"\x00" * int(SR * 1.2) * 2
    step = int(SR * CHUNK_MS / 1000) * 2  # bytes per 500ms int16 frame
    return [pcm[i:i + step] for i in range(0, len(pcm), step)]


async def main() -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.post(f"{BASE}/api/live/sessions", json={
            "org_name": "Acme Corp",
            "job_role": "Senior Backend Engineer",
            "job_description": ("We need a Senior Backend Engineer with deep "
                                "Kafka, Kubernetes and microservices "
                                "experience. Java or Python, PostgreSQL, "
                                "Redis, AWS. You will design event-driven "
                                "distributed systems at scale."),
            "notes": "Payments team, technical round",
        })
        r.raise_for_status()
        sid = r.json()["id"]
    print(f"session: {sid} (org details attached)")

    _wsbase = BASE.replace("http://", "ws://").replace("https://", "wss://")
    url = f"{_wsbase}/ws/live?session_id={sid}"
    t_start = time.perf_counter()
    events: list[dict] = []
    answer_by_qid: dict[str, str] = {}
    transcript_at: float | None = None

    async with websockets.connect(url, max_size=None) as ws:
        # handshake
        first = json.loads(await asyncio.wait_for(ws.recv(), 15))
        print(f"[{time.perf_counter()-t_start:6.2f}s] {first.get('type')}")

        async def pump() -> None:
            nonlocal transcript_at
            while True:
                raw = await ws.recv()
                ev = json.loads(raw)
                events.append(ev)
                t = time.perf_counter() - t_start
                kind = ev.get("type")
                if kind == "transcript":
                    transcript_at = t
                    print(f"[{t:6.2f}s] TRANSCRIPT: {ev.get('text')!r}"
                          + (f" (revised_of={ev['revised_of'][:8]})"
                             if ev.get("revised_of") else ""))
                elif kind == "partial":
                    print(f"[{t:6.2f}s] PARTIAL: {ev.get('text')!r}")
                elif kind == "meta" and ev.get("verify"):
                    print(f"[{t:6.2f}s] VERIFY qid={str(ev.get('qid'))[:8]}: "
                          f"{ev['verify']}")
                elif kind == "meta" and ev.get("intent"):
                    print(f"[{t:6.2f}s] INTENT qid={str(ev.get('qid'))[:8]}: "
                          f"{ev['intent']!r}")
                elif kind == "meta" and ev.get("org_grounded"):
                    print(f"[{t:6.2f}s] meta: org_grounded=True")
                elif kind == "meta":
                    print(f"[{t:6.2f}s] meta: qtype={ev.get('qtype')} "
                          f"question={ev.get('question')!r}")
                elif kind == "token":
                    qid = ev.get("qid") or ""
                    answer_by_qid[qid] = answer_by_qid.get(qid, "") + ev.get("text", "")
                elif kind == "done":
                    print(f"[{t:6.2f}s] done qid={ev.get('qid')} "
                          f"skipped={ev.get('skipped')}")
                elif kind == "error":
                    print(f"[{t:6.2f}s] ERROR: {ev.get('detail')}")

        pump_task = asyncio.create_task(pump())

        # stream the audio in real-time-ish pacing (4x speed is fine: the
        # segmenter endpoints on in-audio silence, not wall clock)
        for frame in pcm_frames():
            await ws.send(frame)
            await asyncio.sleep(CHUNK_MS / 1000 / 4)
        await ws.send(json.dumps({"type": "flush"}))
        print(f"[{time.perf_counter()-t_start:6.2f}s] audio fully sent")

        # give detection + answering time to finish
        try:
            await asyncio.wait_for(asyncio.shield(pump_task), timeout=90)
        except asyncio.TimeoutError:
            pump_task.cancel()

    print("\n===== RESULT =====")
    print(f"transcript latency after audio end: "
          f"{transcript_at:.2f}s (stream point)" if transcript_at else "NO TRANSCRIPT")
    for qid, ans in answer_by_qid.items():
        print(f"\n--- answer (qid={qid or 'n/a'}) ---\n{ans[:600]}")
    kinds = [e.get("type") for e in events]
    print(f"\nevent kinds seen: {sorted(set(kinds))}")


if __name__ == "__main__":
    asyncio.run(main())
