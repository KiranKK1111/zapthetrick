"""Manual test: local STT chain (Qwen3-ASR primary -> Parakeet fallback).

Run from zapthetrick_be with the venv python:
    .venv/Scripts/python test_local_stt.py

Exercises the REAL factory path the live WebSocket uses
(transcribe_with_confidence), then simulates a primary-engine crash to
prove the fallback engages. Not a pytest file on purpose — it downloads
models and needs the local machine.
"""
from __future__ import annotations

import asyncio
import time
import wave

import numpy as np

WAV = (
    r"C:\Users\kiran\AppData\Local\Temp\claude\d--DTT-Backup"
    r"\7e4a13aa-8c39-4a98-aaf4-ed245da075e5\scratchpad\test_q1.wav"
)


def load_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        assert w.getframerate() == 16000, w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32768.0


async def main() -> None:
    audio = load_wav(WAV)
    print(f"audio: {len(audio)/16000:.1f}s @16kHz")

    from app.core.config_loader import cfg
    from app.stt import factory, parakeet_stt, qwen_asr_stt

    print(f"provider={cfg.stt.provider} fallback={cfg.stt.fallback_providers}")

    # 1) Each provider standalone (also warms both models).
    for name, mod in (("qwen_asr", qwen_asr_stt), ("parakeet", parakeet_stt)):
        t0 = time.perf_counter()
        text = await asyncio.to_thread(mod.transcribe, audio)
        dt = time.perf_counter() - t0
        print(f"[{name}] {dt:.2f}s (incl. first-run model load): {text!r}")

    # 2) The real live path (factory chain, models now warm).
    t0 = time.perf_counter()
    text, conf = await factory.transcribe_with_confidence(audio)
    dt = time.perf_counter() - t0
    print(f"[chain/warm] {dt:.2f}s conf={conf}: {text!r}")

    # 3) Fallback: make the primary blow up like a mid-session OOM would.
    def boom(_audio):
        raise RuntimeError("simulated Qwen crash (OOM)")

    factory._PROVIDERS["qwen_asr"] = boom
    t0 = time.perf_counter()
    text, _ = await factory.transcribe_with_confidence(audio)
    dt = time.perf_counter() - t0
    print(f"[fallback->parakeet] {dt:.2f}s: {text!r}")
    factory._PROVIDERS["qwen_asr"] = qwen_asr_stt.transcribe


if __name__ == "__main__":
    asyncio.run(main())
