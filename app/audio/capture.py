"""
Server-side audio capture (system loopback or mic).

Used when the Flutter client doesn't capture audio itself — e.g. headless
or kiosk mode. The default Phase-4 flow has the Flutter app stream PCM
over WebSocket, so this module is the fallback path.

Platform notes:
- Windows: pyaudiowpatch for WASAPI loopback (capture what the speakers
  play), or sounddevice for mic. pyaudiowpatch is a fork that ships
  WASAPI Loopback host APIs in PyAudio.
- macOS: sounddevice + BlackHole or other virtual cable for loopback.
- Linux: sounddevice with the PulseAudio "monitor" source.

This module exposes a `read_chunks(duration_ms)` async generator. The
caller is the WebSocket route, which forwards chunks through VAD->STT.
"""
from __future__ import annotations

import asyncio
import sys
from typing import AsyncGenerator

from app.core.config_loader import cfg


class CaptureError(RuntimeError):
    pass


async def read_chunks(
    duration_ms: int | None = None,
    source: str | None = None,
) -> AsyncGenerator["object", None]:  # numpy.ndarray at runtime
    """Yield `chunk_ms`-sized audio frames as numpy arrays until cancelled.

    The backend is chosen by `source` (falling back to `cfg.audio.source`):
      - "mic": sounddevice default input
      - "system_loopback": pyaudiowpatch loopback (Windows only here) — this
        captures whatever the speakers are playing, i.e. the OTHER party's
        voice in a Zoom/Teams/Meet call. Live Listen uses this so it
        transcribes the interviewer, not the candidate.
      - "both": tries loopback first, falls back to mic

    `source` lets the Live Listen WebSocket pick the device per session (the
    user toggles "Interviewer (system audio)" vs "My microphone" in the UI)
    without mutating global config. Use `asyncio.timeout(...)` upstream to cap
    the capture session.
    """
    source = source or cfg.audio.source
    if source == "system_loopback" and sys.platform != "win32":
        # On non-Windows the user needs to route via BlackHole/PulseAudio
        # to a virtual device and pick that with `source=mic`. Capturing
        # the OS mix directly is platform-dependent and out of scope here.
        raise CaptureError(
            "system_loopback capture from Python is currently only "
            "implemented on Windows (via pyaudiowpatch). On macOS/Linux, "
            "route system audio to a virtual input device (BlackHole / "
            "PulseAudio monitor) and use audio.source: mic."
        )

    if source in ("system_loopback", "both") and sys.platform == "win32":
        try:
            async for chunk in _windows_loopback_stream():
                yield chunk
            return
        except CaptureError:
            if source == "both":
                async for chunk in _mic_stream():
                    yield chunk
                return
            raise

    async for chunk in _mic_stream():
        yield chunk


async def _mic_stream() -> AsyncGenerator["object", None]:
    try:
        import numpy as np
        import sounddevice as sd
    except ImportError as exc:
        raise CaptureError(
            "sounddevice/numpy are not installed. Run: pip install sounddevice numpy"
        ) from exc

    sample_rate = cfg.audio.sample_rate
    chunk_samples = int(sample_rate * cfg.audio.chunk_ms / 1000)
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def cb(indata, frames, _t, _status):
        # sounddevice runs the callback on a worker thread; push samples
        # into the asyncio queue for the consumer.
        loop.call_soon_threadsafe(queue.put_nowait, indata.copy())

    with sd.InputStream(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=chunk_samples,
        callback=cb,
    ):
        while True:
            chunk = await queue.get()
            yield np.ascontiguousarray(chunk.reshape(-1))


async def _windows_loopback_stream() -> AsyncGenerator["object", None]:
    """WASAPI loopback capture on Windows via pyaudiowpatch."""
    try:
        import numpy as np
        import pyaudiowpatch as pyaudio
    except ImportError as exc:
        raise CaptureError(
            "pyaudiowpatch is not installed. Run: pip install PyAudioWPatch"
        ) from exc

    sample_rate = cfg.audio.sample_rate
    chunk_samples = int(sample_rate * cfg.audio.chunk_ms / 1000)
    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    p = pyaudio.PyAudio()
    try:
        # Find the default WASAPI loopback device — the one whose name
        # ends with "[Loopback]" matching the active output device.
        wasapi_info = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        default_speakers = p.get_device_info_by_index(wasapi_info["defaultOutputDevice"])
        if not default_speakers.get("isLoopbackDevice"):
            for i in range(p.get_device_count()):
                info = p.get_device_info_by_index(i)
                if (
                    info.get("isLoopbackDevice")
                    and default_speakers["name"] in info["name"]
                ):
                    default_speakers = info
                    break
            else:
                raise CaptureError(
                    "No WASAPI loopback device found matching the default speakers."
                )

        device_rate = int(default_speakers["defaultSampleRate"])

        def cb(in_data, _frame_count, _time_info, _status):
            arr = np.frombuffer(in_data, dtype=np.float32)
            # Mix down to mono if the device delivers stereo.
            ch = int(default_speakers["maxInputChannels"])
            if ch > 1:
                arr = arr.reshape(-1, ch).mean(axis=1)
            # Cheap linear resample to our target rate.
            if device_rate != sample_rate:
                ratio = sample_rate / device_rate
                idx = (np.arange(int(len(arr) * ratio)) / ratio).astype(int)
                arr = arr[np.clip(idx, 0, len(arr) - 1)]
            loop.call_soon_threadsafe(queue.put_nowait, np.ascontiguousarray(arr))
            return (None, pyaudio.paContinue)

        stream = p.open(
            format=pyaudio.paFloat32,
            channels=int(default_speakers["maxInputChannels"]),
            rate=device_rate,
            frames_per_buffer=chunk_samples,
            input=True,
            input_device_index=default_speakers["index"],
            stream_callback=cb,
        )
        try:
            while True:
                chunk = await queue.get()
                yield chunk
        finally:
            stream.stop_stream()
            stream.close()
    finally:
        p.terminate()
