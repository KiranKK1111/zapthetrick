"""GPU-local Kokoro TTS seam. The pod build installs the runtime; this dev box
does not — so these assert the FAIL-OPEN guarantee that makes shipping it safe:
without the runtime the engine is inert and synthesis degrades to Edge exactly
as before, and a broken/partial runtime can never crash a turn."""
import asyncio

from app.live import kokoro_engine as K
from app.live import tts_synth


def test_available_is_honest():
    # Never raises; simply reports whether the runtime is importable.
    assert isinstance(K.available(), bool)


def test_synth_returns_empty_without_runtime():
    # b"" is the contract that triggers the Edge fallback.
    if K.available():
        return  # runtime present (pod) — nothing to assert here
    assert K.synth("hello there", "af_heart", 1.0) == b""


def test_register_is_noop_without_runtime():
    if K.available():
        return
    assert K.register_if_available() is False


def test_empty_text_is_empty():
    assert K.synth("", "af_heart", 1.0) == b""
    assert K.synth("   ", "af_heart", 1.0) == b""


def test_synthesize_falls_back_to_edge_when_kokoro_yields_nothing(monkeypatch):
    """The whole safety story: engine=kokoro + a dead kokoro must still speak."""
    monkeypatch.setattr(tts_synth, "_engine", lambda: "kokoro")
    tts_synth.set_kokoro(lambda text, voice, speed: b"")   # dead engine
    called = {"edge": False}

    async def _fake_edge(text, voice_id, *, speed):
        called["edge"] = True
        return b"MP3BYTES"

    monkeypatch.setattr(tts_synth, "_synth_edge", _fake_edge)
    out = asyncio.run(tts_synth.synthesize("hello", speed=1.0))
    assert out == b"MP3BYTES" and called["edge"] is True
    tts_synth.set_kokoro(None)


def test_synthesize_prefers_kokoro_when_it_produces_audio(monkeypatch):
    monkeypatch.setattr(tts_synth, "_engine", lambda: "kokoro")
    tts_synth.set_kokoro(lambda text, voice, speed: b"KOKOROMP3")

    async def _fake_edge(text, voice_id, *, speed):
        raise AssertionError("Edge must not be called when Kokoro produced audio")

    monkeypatch.setattr(tts_synth, "_synth_edge", _fake_edge)
    assert asyncio.run(tts_synth.synthesize("hello", speed=1.0)) == b"KOKOROMP3"
    tts_synth.set_kokoro(None)


def test_a_raising_kokoro_still_degrades(monkeypatch):
    monkeypatch.setattr(tts_synth, "_engine", lambda: "kokoro")

    def _boom(text, voice, speed):
        raise RuntimeError("cuda oom")

    tts_synth.set_kokoro(_boom)

    async def _fake_edge(text, voice_id, *, speed):
        return b"EDGE"

    monkeypatch.setattr(tts_synth, "_synth_edge", _fake_edge)
    assert asyncio.run(tts_synth.synthesize("hello", speed=1.0)) == b"EDGE"
    tts_synth.set_kokoro(None)


def test_non_english_text_falls_back_to_edge():
    """Kokoro's pipeline is English G2P; Hindi/Telugu replies must return b""
    so tts_synth degrades to Edge's LANGUAGE-MATCHED voices — otherwise the
    multilingual voice feature comes out mangled."""
    assert K.synth("काफ्का कैसे काम करता है", "af_heart", 1.0) == b""
    assert K.synth("కాఫ్కా అంటే ఏమిటి", "af_heart", 1.0) == b""
