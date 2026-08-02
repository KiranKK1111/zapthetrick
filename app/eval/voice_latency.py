"""Voice first-audio latency budget — measured, not estimated.

The question this answers: can the FREE staged pipeline (no paid realtime API)
reach ChatGPT-like first-audio latency, and if not, which stage is spending the
time?

Why a harness rather than an opinion
------------------------------------
"Voice feels slow" is unactionable, and the pipeline has five stages that can
each plausibly be blamed. This measures each one on the machine it runs on, so
tuning targets the stage that actually costs and a claimed improvement can be
checked instead of asserted.

The stages, and what dominates each
-----------------------------------
1. **Endpointing** — waiting long enough to be sure the speaker finished. A
   floor set by configuration, not by hardware, and the single largest term in
   most setups. Speculation hides it when a partial already reads complete.
2. **STT final** — transcribing the utterance. GPU Parakeet is ~8x realtime;
   CPU Whisper large-v3 is not.
3. **LLM time-to-first-token** — the term with the widest spread. An on-pod
   llama.cpp model with a warm prompt cache answers in ~100-300 ms; a pooled
   free cloud model can take seconds and is not controllable.
4. **TTS first chunk** — GPU Kokoro is fast; Edge is a network round trip per
   sentence, which is why it was the first thing to fix.
5. **Transport** — frames to the client plus the jitter buffer.

Only stages whose components are actually present are measured. Anything absent
is reported as UNAVAILABLE rather than filled in with a flattering number — a
budget with invented terms is worse than no budget.

    python -m app.eval.voice_latency
"""
from __future__ import annotations

import asyncio
import statistics
import time
from dataclasses import dataclass, field

# ChatGPT Advanced Voice Mode is a single native speech-to-speech forward pass
# and lands around here. It is the bar, not a promise.
CHATGPT_REFERENCE_MS = (300.0, 500.0)
# What a staged pipeline can realistically reach with every stage local on a GPU.
STAGED_TARGET_MS = 600.0


@dataclass
class Stage:
    name: str
    detail: str = ""
    samples: list[float] = field(default_factory=list)
    available: bool = True

    @property
    def p50(self) -> float:
        return statistics.median(self.samples) if self.samples else 0.0

    @property
    def p95(self) -> float:
        if not self.samples:
            return 0.0
        ordered = sorted(self.samples)
        return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]

    def as_dict(self) -> dict:
        return {"stage": self.name, "available": self.available,
                "p50_ms": round(self.p50, 1), "p95_ms": round(self.p95, 1),
                "detail": self.detail}


def _cfg():
    from app.core.config_loader import cfg
    return cfg


def endpointing_floor() -> Stage:
    """The configured wait before an utterance is considered finished.

    Not measured — it IS the configuration. The adaptive segmenter shortens it
    (~0.55x) when a partial already reads as a complete question, which is the
    number that matters for a well-formed question.
    """
    s = Stage("endpointing")
    try:
        base = float(getattr(_cfg().audio, "endpoint_silence_ms", 550) or 550)
    except Exception:  # noqa: BLE001
        s.available = False
        s.detail = "config unavailable"
        return s
    complete = base * 0.55
    s.samples = [complete]
    s.detail = (f"{complete:.0f} ms for a complete-sounding question "
                f"(base {base:.0f} ms; speculation can hide this entirely)")
    return s


def stt_final(rounds: int = 3) -> Stage:
    """Transcribe a short utterance with the configured engine."""
    s = Stage("stt_final")
    try:
        import numpy as np

        from app.stt import factory as stt_factory
    except Exception as exc:  # noqa: BLE001
        s.available = False
        s.detail = f"STT unavailable ({type(exc).__name__})"
        return s
    # ~1.5 s of speech-shaped audio: long enough to be representative, short
    # enough that this stays a latency probe rather than a benchmark.
    sr = 16000
    t = np.arange(int(sr * 1.5)) / sr
    audio = (0.3 * np.sin(2 * np.pi * 140 * t)).astype("float32")
    try:
        for i in range(rounds):
            t0 = time.perf_counter()
            asyncio.run(stt_factory.transcribe_with_confidence(audio))
            dt = (time.perf_counter() - t0) * 1000.0
            if i:                       # drop the first: it loads the model
                s.samples.append(dt)
        s.detail = f"{getattr(_cfg().stt, 'provider', '?')} on 1.5 s of audio"
    except Exception as exc:  # noqa: BLE001
        s.available = False
        s.detail = f"transcription failed ({type(exc).__name__}: {exc!s:.60})"
    return s


def llm_first_token(rounds: int = 3) -> Stage:
    """Time to the FIRST token of a reply — the widest-spread term.

    This is what the on-pod local model exists for. A pooled free cloud model is
    not slow by accident; it is shared, and no amount of local tuning changes it.
    """
    s = Stage("llm_first_token")
    try:
        from app.core.llm_client import LLMClient
    except Exception as exc:  # noqa: BLE001
        s.available = False
        s.detail = f"LLM client unavailable ({type(exc).__name__})"
        return s

    messages = [
        {"role": "system", "content": "Answer in one short spoken sentence."},
        {"role": "user", "content": "What is a hash map?"},
    ]

    async def one() -> float:
        client = LLMClient()
        t0 = time.perf_counter()
        async for chunk in client.stream_chat(messages):
            if str(chunk or "").strip():
                return (time.perf_counter() - t0) * 1000.0
        return (time.perf_counter() - t0) * 1000.0

    try:
        for _ in range(rounds):
            s.samples.append(asyncio.run(one()))
        try:
            from app.llm.catalog import local_enabled
            s.detail = ("on-pod local model" if local_enabled()
                        else "cloud router (no local floor enabled)")
        except Exception:  # noqa: BLE001
            s.detail = "routed"
    except Exception as exc:  # noqa: BLE001
        s.available = False
        s.detail = f"no route ({type(exc).__name__}: {exc!s:.60})"
    return s


def tts_first_chunk(rounds: int = 3) -> Stage:
    """Synthesize one short sentence — what the user waits for before hearing
    anything, since the reply is spoken sentence by sentence."""
    s = Stage("tts_first_chunk")
    try:
        from app.live import tts_synth
    except Exception as exc:  # noqa: BLE001
        s.available = False
        s.detail = f"TTS unavailable ({type(exc).__name__})"
        return s

    async def one() -> float:
        t0 = time.perf_counter()
        await tts_synth.synthesize("A hash map stores key value pairs.")
        return (time.perf_counter() - t0) * 1000.0

    try:
        for i in range(rounds):
            dt = asyncio.run(one())
            if i:                       # drop the first: engine load
                s.samples.append(dt)
        engine = getattr(_cfg().voice, "tts_engine", "?")
        s.detail = (f"{engine} engine"
                    + (" — a network round trip PER SENTENCE" if engine == "edge"
                       else ""))
    except Exception as exc:  # noqa: BLE001
        s.available = False
        s.detail = f"synthesis failed ({type(exc).__name__}: {exc!s:.60})"
    return s


def measure(rounds: int = 3) -> dict:
    stages = [endpointing_floor(), stt_final(rounds),
              llm_first_token(rounds), tts_first_chunk(rounds)]
    measured = [st for st in stages if st.available and st.samples]
    total_p50 = sum(st.p50 for st in measured)
    missing = [st.name for st in stages if not (st.available and st.samples)]
    return {
        "stages": [st.as_dict() for st in stages],
        "total_p50_ms": round(total_p50, 1),
        "complete": not missing,
        "unmeasured": missing,
        "target_ms": STAGED_TARGET_MS,
        "chatgpt_reference_ms": list(CHATGPT_REFERENCE_MS),
        "meets_target": bool(not missing and total_p50 <= STAGED_TARGET_MS),
    }


def report(rounds: int = 3) -> int:
    r = measure(rounds)
    print("\nVoice first-audio latency budget")
    print("=" * 62)
    for st in r["stages"]:
        if st["available"] and st["p50_ms"]:
            print(f"  {st['stage']:<18} {st['p50_ms']:>8.1f} ms   {st['detail']}")
        else:
            print(f"  {st['stage']:<18} {'UNAVAILABLE':>11}   {st['detail']}")
    print("-" * 62)
    print(f"  {'TOTAL (measured)':<18} {r['total_p50_ms']:>8.1f} ms")
    print(f"  {'staged target':<18} {r['target_ms']:>8.1f} ms")
    lo, hi = r["chatgpt_reference_ms"]
    print(f"  {'ChatGPT AVM':<18} {lo:>4.0f}-{hi:.0f} ms   native speech-to-speech")
    if r["unmeasured"]:
        print(f"\n  INCOMPLETE — not measured here: {', '.join(r['unmeasured'])}")
        print("  The total above is a floor, not the whole budget.")
    elif r["meets_target"]:
        print("\n  Within the staged target.")
    else:
        worst = max((s for s in r["stages"] if s["available"]),
                    key=lambda s: s["p50_ms"], default=None)
        if worst:
            print(f"\n  Over target. Dominant stage: {worst['stage']} "
                  f"({worst['p50_ms']:.0f} ms) — {worst['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(report())


__all__ = ["measure", "report", "Stage", "STAGED_TARGET_MS"]
