"""
STT factory: picks a transcriber from cfg.stt.provider.

Adding a provider: write a module with a top-level `transcribe(audio_np) -> str`
and register it in `_PROVIDERS` here.

When `cfg.stt.dual_engine_enabled` is True, the factory wraps the
primary in a [DualSTT] that fans out to a secondary engine in
parallel (Architecture.md §"Dual-STT redundancy"). The secondary is
chosen by `cfg.stt.secondary_provider`; if it isn't installed the
DualSTT silently degrades to single-engine mode.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Callable

from app.core.config_loader import cfg
from app.stt import parakeet_stt, qwen_asr_stt, whisper_stt


log = logging.getLogger(__name__)


class STTConfigError(RuntimeError):
    pass


_PROVIDERS: dict[str, Callable] = {
    "faster_whisper": whisper_stt.transcribe,
    # Local-only chain (no API STT): Qwen3-ASR primary (multilingual,
    # GPU-preferring), Parakeet TDT fallback (English, CPU-fast). Heavy
    # deps load lazily inside each module's _model().
    "qwen_asr": qwen_asr_stt.transcribe,
    "parakeet": parakeet_stt.transcribe,
}

# Async (cloud) STT providers — coroutine `transcribe(audio_np) -> str`.
# These run on the provider's GPUs (Groq Whisper etc.) and are awaited
# directly by `transcribe_async`. Aliases all map to the cloud chain, whose
# first entry (and fallbacks) are chosen by cfg.stt.cloud_chain.
def unload_all() -> None:
    """Free EVERY cached STT engine (2026-07-12: local-only, one model
    resident at a time — a KVM VPS can't afford three sets of weights). The
    active provider lazy-loads on the next utterance / warm_active()."""
    global _dual_singleton
    _dual_singleton = None
    try:
        from app.stt import parakeet_stt
        parakeet_stt._model_cache = None
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.stt import qwen_asr_stt
        qwen_asr_stt._model.cache_clear()
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.stt import whisper_stt
        whisper_stt._model.cache_clear()
    except Exception:  # noqa: BLE001
        pass
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def resident_engines() -> list[str]:
    """Which engines currently hold weights in memory — the honest "is the
    old model actually dead" metric the switch popup shows."""
    out: list[str] = []
    try:
        from app.stt import parakeet_stt
        if parakeet_stt._model_cache is not None:
            out.append("parakeet")
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.stt import qwen_asr_stt
        if qwen_asr_stt._model.cache_info().currsize > 0:
            out.append("qwen_asr")
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.stt import whisper_stt
        if whisper_stt._model.cache_info().currsize > 0:
            out.append("faster_whisper")
    except Exception:  # noqa: BLE001
        pass
    return out


async def warm_active() -> None:
    """Load the SELECTED engine now (a second of silence through the normal
    path) so the first real utterance isn't the one paying the cold load.
    Best-effort — failures surface on the real utterance instead."""
    try:
        import numpy as np
        silence = np.zeros(int(cfg.audio.sample_rate), dtype=np.float32)
        await transcribe_with_confidence(silence)
    except Exception as exc:  # noqa: BLE001
        log.info("stt warm skipped: %s", exc)


class _EngineAdapter:
    """Wraps a `transcribe(audio_np) -> str` callable to look like the
    object DualSTT expects (`.transcribe()` method + `name` attribute)."""

    def __init__(self, name: str, fn: Callable) -> None:
        self.name = name
        self._fn = fn

    def transcribe(self, audio_np) -> str:
        return self._fn(audio_np)


_dual_singleton = None


def _get_dual():
    """Lazy build + cache the DualSTT instance. Reset by `reset_for_tests`."""
    global _dual_singleton
    if _dual_singleton is not None:
        return _dual_singleton
    from .dual_engine import DualSTT

    primary_fn = _PROVIDERS.get(cfg.stt.provider)
    if primary_fn is None:
        return None
    primary = _EngineAdapter(cfg.stt.provider, primary_fn)
    secondary = None
    sec_name = getattr(cfg.stt, "secondary_provider", None)
    if sec_name and sec_name in _PROVIDERS:
        secondary = _EngineAdapter(sec_name, _PROVIDERS[sec_name])
    elif sec_name:
        log.info("dual STT: secondary '%s' not registered; single-engine mode", sec_name)
    d = DualSTT()
    d.configure(primary, secondary)
    _dual_singleton = d
    return d


def transcribe(audio_np) -> str:
    """Sync dispatch — primary engine only.

    Live callers from inside a coroutine should call
    [transcribe_async] instead so dual-engine mode actually runs.
    The sync path falls back to single-engine even when dual is
    enabled, because `asyncio.run` inside a running loop is a
    RuntimeError.
    """
    fn = _PROVIDERS.get(cfg.stt.provider)
    if fn is None:
        raise STTConfigError(
            f"STT provider '{cfg.stt.provider}' is not implemented. "
            f"Available: {list(_PROVIDERS)}"
        )
    return fn(audio_np)


async def transcribe_async(audio_np, prompt: str | None = None) -> str:
    """Async dispatch — honours cloud (async) and dual-engine modes.

    Local engines are offloaded via `asyncio.to_thread` so they never block
    the loop. This is what the WebSocket / live path calls. `prompt` is kept
    for interface stability (local engines ignore it).
    """
    text, _ = await transcribe_with_confidence(audio_np, prompt)
    return text


def _provider_chain() -> list[str]:
    """Primary provider followed by `cfg.stt.fallback_providers` (deduped).

    The chain is what makes a fully local deployment resilient: if the
    primary engine can't load or dies mid-session (OOM, missing dep, model
    download failure), the next provider transcribes the same utterance and
    the live session never notices.
    """
    chain = [cfg.stt.provider]
    for name in getattr(cfg.stt, "fallback_providers", None) or []:
        if name and name not in chain:
            chain.append(name)
    return chain


async def transcribe_with_confidence(
    audio_np, prompt: str | None = None,
) -> tuple[str, float | None]:
    """Like `transcribe_async`, but also returns the engine's confidence when
    one exists (the dual-engine arbitrator produces a real score; single
    engines return None). Lets downstream answer confidence reflect a
    poorly-heard utterance instead of trusting every transcript equally.

    Providers are tried in `_provider_chain` order; each failure logs and
    falls through to the next. Only the final failure propagates.
    """
    # CLOUD mode (Settings toggle): transcribe via Groq Whisper. Falls back to
    # the local chain below on any failure (no key, network, etc.).
    if str(getattr(cfg.stt, "mode", "local") or "local").lower() == "cloud":
        try:
            from app.stt import cloud_stt
            text = await cloud_stt.transcribe(audio_np, prompt)
            if text.strip():
                return text, None
            log.info("cloud STT returned empty — falling back to local")
        except Exception as exc:  # noqa: BLE001
            log.warning("cloud STT failed (%s); falling back to local", exc)

    if getattr(cfg.stt, "dual_engine_enabled", False):
        dual = _get_dual()
        if dual is not None:
            try:
                result = await dual.transcribe(audio_np)
                return result.text, getattr(result, "confidence", None)
            except Exception as exc:  # noqa: BLE001
                log.warning("dual STT failed (%s); falling back to single engine", exc)

    chain = _provider_chain()
    last_exc: Exception | None = None
    for i, name in enumerate(chain):
        is_last = i == len(chain) - 1
        try:
            fn = _PROVIDERS.get(name)
            if fn is None:
                raise STTConfigError(
                    f"STT provider '{name}' is not implemented. "
                    f"Available: {list(_PROVIDERS)}"
                )
            # Offload sync engines so we never block the event loop.
            return await asyncio.to_thread(fn, audio_np), None
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            log.warning(
                "STT provider '%s' failed: %s%s",
                name, exc, "" if is_last else " — trying next in chain",
            )
    assert last_exc is not None
    raise last_exc


async def transcribe_partial(audio_np) -> str:
    """Interim (streaming) transcription of an IN-PROGRESS utterance.

    Uses the fast provider configured as `cfg.stt.partial_provider` (default
    Parakeet: ~8x realtime on CPU) so partials keep up with speech while the
    accurate primary chain handles the final pass. Returns "" when partials
    are unconfigured or the provider isn't registered — callers treat that
    as "no partial", never an error.
    """
    name = getattr(cfg.stt, "partial_provider", "") or ""
    fn = _PROVIDERS.get(name)
    if fn is None:
        return ""
    return await asyncio.to_thread(fn, audio_np)


def reset_for_tests() -> None:
    global _dual_singleton
    _dual_singleton = None


# ── GPU STT device selection (live-conversational-intelligence R60) ────────
# Latency-only: when `live.gpu_stt` is enabled AND a CUDA device is available,
# resolve the faster-whisper device to "cuda" (float16); otherwise fall back
# cleanly to the configured CPU device. Transcription SEMANTICS are unchanged —
# only where the model runs. Never raises.

def _cuda_available() -> bool:
    try:
        import ctranslate2  # type: ignore
        count = ctranslate2.get_cuda_device_count()
        return bool(count and count > 0)
    except Exception:  # noqa: BLE001
        try:
            import torch  # type: ignore
            return bool(torch.cuda.is_available())
        except Exception:  # noqa: BLE001
            return False


def resolve_device() -> tuple[str, str]:
    """Return (device, compute_type) for the STT model. GPU only when
    `live.gpu_stt` is on AND CUDA is present; else the configured CPU settings.
    Latency-only — semantics unchanged. Never raises."""
    try:
        gpu_on = bool(getattr(cfg.live, "gpu_stt", False))
        if gpu_on and _cuda_available():
            ct = getattr(cfg.stt, "compute_type", "int8")
            # int8 is a CPU compute type; prefer float16 on GPU.
            if ct in ("int8", "int8_float16"):
                ct = "float16"
            return "cuda", ct
        return getattr(cfg.stt, "device", "cpu"), getattr(cfg.stt, "compute_type", "int8")
    except Exception:  # noqa: BLE001
        return "cpu", "int8"
