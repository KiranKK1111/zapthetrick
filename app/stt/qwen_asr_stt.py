"""
Qwen3-ASR local STT — the FALLBACK live transcriber.

Runs Alibaba's Qwen3-ASR (default 1.7B) fully locally via the `qwen-asr`
package (transformers backend). Multilingual: 30 languages + dialects with
automatic language detection (`language=None`). Backs up the (also
multilingual, much faster) Parakeet v3 primary in the provider chain.

Device: CUDA when available (bfloat16), else CPU (float32). On CPU a 1.7B
autoregressive ASR model is SLOW (seconds per utterance) — fine for a
correctness fallback, not for live latency. The GPU path is the intended
deployment.

Interface contract (see app/stt/factory.py): top-level
`transcribe(audio_np) -> str` where `audio_np` is 16-kHz float32 mono.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from app.core.config_loader import cfg

log = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000


class STTError(RuntimeError):
    """Raised when the Qwen ASR model cannot be loaded or used."""


@lru_cache(maxsize=1)
def _model():
    """Lazy-load Qwen3-ASR once per process (~3.4 GB VRAM at bf16)."""
    try:
        import torch
        from qwen_asr import Qwen3ASRModel
    except ImportError as exc:
        # Surface the REAL missing module — qwen_asr IS bundled, but its
        # sprawling dep chain (accelerate/soundfile/tiktoken/dynet/…) may not
        # be fully traced in a frozen build. The old generic "pip install
        # qwen-asr" was misleading. Qwen is a FALLBACK; the primary (Parakeet)
        # is unaffected.
        raise STTError(
            f"Qwen3-ASR runtime incomplete ({exc}). Optional fallback — "
            "Parakeet handles transcription."
        ) from exc
    use_cuda = torch.cuda.is_available()
    device = "cuda:0" if use_cuda else "cpu"
    dtype = torch.bfloat16 if use_cuda else torch.float32
    model_id = getattr(cfg.stt, "qwen_model", "Qwen/Qwen3-ASR-1.7B")
    log.info("loading Qwen ASR %s on %s (%s)", model_id, device, dtype)
    return Qwen3ASRModel.from_pretrained(
        model_id,
        dtype=dtype,
        device_map=device,
        # Live utterances arrive one at a time; a big batch cap only
        # reserves memory we never use.
        max_inference_batch_size=1,
        # Interview utterances are short (<=15s by segmenter cap) — 256
        # tokens is ample and caps worst-case decode latency.
        max_new_tokens=256,
    )


def _extract_text(results) -> str:
    """Normalize qwen-asr's return shape (list of result objects / dicts /
    plain strings across versions) to plain text."""
    if results is None:
        return ""
    items = results if isinstance(results, (list, tuple)) else [results]
    parts: list[str] = []
    for r in items:
        text = getattr(r, "text", None)
        if text is None and isinstance(r, dict):
            text = r.get("text")
        if text is None and isinstance(r, str):
            text = r
        if text:
            parts.append(str(text).strip())
    return " ".join(p for p in parts if p)


def transcribe(audio_np) -> str:
    """Transcribe a 16-kHz float32 mono numpy array. Returns plain text.

    `qwen_language` = null in config means autodetect — the point of having
    the multilingual model first in the chain. Set it (e.g. "en") to pin.
    """
    model = _model()
    language = getattr(cfg.stt, "qwen_language", None) or None
    results = model.transcribe(audio=(audio_np, _SAMPLE_RATE), language=language)
    return _extract_text(results)
