"""
NVIDIA Parakeet TDT local STT — the PRIMARY live transcriber.

Runs parakeet-tdt-0.6b-v3 via `onnx-asr` (onnxruntime): no torch/NeMo
needed, int8-quantized CPU inference is ~8x faster than real time, accuracy
sits at the top of the Open ASR leaderboard, and v3 is MULTILINGUAL (25
languages). The same warm model also produces the streaming partials, so
partial and final text always agree. Qwen3-ASR backs it up in the fallback
chain (model failed to load, OOM, mid-session crash, ...).

Interface contract (see app/stt/factory.py): top-level
`transcribe(audio_np) -> str` where `audio_np` is 16-kHz float32 mono.
"""
from __future__ import annotations

import logging
import threading

from app.core.config_loader import cfg

log = logging.getLogger(__name__)

_SAMPLE_RATE = 16_000

# Single-flight model load. lru_cache is NOT atomic: on the first utterance a
# partial and the final pass can both trigger a cold load concurrently and run
# `onnx_asr.load_model` TWICE (doubling the ~seconds load + CPU contention →
# the first transcript arrives very late, looking like "the mic didn't work").
# A lock + a plain cached slot makes the load happen exactly once.
_model_lock = threading.Lock()
_model_cache = None


class STTError(RuntimeError):
    """Raised when the Parakeet model cannot be loaded or used."""


def _cuda_ready() -> bool:
    """True when ORT has the CUDA provider AND its CUDA/cuDNN DLLs are locatable.

    The CUDA EP loads cublasLt / cudnn as TRANSITIVE dependencies; on Windows a
    dependency load is NOT covered by os.add_dll_directory, so those dirs must be
    on PATH before the EP DLL loads. We point ORT at the `nvidia-*-cu12` pip
    packages (which match the CUDA-12 onnxruntime-gpu build) — deliberately NOT
    torch/lib, whose bundled cuDNN is a different ABI and makes the EP silently
    fall back to CPU (observed: "cublasLt64_13.dll missing" / procedure-not-found
    on cudnn). Requires onnxruntime-gpu built for CUDA 12 (see requirements)."""
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            return False
        import os
        # site-packages/onnxruntime/__init__.py -> site-packages
        sp = os.path.dirname(os.path.dirname(ort.__file__))
        nvidia_dirs = [
            os.path.join(sp, "nvidia", pkg, "bin")
            for pkg in ("cublas", "cudnn", "cuda_runtime", "cuda_nvrtc")
        ]
        existing = [d for d in nvidia_dirs if os.path.isdir(d)]
        if existing:
            os.environ["PATH"] = (os.pathsep.join(existing) + os.pathsep
                                  + os.environ.get("PATH", ""))
            for d in existing:
                try:
                    os.add_dll_directory(d)
                except Exception:  # noqa: BLE001
                    pass
        if hasattr(ort, "preload_dlls"):
            import contextlib
            import io
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    ort.preload_dlls()
                except Exception:  # noqa: BLE001 — best-effort; PATH already set
                    pass
        return True
    except Exception:  # noqa: BLE001
        return False


def _model():
    """Lazy-load Parakeet once per process, single-flight.

    GPU-first: when `stt.parakeet_use_gpu` (default) and CUDA is available,
    load fp32 on the CUDAExecutionProvider — utterance transcription drops
    from ~seconds (int8 CPU) to tens of milliseconds, which is what makes
    sub-second live responses possible. Any GPU failure falls back to the
    int8 CPU model transparently."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    with _model_lock:
        if _model_cache is not None:  # another thread loaded it while we waited
            return _model_cache
        try:
            import onnx_asr
        except ImportError as exc:
            raise STTError(
                "onnx-asr is not installed. Run: pip install onnx-asr"
            ) from exc
        model_id = getattr(cfg.stt, "parakeet_model", "nemo-parakeet-tdt-0.6b-v2")
        quant = getattr(cfg.stt, "parakeet_quantization", "int8") or None
        if bool(getattr(cfg.stt, "parakeet_use_gpu", True)) and _cuda_ready():
            try:
                import onnxruntime as ort
                so = ort.SessionOptions()
                # Quiet the known-benign "N Memcpy nodes are added" warning:
                # a few ops in the TDT decoder graph run on CPU, so ORT adds
                # host<->device copies at session build. Our measured 48-125ms
                # per utterance already includes that cost. ERROR-only logging
                # for these sessions keeps startup logs clean.
                so.log_severity_level = 3
                log.info("loading Parakeet %s on CUDA (fp32)", model_id)
                _model_cache = onnx_asr.load_model(
                    model_id,
                    # fp32 on GPU: int8 is a CPU-quantization format; fp32 is
                    # both faster on CUDA and slightly more accurate.
                    quantization=None,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    sess_options=so,
                )
                return _model_cache
            except Exception as exc:  # noqa: BLE001 — GPU is opportunistic
                log.warning("Parakeet CUDA load failed (%s) — falling back "
                            "to int8 CPU.", exc)
        log.info("loading Parakeet %s (quantization=%s, CPU)", model_id, quant)
        _model_cache = onnx_asr.load_model(model_id, quantization=quant)
    return _model_cache


def transcribe(audio_np) -> str:
    """Transcribe a 16-kHz float32 mono numpy array. Returns plain text."""
    import numpy as np

    model = _model()
    # onnx-asr expects float32 in [-1, 1]; the segmenter already provides
    # that, but be defensive about dtype so a fallback call never dies on
    # an int16 buffer.
    audio = np.asarray(audio_np)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32) / (32768.0 if audio.dtype == np.int16 else 1.0)
    text = model.recognize(audio, sample_rate=_SAMPLE_RATE)
    return (text or "").strip()
