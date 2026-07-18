"""MiniCPM-V local vision engine (VisionAnalysis.md #5 — the fast fallback).

One of the fastest local VLMs; the lightweight fallback for low-VRAM /
speed-first. Uses MiniCPM-V's custom `.chat()` API (trust_remote_code). GPU-
first with CPU fallback; single-flight load; fail-open (raises on failure so the
factory falls through).
"""
from __future__ import annotations

import io
import logging
import threading
from typing import Sequence

log = logging.getLogger(__name__)

# (model, tokenizer, device) once loaded.
_cache: tuple[object, object, str] | None = None
_lock = threading.Lock()


def _cfg():
    from app.core.config_loader import cfg
    return cfg.vision


def _load() -> tuple[object, object, str]:
    global _cache
    if _cache is not None:
        return _cache
    with _lock:
        if _cache is not None:
            return _cache
        import torch  # noqa: PLC0415
        from transformers import AutoModel, AutoTokenizer  # noqa: PLC0415

        from ._hf import load_local_first  # noqa: PLC0415

        model_id = _cfg().minicpm_model
        use_gpu = bool(_cfg().use_gpu) and torch.cuda.is_available()
        dtype = (torch.bfloat16 if (use_gpu and torch.cuda.is_bf16_supported())
                 else (torch.float16 if use_gpu else torch.float32))
        # local-first: cached model loads with no network HEAD check.
        model = load_local_first(
            AutoModel.from_pretrained, model_id,
            trust_remote_code=True, torch_dtype=dtype)
        device = "cuda" if use_gpu else "cpu"
        model = model.eval().to(device)
        tokenizer = load_local_first(
            AutoTokenizer.from_pretrained, model_id, trust_remote_code=True)
        _cache = (model, tokenizer, device)
        log.info("minicpm-v loaded: %s on %s", model_id, device)
        return _cache


def _pil_images(images: Sequence[bytes]) -> list:
    from PIL import Image  # noqa: PLC0415
    out = []
    for raw in images:
        try:
            out.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            log.info("minicpm-v: skipping undecodable image (%s)", exc)
    return out


def describe(images: Sequence[bytes], prompt: str) -> str:
    imgs = _pil_images(images)
    if not imgs:
        return ""
    model, tokenizer, _ = _load()
    # MiniCPM-V takes images + text interleaved in the message content.
    msgs = [{"role": "user", "content": [*imgs, prompt]}]
    res = model.chat(
        image=None, msgs=msgs, tokenizer=tokenizer,
        sampling=False, max_new_tokens=int(_cfg().max_new_tokens))
    return (res or "").strip() if isinstance(res, str) else str(res).strip()


def unload() -> None:
    global _cache
    with _lock:
        _cache = None


def is_loaded() -> bool:
    return _cache is not None
