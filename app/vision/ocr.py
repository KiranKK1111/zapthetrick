"""RapidOCR text pass — exact on-screen text extraction (VisionAnalysis.md).

A small local VLM understands an image but READS dense text unreliably: on a
LeetCode/IDE screenshot it transcribes the problem yet skips the code panel, so
the selected language and the code stub are lost. OCR is the right tool for
exact text — it reads EVERY text region regardless of layout, fast and locally.

RapidOCR runs on the onnxruntime we already ship for STT (no new native binary,
no cloud). It COMPLEMENTS the VLM: the VLM gives structure/understanding, OCR
gives the exact characters (the "Java" chip, the `public int f(int[])` stub).

Single-flight engine load; everything fail-open (any error → "" so the caller
keeps the VLM-only text). Runs on a worker thread — never blocks the loop.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import io
import logging
import threading
from typing import Sequence

log = logging.getLogger(__name__)

_engine = None
_lock = threading.Lock()


def _cfg():
    from app.core.config_loader import cfg
    return cfg.vision


def _load():
    """Load the RapidOCR engine once (models are tiny ONNX, cached by the lib)."""
    global _engine
    if _engine is not None:
        return _engine
    with _lock:
        if _engine is not None:
            return _engine
        from rapidocr_onnxruntime import RapidOCR  # noqa: PLC0415
        _engine = RapidOCR()
        log.info("rapidocr engine loaded")
        return _engine


def _decode(images_b64: Sequence[str]) -> list[bytes]:
    out: list[bytes] = []
    for s in images_b64:
        if not s:
            continue
        raw = s.split(",", 1)[1] if s.startswith("data:") else s
        try:
            out.append(base64.b64decode(raw))
        except (binascii.Error, ValueError):
            continue
    return out


def _prep_for_ocr(img):
    """Size the image so RapidOCR reliably reads SMALL text (a tiny editor
    "<Lang> Auto" language chip) without paying for wasted 4K pixels. Small
    captures are UPSCALED (LANCZOS, capped ~2x) so the chip becomes legible;
    oversized captures are DOWNSCALED so CPU cost is bounded. The FE already
    downscales screenshots before upload, so the chip usually arrives small —
    this is the single highest-leverage fix for chip mis-reads."""
    from PIL import Image  # noqa: PLC0415
    try:
        c = _cfg()
        min_side = int(getattr(c, "ocr_min_side", 1600) or 1600)
        max_side = int(getattr(c, "ocr_max_side", 2600) or 2600)
    except Exception:  # noqa: BLE001
        min_side, max_side = 1600, 2600
    w, h = img.size
    longest = max(w, h)
    if longest <= 0:
        return img
    if longest < min_side:
        scale = min(2.0, min_side / float(longest))
    elif longest > max_side:
        scale = max_side / float(longest)
    else:
        return img
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                      Image.LANCZOS)


def _ocr_bytes(images: list[bytes]) -> str:
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    eng = _load()
    parts: list[str] = []
    for raw in images:
        try:
            img = _prep_for_ocr(
                Image.open(io.BytesIO(raw)).convert("RGB"))
            arr = np.array(img)
            result, _elapse = eng(arr)
            if result:
                # result rows are [box, text, score] — keep reading order.
                parts.append("\n".join(r[1] for r in result if len(r) > 1))
        except Exception as exc:  # noqa: BLE001 — skip a bad image, keep the rest
            log.info("rapidocr: skipping an image (%s)", exc)
    return "\n".join(p for p in parts if p).strip()


async def ocr_images(images_b64: Sequence[str]) -> str:
    """Exact on-screen text for image(s), via RapidOCR. Base64 in, text out.
    Returns "" when OCR is disabled, unavailable, or finds nothing. Never raises."""
    if not getattr(_cfg(), "ocr_enabled", False):
        return ""
    imgs = _decode(images_b64)
    if not imgs:
        return ""
    try:
        text = await asyncio.to_thread(_ocr_bytes, imgs)
    except Exception as exc:  # noqa: BLE001 — OCR is best-effort, never fatal
        log.info("rapidocr pass failed (%s)", exc)
        return ""
    if text:
        log.info("rapidocr read %d chars", len(text))
    return text


def is_available() -> bool:
    try:
        import rapidocr_onnxruntime  # noqa: F401,PLC0415
        return True
    except Exception:  # noqa: BLE001
        return False
