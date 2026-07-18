"""RapidOCR text pass — exact on-screen text extraction."""
import asyncio
import base64
import io

import pytest

from app.vision import ocr


def _png(text_lines: list[str]) -> str:
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (480, 60 + 34 * len(text_lines)), (255, 255, 255))
    d = ImageDraw.Draw(img)
    y = 20
    for ln in text_lines:
        d.text((20, y), ln, fill=(0, 0, 0))
        y += 34
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_disabled_returns_empty(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.vision, "ocr_enabled", False, raising=False)
    assert asyncio.run(ocr.ocr_images([_png(["Hello"])])) == ""


def test_empty_input_returns_empty():
    assert asyncio.run(ocr.ocr_images([])) == ""
    assert asyncio.run(ocr.ocr_images([""])) == ""


@pytest.mark.skipif(not ocr.is_available(), reason="rapidocr not installed")
def test_reads_text(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.vision, "ocr_enabled", True, raising=False)
    out = asyncio.run(ocr.ocr_images([_png(["Java", "firstMissingPositive"])]))
    low = out.lower()
    # OCR is not pixel-perfect on synthetic bitmaps, but the language token and
    # the method name should both survive.
    assert "java" in low
    assert "missing" in low
