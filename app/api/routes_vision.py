"""Local Vision Intelligence Layer — model selection + one-shot analysis.

LOCAL ONLY (VisionAnalysis.md): a local vision model reads every image to
structured TEXT; provider models never receive raw images. One engine resident
at a time, selected like STT.

GET  /api/vision/models   — selectable local vision engines + downloaded state.
POST /api/vision/select   — persist the choice + start the tracked switch.
GET  /api/vision/status   — live switch state for the popup.
POST /api/vision/analyze  — one-shot: image bytes in, structured text out.
"""
from __future__ import annotations

import base64
import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.config_loader import cfg

log = logging.getLogger("zapthetrick.vision")
router = APIRouter(prefix="/api/vision", tags=["vision"])


def _downloaded(repo: str | None) -> bool:
    if not repo:
        return True
    try:
        from app.models_warmup import hf_repo_cached
        return bool(hf_repo_cached(repo))
    except Exception:  # noqa: BLE001
        return False


def _active_id() -> str:
    return str(getattr(cfg.vision, "provider", "qwen2_5_vl"))


@router.get("/models")
async def vision_models() -> dict:
    """The selectable LOCAL vision engines, labeled for the dropdown."""
    smol_s = str(getattr(cfg.vision, "smolvlm_small_model",
                         "HuggingFaceTB/SmolVLM-500M-Instruct"))
    smol = str(getattr(cfg.vision, "smolvlm_model", "HuggingFaceTB/SmolVLM-Instruct"))
    qwen = str(getattr(cfg.vision, "qwen_vl_model", "Qwen/Qwen2.5-VL-3B-Instruct"))
    mcpm = str(getattr(cfg.vision, "minicpm_model", "openbmb/MiniCPM-V-2_6"))
    models: list[dict] = [
        {
            "id": "smolvlm_500m",
            "label": "SmolVLM 500M — fast & recommended",
            "detail": f"{smol_s} running on this machine (GPU when available). "
                      "Tiny and fast (~1 GB) yet reads UI, documents, tables and "
                      "charts well — fits alongside the speech model, the "
                      "recommended default.",
            "kind": "local",
            "downloaded": _downloaded(smol_s),
        },
        {
            "id": "smolvlm",
            "label": "SmolVLM 2.2B — more accurate (needs ~7 GB free VRAM)",
            "detail": f"{smol} running on this machine. Noticeably more accurate "
                      "than 500M, but the 2.2B weights need a bigger GPU; on a "
                      "small machine it's refused automatically and the 500M "
                      "model is used instead.",
            "kind": "local",
            "downloaded": _downloaded(smol),
        },
        {
            "id": "qwen2_5_vl",
            "label": "Qwen2.5-VL 3B — highest accuracy (needs ~10 GB VRAM)",
            "detail": f"{qwen} running on this machine. The most accurate reader, "
                      "but the 3B weights need a large GPU; refused automatically "
                      "when they won't fit.",
            "kind": "local",
            "downloaded": _downloaded(qwen),
        },
        {
            "id": "minicpm_v",
            "label": "MiniCPM-V 8B — max accuracy (needs a large GPU)",
            "detail": f"{mcpm} running on this machine. Highest ceiling but 8B "
                      "needs plenty of VRAM; refused automatically when it "
                      "won't fit.",
            "kind": "local",
            "downloaded": _downloaded(mcpm),
        },
    ]
    active = _active_id()
    for m in models:
        m["active"] = m["id"] == active
    return {"models": models, "active": active, "enabled": bool(
        getattr(cfg.vision, "enabled", True))}


class SelectBody(BaseModel):
    id: str


def _valid_ids() -> set[str]:
    try:
        from app.vision import factory
        return set(factory.available_providers())
    except Exception:  # noqa: BLE001
        return {"qwen2_5_vl", "minicpm_v"}


@router.post("/select")
async def select_model(body: SelectBody) -> dict:
    """EXCLUSIVE engine selection: persist the choice (one engine, no fallbacks)
    and start the TRACKED switch the popup observes via GET /api/vision/status."""
    target = (body.id or "").strip()
    if target not in _valid_ids():
        raise HTTPException(400, detail=f"Unknown vision model '{target}'.")
    partial = {
        "provider": target,
        "fallback_providers": [],
        "enabled": True,
    }
    from app.api.routes_settings import write_settings
    await write_settings({"vision": partial})
    from app.vision import switch as _switch
    await _switch.start_switch(target)
    return {"ok": True, "switching_to": target}


class ModeBody(BaseModel):
    mode: str


@router.post("/mode")
async def set_mode(body: ModeBody) -> dict:
    """Switch vision between LOCAL (on-device VLM, dropdown-selected) and CLOUD
    (image sent to a vision provider LLM for extraction; still stored in
    Postgres, never sent to the answer model)."""
    mode = (body.mode or "").strip().lower()
    if mode not in ("local", "cloud"):
        raise HTTPException(400, detail="mode must be 'local' or 'cloud'")
    from app.api.routes_settings import write_settings
    await write_settings({"vision": {"mode": mode}})
    return {"ok": True, "mode": mode}


@router.get("/status")
async def switch_status() -> dict:
    from app.vision import switch as _switch
    snap = _switch.state()
    snap["active"] = _active_id()
    snap["mode"] = str(getattr(cfg.vision, "mode", "local") or "local").lower()
    return snap


@router.post("/analyze")
async def analyze_once(request: Request) -> dict:
    """One-shot image analysis: raw image bytes (PNG/JPEG) in the body → the
    local vision model's structured text out. Never 500s."""
    raw = await request.body()
    if not raw:
        return {"text": ""}
    try:
        import asyncio
        b64 = base64.b64encode(raw).decode("ascii")
        from app.vision.factory import describe_images
        from app.vision.ocr import ocr_images
        # VLM (understanding) + OCR (exact text) concurrently, then merge — same
        # policy as the chat image path.
        vlm_res, ocr_res = await asyncio.gather(
            describe_images([b64], cfg.vision.prompt),
            ocr_images([b64]),
            return_exceptions=True,
        )
        vlm = (vlm_res if isinstance(vlm_res, str) else "").strip()
        ocr = (ocr_res if isinstance(ocr_res, str) else "").strip()
        _ocr_block = (f"[Exact text read from the image (OCR — authoritative)]:"
                      f"\n{ocr[:6000]}" if ocr else "")
        if vlm and ocr:
            text = (f"{_ocr_block}\n\n[Rough visual description (defer to the "
                    f"exact text above)]:\n{vlm}")
        else:
            text = _ocr_block or vlm
    except Exception as exc:  # noqa: BLE001
        log.warning("vision analyze failed: %s", exc)
        return JSONResponse({"text": "", "error": str(exc)}, status_code=200)
    return {"text": (text or "").strip()}
