"""STT model selection — LOCAL ONLY (2026-07-12 policy: no cloud STT, no
fallback chain; ONE engine resident at a time so a KVM VPS can budget its
memory deliberately).

GET /api/stt/models — the local engines this install can run: Parakeet,
Qwen3-ASR (Hugging Face), and the faster-whisper size ladder — each with its
downloaded state (models download automatically on first use, on desktop and
server alike).

Entry ids: "parakeet" | "qwen_asr" | "faster_whisper::<size>". The client
writes the selection EXCLUSIVELY through POST /api/settings (provider +
partial_provider pinned to the same engine, fallbacks cleared, dual engine
off); the settings bus then frees every other engine and warms the chosen
one, so exactly one model holds memory.
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.core.config_loader import cfg

log = logging.getLogger("zapthetrick.stt")
router = APIRouter(prefix="/api/stt", tags=["stt"])

_PARAKEET_REPO_PREFIX = "istupakov/"

# faster-whisper local size ladder (CTranslate2; downloads on first use).
_WHISPER_SIZES = [
    ("tiny.en", "fastest, lowest accuracy"),
    ("base.en", "fast, light"),
    ("small.en", "balanced"),
    ("medium.en", "accurate, slower"),
    ("large-v3", "most accurate, multilingual, needs a strong machine"),
]


def _parakeet_repo() -> str:
    model = str(getattr(cfg.stt, "parakeet_model",
                        "nemo-parakeet-tdt-0.6b-v3"))
    return f"{_PARAKEET_REPO_PREFIX}{model}-onnx"


def _downloaded(repo: str | None) -> bool:
    if not repo:
        return True
    try:
        from app.models_warmup import hf_repo_cached
        return bool(hf_repo_cached(repo))
    except Exception:  # noqa: BLE001
        return False


def _active_id() -> str:
    provider = str(getattr(cfg.stt, "provider", "parakeet"))
    if provider == "faster_whisper":
        return f"faster_whisper::{getattr(cfg.stt, 'model', 'base.en')}"
    return provider


@router.get("/models")
async def stt_models() -> dict:
    """The selectable LOCAL STT engines, richly labeled for the dropdown."""
    qwen_model = str(getattr(cfg.stt, "qwen_model", "Qwen/Qwen3-ASR-1.7B"))
    models: list[dict] = [
        {
            "id": "parakeet",
            "label": "Parakeet TDT 0.6B v3 — fast",
            "detail": "NVIDIA NeMo ONNX model running on this machine "
                      "(GPU when available). Best speed; the recommended "
                      "default.",
            "kind": "local",
            "downloaded": _downloaded(_parakeet_repo()),
        },
        {
            "id": "qwen_asr",
            "label": "Qwen3-ASR 1.7B (Hugging Face) — most accurate",
            "detail": f"{qwen_model} running on this machine. Highest "
                      "accuracy; needs a GPU (or patience) to be fast.",
            "kind": "local",
            "downloaded": _downloaded(qwen_model),
        },
    ]
    for size, note in _WHISPER_SIZES:
        models.append({
            "id": f"faster_whisper::{size}",
            "label": f"Whisper {size} — {note.split(',')[0]}",
            "detail": f"faster-whisper (CTranslate2) on this machine — "
                      f"{note}.",
            "kind": "local",
            "downloaded": True,   # small; fetched on first use
        })

    active = _active_id()
    for m in models:
        m["active"] = m["id"] == active
    return {"models": models, "active": active}


class SelectBody(BaseModel):
    id: str


def _valid_ids() -> set[str]:
    ids = {"parakeet", "qwen_asr"}
    ids.update(f"faster_whisper::{size}" for size, _ in _WHISPER_SIZES)
    return ids


@router.post("/select")
async def select_model(body: SelectBody) -> dict:
    """EXCLUSIVE engine selection: persist the settings (one engine for
    partials + finals, no fallbacks, dual off) and start the TRACKED switch
    the popup observes via GET /api/stt/status."""
    target = (body.id or "").strip()
    if target not in _valid_ids():
        raise HTTPException(400, detail=f"Unknown STT model '{target}'.")
    provider = target.split("::")[0]
    partial: dict = {
        "provider": provider,
        "partial_provider": provider,
        "fallback_providers": [],
        "dual_engine_enabled": False,
    }
    if "::" in target:
        partial["model"] = target.split("::", 1)[1]
    from app.api.routes_settings import write_settings
    await write_settings({"stt": partial})
    # write_settings' bus publish already delegated to the switch tracker
    # (subscribers._on_stt) — starting it here too just joins single-flight.
    from app.stt import switch as _switch
    await _switch.start_switch(target)
    return {"ok": True, "switching_to": target}


class ModeBody(BaseModel):
    mode: str


@router.post("/mode")
async def set_mode(body: ModeBody) -> dict:
    """Switch STT between LOCAL (Parakeet/Qwen-ASR, dropdown-selected) and CLOUD
    (Groq Whisper API, using the stored Groq key)."""
    mode = (body.mode or "").strip().lower()
    if mode not in ("local", "cloud"):
        raise HTTPException(400, detail="mode must be 'local' or 'cloud'")
    from app.api.routes_settings import write_settings
    await write_settings({"stt": {"mode": mode}})
    return {"ok": True, "mode": mode}


@router.get("/status")
async def switch_status() -> dict:
    """The live switch state for the popup: phase, freed engines, download
    bytes, resident engines (the truth about what's in memory), active id."""
    from app.stt import switch as _switch
    snap = _switch.state()
    snap["active"] = _active_id()
    snap["mode"] = str(getattr(cfg.stt, "mode", "local") or "local").lower()
    return snap


@router.post("/transcribe")
async def transcribe_once(request: Request) -> dict:
    """One-shot dictation (chat voice input, UI gap-fill #4). Body is raw 16-bit
    little-endian PCM, mono, at ``cfg.audio.sample_rate`` (16 kHz) — exactly what
    the Flutter recorder streams — so there's no container to parse. Runs the
    same local STT engine the live path uses and returns the transcript."""
    raw = await request.body()
    if not raw or len(raw) < 2:
        return {"text": ""}
    try:
        import numpy as np  # noqa: PLC0415

        audio_np = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        from app.stt.factory import transcribe_async  # noqa: PLC0415

        text = await transcribe_async(audio_np)
    except Exception as exc:  # noqa: BLE001 — never 500 the composer's mic
        log.warning("dictation transcribe failed: %s", exc)
        return JSONResponse({"text": "", "error": str(exc)}, status_code=200)
    return {"text": (text or "").strip()}
