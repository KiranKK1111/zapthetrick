"""Startup model warm-up status.

The app warms its local models (STT chain + the sentence-transformers embedder)
in background threads at boot — on the FIRST deploy these download from
HuggingFace (~3 GB), which takes a while. This registry tracks each model's
download/load stage so the UI can show a modal with the live status until
everything is ready (`GET /api/models/warmup`).

Stages: pending -> loading (download + load) -> ready | error | skipped.

Design: in-process, thread-safe (the warm-up runs in daemon threads), and
fail-open — a status bug must never break startup or a warm-up.
"""
from __future__ import annotations

import os
import pathlib
import threading
import time
from typing import Any

STAGE_PENDING = "pending"
STAGE_LOADING = "loading"   # covers both download and in-RAM load
STAGE_READY = "ready"
STAGE_ERROR = "error"
STAGE_SKIPPED = "skipped"   # provider not installed / not applicable

_TERMINAL = {STAGE_READY, STAGE_ERROR, STAGE_SKIPPED}

_LOCK = threading.Lock()
_MODELS: dict[str, dict[str, Any]] = {}
_ORDER: list[str] = []


def hf_repo_cached(repo_id: str | None) -> bool | None:
    """Best-effort: is this HuggingFace repo already in the local hub cache?

    Returns True/False when it can tell, None when unknown (odd repo id /
    lookup failure). Used to decide whether the warm-up is a FIRST-RUN
    DOWNLOAD (worth a blocking modal) or just an in-RAM load from disk
    (silent). Checks for `models--{org}--{name}/snapshots/<something>` under
    the hub cache dir — the layout huggingface_hub has used for years.
    """
    try:
        if not repo_id or "/" not in repo_id:
            return None
        cache = os.environ.get("HF_HUB_CACHE")
        if not cache:
            home = os.environ.get("HF_HOME")
            cache = (pathlib.Path(home) / "hub") if home else (
                pathlib.Path.home() / ".cache" / "huggingface" / "hub")
        repo_dir = pathlib.Path(cache) / f"models--{repo_id.replace('/', '--')}"
        snaps = repo_dir / "snapshots"
        return snaps.is_dir() and any(snaps.iterdir())
    except Exception:  # noqa: BLE001 — a cache probe must never break warm-up
        return None


def register(key: str, name: str, cached: bool | None = None) -> None:
    """Declare a model to warm (idempotent). Starts in `pending`.

    `cached` says whether the model's weights are already on disk (True),
    definitely need downloading (False), or unknown (None). Only an explicit
    False makes the UI show the first-run download modal.
    """
    try:
        with _LOCK:
            if key not in _MODELS:
                _ORDER.append(key)
            _MODELS[key] = {
                "key": key,
                "name": name,
                "stage": STAGE_PENDING,
                "detail": "",
                "cached": cached,
                "bytes_done": 0,
                "bytes_total": 0,
                "pct": None,
                "updated_at": time.time(),
            }
    except Exception:  # noqa: BLE001 — status must never break the warm-up
        pass


def set_progress(key: str, bytes_done: int, bytes_total: int) -> None:
    """Report byte-level download progress for a model row (drives the bar).

    `bytes_total == 0` means "unknown total" — the UI then shows the downloaded
    MB without a percentage. Auto-registers if unseen. Fail-open.
    """
    try:
        with _LOCK:
            m = _MODELS.get(key)
            if m is None:
                _MODELS[key] = m = {
                    "key": key, "name": key, "stage": STAGE_LOADING,
                    "detail": "", "updated_at": time.time(),
                }
                _ORDER.append(key)
            done = max(0, int(bytes_done or 0))
            total = max(0, int(bytes_total or 0))
            if total and done > total:
                total = done
            m["bytes_done"] = done
            m["bytes_total"] = total
            m["pct"] = round(done / total * 100.0, 1) if total else None
            m["updated_at"] = time.time()
    except Exception:  # noqa: BLE001
        pass


def set_stage(key: str, stage: str, detail: str = "") -> None:
    """Update a model's stage (auto-registers if unseen)."""
    try:
        with _LOCK:
            m = _MODELS.get(key)
            if m is None:
                _MODELS[key] = m = {
                    "key": key, "name": key, "stage": stage,
                    "detail": detail, "updated_at": time.time(),
                }
                _ORDER.append(key)
                return
            m["stage"] = stage
            if detail:
                m["detail"] = detail
            m["updated_at"] = time.time()
    except Exception:  # noqa: BLE001
        pass


def snapshot() -> dict[str, Any]:
    """Current status for the UI: per-model list + overall readiness."""
    try:
        with _LOCK:
            models = [dict(_MODELS[k]) for k in _ORDER if k in _MODELS]
    except Exception:  # noqa: BLE001
        models = []
    # A model that reached READY is 100% by definition, even if the last byte
    # sample lagged — normalise so the bar always lands full when done.
    for m in models:
        if m.get("stage") == STAGE_READY:
            bt = m.get("bytes_total") or m.get("bytes_done") or 0
            if bt:
                m["bytes_done"] = bt
                m["bytes_total"] = bt
            m["pct"] = 100.0
    total = len(models)
    done = sum(1 for m in models if m.get("stage") in _TERMINAL)
    # Overall byte progress across models that actually report a total (an
    # honest "X GB of Y GB" for the headline bar; falls back to the count-based
    # percent when no byte totals are known yet).
    bytes_done = sum(int(m.get("bytes_done") or 0) for m in models)
    bytes_total = sum(int(m.get("bytes_total") or 0) for m in models)
    # `all_ready` is True once every declared model has reached a terminal
    # state (ready/error/skipped) — the modal dismisses on that. If NOTHING
    # was ever registered (warm-up disabled / already cached before this ran),
    # treat as ready so the UI never gets stuck.
    all_ready = (total == 0) or all(
        m.get("stage") in _TERMINAL for m in models)
    active = any(m.get("stage") in (STAGE_PENDING, STAGE_LOADING)
                 for m in models)
    # First-run signal for the UI: True only when some model that is NOT on
    # disk yet (cached == False) is still pending/loading — i.e. an actual
    # download is (about to be) happening. A warm start (weights cached, just
    # loading into RAM) keeps this False so the client can skip the blocking
    # "Preparing models" screen entirely.
    needs_download = any(
        m.get("cached") is False
        and m.get("stage") in (STAGE_PENDING, STAGE_LOADING)
        for m in models)
    return {
        "models": models,
        "total": total,
        "done_count": done,
        "all_ready": all_ready,
        "any_active": active,
        "needs_download": needs_download,
        "percent": round((done / total) * 100.0, 1) if total else 100.0,
        "bytes_done": bytes_done,
        "bytes_total": bytes_total,
        "byte_percent": (round(bytes_done / bytes_total * 100.0, 1)
                         if bytes_total else None),
    }


def reset_for_tests() -> None:
    with _LOCK:
        _MODELS.clear()
        _ORDER.clear()
