"""Tracked STT engine switching (2026-07-12): the observable state machine
behind the Settings popup.

    unloading → downloading → loading → ready | error

One switch at a time (single-flight — re-selecting the same target joins the
running switch instead of double-loading). The state answers the popup's
three questions honestly:

  • is the PREVIOUS model actually dead?  → `freed_engines` +
    `factory.resident_engines()` (live truth, not a claim);
  • how is the download going?            → sampled byte growth of the
    engine's Hugging Face cache dir (models download inside their libs, so
    bytes-on-disk is the reliable cross-library metric);
  • is the new model live?                → phase `ready` after a real
    warm-up transcription succeeded.

Works identically on the VPS: single-process uvicorn owns this state, and
the HF cache lives on the persisted volume.
"""
from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import time

log = logging.getLogger("zapthetrick.stt")

_STATE: dict = {"phase": "idle"}
_LOCK = asyncio.Lock()
_TASK: asyncio.Task | None = None


def state() -> dict:
    """Snapshot of the current/most-recent switch."""
    snap = dict(_STATE)
    try:
        from app.stt import factory
        snap["resident_engines"] = factory.resident_engines()
    except Exception:  # noqa: BLE001
        snap["resident_engines"] = []
    return snap


def _repo_for(provider: str) -> str | None:
    """The Hugging Face repo id backing an STT provider (or None if unknown)."""
    from app.core.config_loader import cfg
    if provider == "parakeet":
        model = str(getattr(cfg.stt, "parakeet_model",
                            "nemo-parakeet-tdt-0.6b-v3"))
        return f"istupakov/{model}-onnx"
    if provider == "qwen_asr":
        return str(getattr(cfg.stt, "qwen_model", "Qwen/Qwen3-ASR-1.7B"))
    if provider == "faster_whisper":
        size = str(getattr(cfg.stt, "model", "base.en"))
        return f"Systran/faster-whisper-{size}"
    return None


def _hf_dir_for(provider: str) -> pathlib.Path | None:
    """The engine's Hugging Face cache dir (for byte-growth sampling)."""
    repo = _repo_for(provider)
    if not repo:
        return None
    base = os.environ.get("HF_HOME")
    hub = (pathlib.Path(base) / "hub") if base \
        else pathlib.Path.home() / ".cache" / "huggingface" / "hub"
    return hub / ("models--" + repo.replace("/", "--"))


def _dir_bytes(p: pathlib.Path | None) -> int:
    if p is None or not p.exists():
        return 0
    total = 0
    try:
        for f in p.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                continue
    except Exception:  # noqa: BLE001
        pass
    return total


async def start_switch(target_id: str) -> None:
    """Kick off (or join) the tracked switch to `target_id`. The settings
    are assumed already persisted — this handles memory + load + state."""
    global _TASK
    async with _LOCK:
        if (_TASK is not None and not _TASK.done()
                and _STATE.get("to") == target_id):
            return                       # join the in-flight switch
        if _TASK is not None and not _TASK.done():
            _TASK.cancel()
        _TASK = asyncio.create_task(_run(target_id))


async def _run(target_id: str) -> None:
    from app.stt import factory
    provider = target_id.split("::")[0]
    before = factory.resident_engines()
    _STATE.clear()
    _STATE.update({
        "phase": "unloading", "to": target_id,
        "previous_resident": before, "freed_engines": [],
        "started_at": time.time(), "downloaded_bytes": 0,
        "was_downloaded": False, "error": None,
    })
    try:
        factory.unload_all()
        _STATE["freed_engines"] = [e for e in before
                                   if e not in factory.resident_engines()]

        hf_dir = _hf_dir_for(provider)
        already = _dir_bytes(hf_dir)
        _STATE["was_downloaded"] = already > 1_000_000
        _STATE["phase"] = ("loading" if _STATE["was_downloaded"]
                           else "downloading")
        # Total for the progress bar's denominator (best-effort; grows to match
        # disk if the estimate is low). 0 ⇒ UI shows downloaded MB only.
        _repo = _repo_for(provider)
        try:
            from app import model_sizes as _ms
            _STATE["total_bytes"] = _ms.est_total(_repo)
        except Exception:  # noqa: BLE001
            _STATE["total_bytes"] = 0

        def _set_bytes() -> None:
            done = _dir_bytes(hf_dir)
            total = _STATE.get("total_bytes") or 0
            if total and done > total:
                total = done
                _STATE["total_bytes"] = total
            _STATE["downloaded_bytes"] = done
            _STATE["percent"] = (round(done / total * 100.0, 1)
                                 if total else None)

        async def _sample() -> None:
            while _STATE["phase"] in ("downloading", "loading"):
                _set_bytes()
                await asyncio.sleep(1.0)

        sampler = asyncio.create_task(_sample())
        try:
            await factory.warm_active()
        finally:
            _set_bytes()
            sampler.cancel()

        # The warm-up swallows engine errors by design — the honest "ready"
        # check is: is the target engine actually resident now?
        resident = factory.resident_engines()
        if provider in resident:
            _STATE["phase"] = "ready"
            # Land the bar full — the estimate may differ from the exact files
            # a given engine pulls, but a resident engine is 100% done.
            _dl = _STATE.get("downloaded_bytes") or 0
            if _dl:
                _STATE["total_bytes"] = _dl
            _STATE["percent"] = 100.0
        else:
            _STATE["phase"] = "error"
            _STATE["error"] = ("The engine did not load — it will retry on "
                               "the next spoken sentence.")
        _STATE["finished_at"] = time.time()
    except asyncio.CancelledError:
        _STATE["phase"] = "cancelled"
        raise
    except Exception as exc:  # noqa: BLE001
        log.warning("stt switch to %s failed: %s", target_id, exc)
        _STATE["phase"] = "error"
        _STATE["error"] = str(exc)[:200]
        _STATE["finished_at"] = time.time()


__all__ = ["start_switch", "state"]
