"""Model download-size helpers for progress bars (2026-07-13).

Local models (STT chain, the bge-m3 embedder) download INSIDE their own
libraries (huggingface_hub, onnx-asr, sentence-transformers), so there is no
per-chunk byte callback to hook. The reliable, cross-library metric is
**bytes-on-disk in the Hugging Face cache dir**, sampled while the download
runs — the same technique `app/stt/switch.py` already uses for the switch popup.

This module centralises three things so both the startup warm-up and the
Settings STT-switch popup report identical, honest progress:

  • `hub_dir(repo)`   — the model's cache directory to sample;
  • `dir_bytes(path)` — current bytes on disk (the "downloaded" number);
  • `est_total(repo)` — the "total" number. Best-effort real total via
    `huggingface_hub` metadata, cached; falls back to a tuned static estimate
    so the bar always has a denominator. Never raises, never blocks the caller
    for long (metadata lookup is wrapped + cached).

Everything is fail-open: any failure yields 0, and the UI simply shows the
downloaded MB without a percentage rather than breaking.
"""
from __future__ import annotations

import os
import pathlib
import threading

# Tuned static download footprints (bytes) keyed by a substring of the repo id.
# These are the FALLBACK denominators when live metadata is unavailable; the UI
# also grows the total to match disk if an estimate turns out too low, so a
# rough number here still yields a sane bar.
_STATIC: dict[str, int] = {
    "bge-m3": 2_300_000_000,
    "bge-large": 1_340_000_000,
    "bge-reranker": 1_120_000_000,
    "parakeet-tdt-0.6b": 680_000_000,
    "Qwen3-ASR": 3_600_000_000,
    "faster-whisper-tiny": 78_000_000,
    "faster-whisper-base": 148_000_000,
    "faster-whisper-small": 500_000_000,
    "faster-whisper-medium": 1_600_000_000,
    "faster-whisper-large": 3_100_000_000,
    # Local vision models (VisionAnalysis.md).
    "SmolVLM-Instruct": 4_500_000_000,
    "SmolVLM-500M": 1_100_000_000,
    "SmolVLM-256M": 600_000_000,
    "Qwen2.5-VL-3B": 7_520_000_000,
    "Qwen2.5-VL-7B": 16_500_000_000,
    "MiniCPM-V-2_6": 16_000_000_000,
}

_cache: dict[str, int] = {}
_lock = threading.Lock()


def hub_dir(repo_id: str | None) -> pathlib.Path | None:
    """The `models--org--name` cache dir HF downloads into (for byte sampling)."""
    if not repo_id or "/" not in repo_id:
        return None
    base = os.environ.get("HF_HUB_CACHE")
    if not base:
        home = os.environ.get("HF_HOME")
        base = (pathlib.Path(home) / "hub") if home else (
            pathlib.Path.home() / ".cache" / "huggingface" / "hub")
    return pathlib.Path(base) / ("models--" + repo_id.replace("/", "--"))


def dir_bytes(path: pathlib.Path | None) -> int:
    """Total bytes currently on disk under `path` (0 if missing)."""
    if path is None or not path.exists():
        return 0
    total = 0
    try:
        for f in path.rglob("*"):
            try:
                if f.is_file():
                    total += f.stat().st_size
            except OSError:
                continue
    except Exception:  # noqa: BLE001 — a size probe must never break a warm-up
        pass
    return total


def _static_total(repo_id: str) -> int:
    for key, size in _STATIC.items():
        if key in repo_id:
            return size
    return 0


def est_total(repo_id: str | None) -> int:
    """Best-effort total download size (bytes) for the progress-bar denominator.

    A tuned static estimate FIRST (instant, offline-safe — the sampler calls
    this every second), and only for an unknown repo a one-shot live HF
    metadata lookup. Result cached. Never raises, never hangs on the known
    models. `0` means "unknown" (UI then shows downloaded MB without a %).
    """
    if not repo_id:
        return 0
    with _lock:
        if repo_id in _cache:
            return _cache[repo_id]
    total = _static_total(repo_id)
    if total <= 0:
        # Unknown model — try live metadata once (best-effort, may be offline).
        try:
            from huggingface_hub import HfApi  # local, always present
            info = HfApi().model_info(repo_id, files_metadata=True)
            total = sum(int(getattr(s, "size", 0) or 0)
                        for s in (info.siblings or []))
        except Exception:  # noqa: BLE001 — offline / rate-limited / odd id
            total = 0
    with _lock:
        _cache[repo_id] = total
    return total


def progress_for(repo_id: str | None) -> tuple[int, int]:
    """`(downloaded_bytes, total_bytes)` for a repo, right now.

    `total` is grown to at least `downloaded` so an under-estimate can't make
    the bar exceed 100%. Returns `(done, 0)` when no total is known (UI then
    shows the downloaded MB without a percentage).
    """
    done = dir_bytes(hub_dir(repo_id))
    total = est_total(repo_id)
    if total and done > total:
        total = done
    return done, total
