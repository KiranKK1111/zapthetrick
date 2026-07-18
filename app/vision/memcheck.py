"""Pre-flight memory guard for local vision engines.

A vision model that doesn't fit in free VRAM/RAM doesn't raise a catchable
`MemoryError` — the load crashes NATIVELY (safetensors/accelerate copy into a
too-small allocation), and on this hardware that surfaces as a **segmentation
fault** that kills the whole process. A dead backend is far worse than a missing
parse, so we must never *attempt* an impossible load.

`pick_device()` estimates a model's resident footprint from its on-disk weight
size and compares it against the memory actually free right now, returning the
device it will fit on ("cuda"/"cpu") or raising an ordinary `VisionOOM` — which
the factory catches and fails open on (next engine, then a graceful text note).

The estimate is deliberately conservative (headroom for activations + the KV
cache during generation + other processes), so we err toward refusing a load
that *might* crash rather than risking the segfault.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

# Resident-footprint multipliers over the on-disk (bf16) weight size:
#   • GPU keeps the bf16 weights ~as-is, plus activations + KV cache in-flight.
#   • CPU fallback upcasts to float32 (2x the bytes) + framework overhead.
# The free-memory margins leave room for the OS, other models (STT/embedder)
# and generation working set — a load that clears these has real breathing room.
# (Empirically, SmolVLM-2.2B's ~4.5GB of weights occupy ~6.4GB live on GPU once
# the vision encoder + working buffers are resident — ~1.45x — so the GPU factor
# is tuned to that, not to the raw weight size. Image resolution is capped in
# the engines so the generate-time activation VRAM stays bounded.)
_GPU_FACTOR = 1.45
_CPU_FACTOR = 2.30
_GPU_HEADROOM_BYTES = 400 * 1024 * 1024   # never fill VRAM to the brim
_CPU_HEADROOM_BYTES = 1024 * 1024 * 1024  # leave the OS a gigabyte


class VisionOOM(RuntimeError):
    """Raised when a model won't fit in free VRAM/RAM (catchable — fail-open)."""


def _weight_bytes(repo: str) -> int:
    """Best-effort weight size in bytes.

    Uses the tuned/metadata estimate (`est_total`) — NOT `dir_bytes`, which
    double-counts the HuggingFace cache on Windows (a model is stored both as a
    `blobs/` object AND a `snapshots/` copy when symlinks aren't available, so
    rglob sums it twice → a ~2x overestimate that would wrongly refuse a model
    that actually fits). `est_total` returns the real total download size."""
    try:
        from app import model_sizes
        est = model_sizes.est_total(repo)
        if est > 0:
            return est
        # Last resort: count only the de-duplicated blobs dir, never the
        # snapshot copies, so we don't double-count.
        blobs = model_sizes.hub_dir(repo)
        if blobs is not None:
            return model_sizes.dir_bytes(blobs / "blobs")
    except Exception as exc:  # noqa: BLE001
        log.info("vision.memcheck: size probe failed for %s (%s)", repo, exc)
    return 0


def _free_vram() -> int:
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return int(free)
    except Exception:  # noqa: BLE001
        pass
    return 0


def _free_ram() -> int:
    try:
        import psutil  # noqa: PLC0415
        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001
        return 0


def pick_device(repo: str, *, prefer_gpu: bool) -> str:
    """Return "cuda" or "cpu" — whichever the model actually fits on right now —
    or raise `VisionOOM`. GPU is tried first when `prefer_gpu`; a model that fits
    neither raises rather than risking a native segfault.

    If the weight size is unknown (probe failed, weights not yet downloaded) we
    can't guarantee a fit, so we optimistically allow the preferred device — the
    caller still runs under the factory's try/except, and the common case (a
    small model on a machine that has room) is unaffected."""
    want = _weight_bytes(repo)
    if want <= 0:
        # Unknown size — can't preflight; let the load attempt proceed.
        return "cuda" if (prefer_gpu and _free_vram() > 0) else "cpu"

    gpu_need = int(want * _GPU_FACTOR) + _GPU_HEADROOM_BYTES
    cpu_need = int(want * _CPU_FACTOR) + _CPU_HEADROOM_BYTES
    free_vram = _free_vram()
    free_ram = _free_ram()

    def _gb(n: int) -> str:
        return f"{n / 1e9:.1f}GB"

    if prefer_gpu and free_vram >= gpu_need:
        log.info("vision.memcheck: %s -> cuda (need ~%s, free %s)",
                 repo, _gb(gpu_need), _gb(free_vram))
        return "cuda"
    if free_ram >= cpu_need:
        if prefer_gpu:
            log.info("vision.memcheck: %s won't fit VRAM (need ~%s, free %s) — "
                     "CPU (need ~%s, free RAM %s)", repo, _gb(gpu_need),
                     _gb(free_vram), _gb(cpu_need), _gb(free_ram))
        return "cpu"
    raise VisionOOM(
        f"'{repo}' needs ~{_gb(gpu_need)} VRAM (free {_gb(free_vram)}) or "
        f"~{_gb(cpu_need)} RAM (free {_gb(free_ram)}); refusing the load to "
        f"avoid a native OOM crash. Pick a smaller vision model in Settings.")
