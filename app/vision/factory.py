"""Local Vision Intelligence Layer — the dispatcher (VisionAnalysis.md).

Mirrors `app/stt/factory.py`. One local vision engine is resident at a time,
chosen by `cfg.vision.provider` with `cfg.vision.fallback_providers` as the
runtime fallback chain. The public entry point `describe_images` decodes the
base64 image side-channel, consults the per-image cache, then runs the chain —
each engine on a worker thread so a sync model never blocks the event loop.

Adding an engine = write a module exposing `describe(images, prompt) -> str`,
`unload()` and `is_loaded()`, then register its `describe` in `_PROVIDERS` and
its `is_loaded` in `_RESIDENCE`.
"""
from __future__ import annotations

import asyncio
import base64
import binascii
import gc
import logging
from typing import Callable, Sequence

from . import cache as _cache
from . import minicpm, qwen_vl, smolvlm

log = logging.getLogger(__name__)

# Default HF repos for the two SmolVLM sizes (config overrides win).
_SMOLVLM_SMALL = "HuggingFaceTB/SmolVLM-500M-Instruct"
_SMOLVLM_BIG = "HuggingFaceTB/SmolVLM-Instruct"


def _smolvlm_repo(field: str, default: str) -> str:
    return str(getattr(_cfg(), field, default) or default)


def _smolvlm_describe(field: str, default: str) -> Callable[[Sequence[bytes], str], str]:
    """Bind a SmolVLM size (by config field) to the shared engine."""
    def _fn(images: Sequence[bytes], prompt: str) -> str:
        return smolvlm.describe(images, prompt, _smolvlm_repo(field, default))
    return _fn


# provider id -> describe(images_bytes, prompt) -> str.
# SmolVLM-500M first: the right-sized default that fits a laptop GPU alongside
# the resident STT model. Larger models (2.2B / Qwen / MiniCPM) are more
# accurate but need more VRAM — memcheck-gated so they fail OPEN, not crash.
_PROVIDERS: dict[str, Callable[[Sequence[bytes], str], str]] = {
    "smolvlm_500m": _smolvlm_describe("smolvlm_small_model", _SMOLVLM_SMALL),
    "smolvlm": _smolvlm_describe("smolvlm_model", _SMOLVLM_BIG),
    "qwen2_5_vl": qwen_vl.describe,
    "minicpm_v": minicpm.describe,
}

# provider id -> (is_loaded, unload) so unload_all / residence work generically.
# Both SmolVLM sizes share one engine (one resident at a time); is_loaded_repo
# keeps the resident readout honest about WHICH size is live.
_RESIDENCE: dict[str, tuple[Callable[[], bool], Callable[[], None]]] = {
    "smolvlm_500m": (lambda: smolvlm.is_loaded_repo(
        _smolvlm_repo("smolvlm_small_model", _SMOLVLM_SMALL)), smolvlm.unload),
    "smolvlm": (lambda: smolvlm.is_loaded_repo(
        _smolvlm_repo("smolvlm_model", _SMOLVLM_BIG)), smolvlm.unload),
    "qwen2_5_vl": (qwen_vl.is_loaded, qwen_vl.unload),
    "minicpm_v": (minicpm.is_loaded, minicpm.unload),
}


def _cfg():
    from app.core.config_loader import cfg
    return cfg.vision


def available_providers() -> list[str]:
    return list(_PROVIDERS.keys())


# Local providers capable enough to read a language chip WITHOUT leaning on OCR.
# SmolVLM-500M is the "small" hallucinator — it needs the OCR safety net.
_CAPABLE_LOCAL = frozenset({"smolvlm", "qwen2_5_vl", "minicpm_v"})


def vision_capability() -> str:
    """"capable" when the active vision reader can be TRUSTED to read the image
    (cloud mode, or a LARGE local model) — OCR is then only a cross-check/
    fallback. "small" when it's the tiny local model (SmolVLM-500M) that
    hallucinates — OCR is authoritative there. Drives OCR reliance in the
    coding-language pipeline. Never raises."""
    try:
        v = _cfg()
        if str(getattr(v, "mode", "local")).lower() == "cloud":
            return "capable"
        return ("capable" if str(getattr(v, "provider", "") or "")
                in _CAPABLE_LOCAL else "small")
    except Exception:  # noqa: BLE001
        return "small"


def _provider_chain() -> list[str]:
    """Active provider first, then the configured fallbacks (deduped)."""
    v = _cfg()
    chain = [v.provider, *(v.fallback_providers or [])]
    seen: set[str] = set()
    out: list[str] = []
    for p in chain:
        if p and p in _PROVIDERS and p not in seen:
            seen.add(p)
            out.append(p)
    return out or (["smolvlm_500m"] if "smolvlm_500m" in _PROVIDERS else [])


def _decode(images_b64: Sequence[str]) -> list[bytes]:
    out: list[bytes] = []
    for s in images_b64:
        if not s:
            continue
        raw = s.split(",", 1)[1] if s.startswith("data:") else s
        try:
            out.append(base64.b64decode(raw))
        except (binascii.Error, ValueError) as exc:
            log.info("vision: skipping undecodable base64 image (%s)", exc)
    return out


async def describe_images(images_b64: Sequence[str], prompt: str) -> str:
    """Parse image(s) into a structured TEXT description using the LOCAL model
    chain. Base64 side-channel in (the `images:[…]` message shape), text out.

    Cached by (image bytes, prompt) so the same screenshot/document is parsed
    once. Returns "" if every engine failed (the caller then decides what to do
    — never silently sends the raw image to a provider). Never raises."""
    v = _cfg()
    if not getattr(v, "enabled", True):
        return ""
    imgs = _decode(images_b64)
    if not imgs:
        return ""

    prompt = prompt or v.prompt
    cache = _cache.get_cache(int(getattr(v, "cache_max_entries", 128)))
    key = _cache.image_key(imgs, prompt) if getattr(v, "cache_enabled", True) else None
    if key is not None:
        hit = cache.get(key)
        if hit is not None:
            log.info("vision: cache hit (%d chars)", len(hit))
            return hit

    # CLOUD mode (Settings toggle): send the image to a vision-capable provider
    # for extraction. The image is still stored in Postgres by the caller; it
    # goes ONLY to the vision model here (never to the answer LLM). Falls back to
    # the local chain if the cloud call comes back empty.
    if str(getattr(v, "mode", "local")).lower() == "cloud":
        text = (await _cloud_describe(images_b64, prompt)).strip()
        if text:
            log.info("vision: cloud model parsed image(s) (%d chars)", len(text))
            if key is not None:
                cache.put(key, text)
            return text
        log.info("vision: cloud model returned empty — falling back to local")

    last_err: Exception | None = None
    for name in _provider_chain():
        fn = _PROVIDERS.get(name)
        if fn is None:
            continue
        try:
            text = await asyncio.to_thread(fn, imgs, prompt)
        except Exception as exc:  # noqa: BLE001 — try the next engine
            last_err = exc
            log.info("vision: engine '%s' failed (%s) — falling back", name, exc)
            continue
        text = (text or "").strip()
        if text:
            log.info("vision: '%s' parsed image(s) (%d chars)", name, len(text))
            if key is not None:
                cache.put(key, text)
            return text
        log.info("vision: engine '%s' returned empty — falling back", name)
    if last_err is not None:
        log.info("vision: all engines failed; last error: %s", last_err)
    return ""


def _shrink_for_upload(b64: str, max_side: int, jpeg_q: int) -> str:
    """Downscale a base64 image to `max_side` on its longest edge and re-encode
    as JPEG so the CLOUD upload isn't paying for pixels the vision model tiles
    away anyway (hosted models cap ~1568px). Cuts the payload — and therefore
    upload + vision-prefill time — several-fold with no OCR loss. Returns the
    ORIGINAL string on any failure (correctness over speed)."""
    try:
        import base64  # noqa: PLC0415
        import io  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415
        raw = base64.b64decode(b64)
        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        if max(w, h) <= max_side and (img.format or "").upper() in {"JPEG", "JPG"}:
            return b64  # already small + already JPEG — nothing to gain
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                             Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=jpeg_q, optimize=True)
        out = base64.b64encode(buf.getvalue()).decode("ascii")
        # Only adopt the re-encode if it actually shrank the payload.
        return out if len(out) < len(b64) else b64
    except Exception as exc:  # noqa: BLE001 — never fail the parse over a resize
        log.info("vision: cloud upload-shrink skipped (%s)", exc)
        return b64


async def _cloud_describe(images_b64: Sequence[str], prompt: str) -> str:
    """Cloud vision: extract the image via a vision-capable provider LLM (the
    router auto-selects a vision model because the message carries an image).

    RESILIENT: a flaky/slow/empty free vision model must never sink the read —
    each attempt ESCALATES the routing tier AND avoids the model that just
    failed/returned empty, so a different model is tried until one answers.
    Returns "" only when EVERY attempt failed (the caller then falls back to the
    local chain, so the user still gets a result). Never raises."""
    try:
        from app.core.llm_client import llm
        v = _cfg()
        max_side = int(getattr(v, "cloud_max_side", 1568) or 1568)
        jpeg_q = int(getattr(v, "cloud_jpeg_quality", 85) or 85)
        raw = [(s.split(",", 1)[1] if s.startswith("data:") else s)
               for s in images_b64 if s]
        if not raw:
            return ""
        raw = [_shrink_for_upload(s, max_side, jpeg_q) for s in raw]
        msgs = [{"role": "user", "content": prompt, "images": raw}]
        caps = int(getattr(v, "cloud_max_tokens", 1500) or 1500)
        # Start FAST ("trivial" = speed-only vision model) then climb to stronger
        # tiers on retry — a slow/empty free model is replaced by a different one.
        base_tier = str(getattr(v, "cloud_difficulty", "trivial") or "trivial")
        # No "hard" tier — that routes a transcription to a slow reasoning model.
        # `avoid_model_db_id` already rotates to a DIFFERENT model each retry, so
        # variety doesn't need tier-climbing into the slow tail.
        tiers = [base_tier, "standard", "standard"]
        retries = max(1, int(getattr(v, "cloud_retries", 3) or 3))
        # Per-attempt deadline: without it a slow/hanging free vision model blocks
        # for the GLOBAL llm.timeout_seconds (120s) per try — the "Reading image
        # is taking longer" stall. Cap each attempt so a stuck model is abandoned
        # fast and a DIFFERENT one is tried within seconds, not minutes.
        attempt_to = float(getattr(v, "cloud_attempt_timeout", 28.0) or 28.0)
        avoid: int | None = None
        for i in range(retries):
            opts: dict = {"difficulty": tiers[min(i, len(tiers) - 1)],
                          "num_predict": caps}
            if avoid is not None:
                # Force a DIFFERENT model than the one that just came back empty.
                opts["avoid_model_db_id"] = avoid
            try:
                text, mid = await asyncio.wait_for(
                    llm.complete_routed(msgs, None, opts), timeout=attempt_to)
                text = (text or "").strip()
                if text:
                    if i:
                        log.info("vision: cloud read succeeded on attempt %d", i + 1)
                    return text
                avoid = mid  # empty → steer off this model next time
                log.info("vision: cloud attempt %d returned empty — retrying "
                         "with a different model", i + 1)
            except asyncio.TimeoutError:
                # This model is too slow — drop it and try another immediately.
                log.info("vision: cloud attempt %d exceeded %.0fs — dropping this "
                         "model, trying another", i + 1, attempt_to)
            except Exception as exc:  # noqa: BLE001 — try the next model/tier
                log.info("vision: cloud attempt %d failed (%s) — retrying",
                         i + 1, exc)
                await asyncio.sleep(0.6)
        return ""
    except Exception as exc:  # noqa: BLE001
        log.info("vision: cloud describe failed (%s)", exc)
        return ""


def resident_engines() -> list[str]:
    """Which engines are actually in memory right now (honest readout)."""
    return [name for name, (loaded, _) in _RESIDENCE.items() if _safe(loaded)]


def _safe(fn: Callable[[], bool]) -> bool:
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001
        return False


def unload_all() -> None:
    """Free every resident vision engine — one model resident at a time."""
    for _name, (_loaded, unload) in _RESIDENCE.items():
        try:
            unload()
        except Exception as exc:  # noqa: BLE001
            log.info("vision: unload '%s' failed (%s)", _name, exc)
    _cache.get_cache().clear()
    gc.collect()
    try:
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


async def warm_active() -> None:
    """Push a small valid image through the active LOCAL engine so the first
    real parse doesn't pay the cold load. Best-effort; never raises. (A 1x1
    image is too small for a VLM's patch encoder — use a real 64x64 tile.)

    NO-OP in CLOUD mode: there is no local model to pre-load, and the warmup runs
    in a throwaway thread event loop — a cloud HTTP call there would build the
    shared httpx pool bound to that (soon-closed) loop and poison it for the main
    server loop ("Event loop is closed" on later triage/LLM calls)."""
    try:
        if str(getattr(_cfg(), "mode", "local")).lower() == "cloud":
            return
        import base64  # noqa: PLC0415
        import io  # noqa: PLC0415

        from PIL import Image  # noqa: PLC0415
        buf = io.BytesIO()
        Image.new("RGB", (64, 64), (238, 238, 238)).save(buf, format="PNG")
        px = base64.b64encode(buf.getvalue()).decode("ascii")
        await describe_images([px], "Reply with the single word: ok.")
    except Exception as exc:  # noqa: BLE001
        log.info("vision: warm_active failed (%s)", exc)
