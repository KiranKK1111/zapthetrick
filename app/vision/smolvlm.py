"""SmolVLM local vision engine (VisionAnalysis.md — the right-sized default).

SmolVLM reads UI screenshots, documents, tables and charts to text accurately
while fitting a laptop GPU. Two sizes share this one engine:
  • SmolVLM-500M (~1 GB live) — the fits-everywhere default; coexists with the
    resident STT model on an 8 GB GPU with room to spare.
  • SmolVLM-2.2B (~6.4 GB live) — more accurate, for machines with a bigger GPU.
Both use the STANDARD transformers image-text-to-text API, so this engine
mirrors `qwen_vl.py` closely. The active repo is passed in by the factory (one
provider id per size); only one model is resident at a time.

Single-flight load via an explicit lock + module global. GPU-first via the
`memcheck` pre-flight (which refuses an impossible load with a CATCHABLE error
instead of letting it segfault the process), CPU fallback when it fits in RAM.
Everything is fail-open: any error raises so the factory falls through.
"""
from __future__ import annotations

import io
import logging
import threading
from typing import Sequence

log = logging.getLogger(__name__)

_DEFAULT_REPO = "HuggingFaceTB/SmolVLM-500M-Instruct"

# (repo, model, processor, device) — one instance resident. Tagged with the repo
# so a size switch swaps cleanly instead of serving the wrong weights.
_cache: tuple[str, object, object, str] | None = None
_lock = threading.Lock()


def _cfg():
    from app.core.config_loader import cfg
    return cfg.vision


def _load(repo: str) -> tuple[object, object, str]:
    """Load the requested SmolVLM once (double-checked locking). Pre-flighted."""
    global _cache
    if _cache is not None and _cache[0] == repo:
        return _cache[1], _cache[2], _cache[3]
    with _lock:
        if _cache is not None and _cache[0] == repo:
            return _cache[1], _cache[2], _cache[3]
        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoModelForImageTextToText,
            AutoProcessor,
        )
        from . import memcheck  # noqa: PLC0415
        from ._hf import load_local_first  # noqa: PLC0415

        # A different size was resident — free it first (one model at a time).
        _cache = None
        prefer_gpu = bool(_cfg().use_gpu) and torch.cuda.is_available()
        # Refuses (raises VisionOOM → fail-open) rather than risking a native
        # OOM segfault when the model won't fit in free VRAM/RAM.
        device = memcheck.pick_device(repo, prefer_gpu=prefer_gpu)
        # local-first: a cached model loads with no network HEAD check (which
        # hangs on a slow/blocked connection); un-cached → downloads.
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            model = load_local_first(
                AutoModelForImageTextToText.from_pretrained, repo,
                dtype=dtype, device_map="cuda", low_cpu_mem_usage=True)
        else:
            model = load_local_first(
                AutoModelForImageTextToText.from_pretrained, repo,
                dtype=torch.float32, low_cpu_mem_usage=True)
        model.eval()
        # Working resolution: SmolVLM tiles an image into 384px patches. Too LOW
        # and a dense screenshot (a LeetCode page, an IDE) is unreadable — the
        # model can't see the small code/UI text and hallucinates. Empirically
        # 1152 reads garbage on a full-screen capture while 2048 reads the code
        # and the selected language correctly (and is no slower for the 500M
        # model). 2048 = ~5x384 caps the token/VRAM cost while keeping fine text
        # legible; a larger image is downscaled to fit, a smaller one is left
        # as-is.
        processor = load_local_first(
            AutoProcessor.from_pretrained, repo, size={"longest_edge": 2048})
        _cache = (repo, model, processor, device)
        log.info("smolvlm loaded: %s on %s", repo, device)
        return model, processor, device


def _pil_images(images: Sequence[bytes]) -> list:
    from PIL import Image  # noqa: PLC0415
    out = []
    for raw in images:
        try:
            out.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception as exc:  # noqa: BLE001 — skip a corrupt image, keep the rest
            log.info("smolvlm: skipping undecodable image (%s)", exc)
    return out


def describe(images: Sequence[bytes], prompt: str, repo: str | None = None) -> str:
    """Parse `images` into a structured text description. Raises on failure."""
    imgs = _pil_images(images)
    if not imgs:
        return ""
    import torch  # noqa: PLC0415

    model, processor, device = _load(repo or _DEFAULT_REPO)
    content = [{"type": "image"} for _ in imgs]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=imgs, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(_cfg().max_new_tokens),
            do_sample=False,  # deterministic transcription, not creative
            # Small VLMs love to fall into a repetition loop under greedy
            # decoding — it burns the whole token budget on garbage AND wrecks
            # latency. no_repeat_ngram_size hard-stops any repeated 3-gram so the
            # model emits EOS once it has read everything; the penalty nudges it
            # off loops earlier.
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
        )
    trimmed = generated[:, inputs.input_ids.shape[1]:]
    out = processor.batch_decode(
        trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)
    return (out[0] if out else "").strip()


def unload() -> None:
    global _cache
    with _lock:
        _cache = None
    _free_cuda()


def _free_cuda() -> None:
    """Release the freed weights' VRAM immediately so the next model fits —
    setting _cache=None only drops the Python ref; the CUDA allocator keeps the
    blocks until empty_cache()."""
    try:
        import gc  # noqa: PLC0415
        gc.collect()
        import torch  # noqa: PLC0415
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def is_loaded() -> bool:
    return _cache is not None


def is_loaded_repo(repo: str) -> bool:
    """True only when THIS specific size is the one currently resident — keeps
    the resident-engines readout honest across the two SmolVLM providers."""
    return _cache is not None and _cache[0] == repo
