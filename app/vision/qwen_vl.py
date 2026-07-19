"""Qwen2.5-VL local vision engine (VisionAnalysis.md #1 — top pick).

Best all-round open VLM for UI screenshots, documents, charts, diagrams,
tables, OCR + reasoning. Runs via `transformers`; weights download from
HuggingFace on first use (like the STT engines). GPU-first (bf16 on CUDA) with
transparent CPU fallback.

Single-flight load via an explicit lock + module global (NOT lru_cache, which
is non-atomic — see app/stt/parakeet_stt.py for the same reasoning): concurrent
requests share one resident model; a partial + final pass can't double-load it.
Everything is fail-open: any import/load/inference error raises so the factory
falls through to the next engine in the chain.
"""
from __future__ import annotations

import io
import logging
import threading
from typing import Sequence

log = logging.getLogger(__name__)

# (model, processor, device) once loaded — one instance resident.
_cache: tuple[object, object, str] | None = None
_lock = threading.Lock()


def _cfg():
    from app.core.config_loader import cfg
    return cfg.vision


def _load() -> tuple[object, object, str]:
    """Load Qwen2.5-VL once (double-checked locking). GPU-first, CPU fallback."""
    global _cache
    if _cache is not None:
        return _cache
    with _lock:
        if _cache is not None:
            return _cache
        import torch  # noqa: PLC0415
        from transformers import (  # noqa: PLC0415
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
        )
        from . import memcheck  # noqa: PLC0415
        from ._hf import load_local_first  # noqa: PLC0415

        model_id = _cfg().qwen_vl_model
        prefer_gpu = bool(_cfg().use_gpu) and torch.cuda.is_available()
        # CRITICAL: a model that overruns free VRAM/RAM doesn't raise — it
        # SEGFAULTS the process during the weight copy. The pre-flight refuses an
        # impossible load with a catchable VisionOOM (→ factory fails open) so a
        # too-big pick can never take down the backend.
        device = memcheck.pick_device(model_id, prefer_gpu=prefer_gpu)
        # local-first: cached model loads with no network HEAD check.
        if device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            # Optional 8-bit (bitsandbytes): halves VRAM (~16GB → ~8GB for the
            # 7B) so the high-accuracy 7B fits a 24GB card alongside a resident
            # STT model. Falls back to bf16 when the flag is off or bitsandbytes
            # isn't installed (e.g. the CPU-only desktop build).
            quant = None
            if bool(getattr(_cfg(), "qwen_vl_load_8bit", False)):
                try:
                    from transformers import BitsAndBytesConfig  # noqa: PLC0415
                    import bitsandbytes as _bnb  # noqa: F401,PLC0415
                    quant = BitsAndBytesConfig(load_in_8bit=True)
                except Exception as exc:  # noqa: BLE001
                    log.info("qwen2.5-vl: 8-bit requested but bitsandbytes "
                             "unavailable (%s) — loading bf16", exc)
            if quant is not None:
                model = load_local_first(
                    Qwen2_5_VLForConditionalGeneration.from_pretrained, model_id,
                    quantization_config=quant, device_map="cuda",
                    low_cpu_mem_usage=True)
            else:
                model = load_local_first(
                    Qwen2_5_VLForConditionalGeneration.from_pretrained, model_id,
                    dtype=dtype, device_map="cuda", low_cpu_mem_usage=True)
        else:
            model = load_local_first(
                Qwen2_5_VLForConditionalGeneration.from_pretrained, model_id,
                dtype=torch.float32, low_cpu_mem_usage=True)
        model.eval()
        # Cap the processor's pixel budget so a huge screenshot doesn't blow up
        # latency/VRAM; the parse doesn't need full 4K resolution to read text.
        processor = load_local_first(
            AutoProcessor.from_pretrained, model_id, max_pixels=1280 * 28 * 28)
        _cache = (model, processor, device)
        log.info("qwen2.5-vl loaded: %s on %s", model_id, device)
        return _cache


def _pil_images(images: Sequence[bytes]) -> list:
    from PIL import Image  # noqa: PLC0415
    out = []
    for raw in images:
        try:
            out.append(Image.open(io.BytesIO(raw)).convert("RGB"))
        except Exception as exc:  # noqa: BLE001 — skip a corrupt image, keep the rest
            log.info("qwen2.5-vl: skipping undecodable image (%s)", exc)
    return out


def describe(images: Sequence[bytes], prompt: str) -> str:
    """Parse `images` into a structured text description. Raises on failure."""
    imgs = _pil_images(images)
    if not imgs:
        return ""
    import torch  # noqa: PLC0415

    model, processor, device = _load()
    # One image placeholder per image, then the parse instruction.
    content = [{"type": "image"} for _ in imgs]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    except Exception as exc:  # noqa: BLE001
        # Some transformers/processor versions don't ship a chat template on the
        # Qwen2.5-VL PROCESSOR (only the tokenizer, or none) — build the prompt
        # in Qwen2.5-VL's exact ChatML + vision format by hand. `<|image_pad|>`
        # is expanded to the right number of image tokens by the processor call
        # below when `images=` is passed.
        log.info("qwen2.5-vl: processor has no chat template (%s) — using the "
                 "manual Qwen2.5-VL prompt", exc)
        vision = "".join(
            "<|vision_start|><|image_pad|><|vision_end|>" for _ in imgs)
        text = ("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\n" + vision + prompt +
                "<|im_end|>\n<|im_start|>assistant\n")
    inputs = processor(text=[text], images=imgs, padding=True,
                       return_tensors="pt").to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=int(_cfg().max_new_tokens),
            do_sample=False,  # deterministic transcription, not creative
            repetition_penalty=1.2,   # stop greedy repetition loops
            no_repeat_ngram_size=3,
        )
    # Drop the prompt tokens; decode only the newly generated answer.
    trimmed = generated[:, inputs.input_ids.shape[1]:]
    out = processor.batch_decode(
        trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)
    return (out[0] if out else "").strip()


def unload() -> None:
    """Free the resident model (called on a model switch / unload_all)."""
    global _cache
    with _lock:
        _cache = None
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
