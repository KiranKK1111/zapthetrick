"""Constrained decoding wiring (roadmap Phase 6 #24).

`llm/constrained.py` is a complete, tested JSON-Schema validator + OpenAI-style
``response_format`` builder — but it had **no callers**, so JSON-producing paths
still parsed free-form model text and could corrupt the envelope. This module is
the wiring surface that lives in `response_arch` (the `llm/` package is owned by
another agent): it exposes small, fail-open helpers that a generation path calls
to (a) *request* schema-enforced output where the model supports it, and (b)
*validate + repair* the returned JSON against the schema for EVERY provider.

`generate_structured()` runs the full loop against the real LLM client (request
→ validate → one schema-guided repair round) and is the intended entry point for
any code path that needs a structured object out of the model. It imports the
LLM client lazily so importing this module has no heavy dependencies and stays
testable in isolation.

See :func:`wiring_note` for the one hook another agent must add in the engine.
"""
from __future__ import annotations

import logging
from typing import Any

from app.llm import constrained as _C

log = logging.getLogger(__name__)


def structured_options(schema: dict, *, model_meta: dict | None = None,
                       name: str = "response", strict: bool = True) -> dict:
    """Options to merge into an LLM call so a capable model decodes to ``schema``.

    Returns ``{"response_format": {...}}`` when the model advertises structured
    support, else ``{}`` (the validate-and-repair path still enforces the schema
    for every provider). Never raises.
    """
    try:
        if _C.supports_structured(model_meta):
            return {"response_format": _C.response_format(
                schema, name=name, strict=strict)}
    except Exception:  # noqa: BLE001
        pass
    return {}


def enforce(text: str, schema: dict) -> tuple[Any | None, list[str]]:
    """Parse+validate model ``text`` against ``schema`` → ``(obj|None, errors)``."""
    try:
        return _C.coerce(text, schema)
    except Exception as exc:  # noqa: BLE001
        return None, [f"enforce failed: {exc}"]


def repair_prompt(schema: dict, bad_text: str, errors: list[str]) -> str:
    """A terse instruction asking the model to re-emit valid JSON for ``schema``."""
    import json
    return (
        "Your previous output did not satisfy the required JSON schema. "
        "Re-emit ONLY a single valid JSON value that conforms — no prose, no "
        "code fence.\n\nSchema:\n" + json.dumps(schema, default=str)[:4000]
        + "\n\nErrors:\n- " + "\n- ".join(errors[:12])
        + "\n\nYour previous output:\n" + (bad_text or "")[:4000]
    )


async def generate_structured(
    messages: list[dict],
    schema: dict,
    *,
    options: dict | None = None,
    model_meta: dict | None = None,
    repair: bool = True,
) -> tuple[Any | None, list[str]]:
    """Request → validate → (one) repair, against the real LLM client.

    Returns ``(obj, errors)``: ``obj`` is the validated object (``None`` if it
    never parsed), ``errors`` the residual schema errors ([] = clean). Fail-open:
    any client error returns ``(None, [reason])`` so the caller falls back to
    free-form parsing rather than crashing the turn.
    """
    try:
        from app.core.llm_client import llm
    except Exception as exc:  # noqa: BLE001
        return None, [f"llm client unavailable: {exc}"]

    opts = dict(options or {})
    opts.update(structured_options(schema, model_meta=model_meta))
    try:
        text, _model = await llm.complete_routed(messages, options=opts)
    except Exception as exc:  # noqa: BLE001
        return None, [f"generation failed: {exc}"]

    obj, errors = enforce(text or "", schema)
    if not errors and obj is not None:
        return obj, []
    if not repair:
        return obj, errors

    # One schema-guided repair round.
    try:
        fix_msgs = list(messages) + [
            {"role": "assistant", "content": text or ""},
            {"role": "user", "content": repair_prompt(schema, text or "", errors)},
        ]
        text2, _m2 = await llm.complete_routed(fix_msgs, options=opts)
        obj2, errors2 = enforce(text2 or "", schema)
        if not errors2 and obj2 is not None:
            return obj2, []
        # Prefer whichever parsed; report the repair's residual errors.
        return (obj2 if obj2 is not None else obj), errors2
    except Exception as exc:  # noqa: BLE001
        log.info("structured repair failed: %s", exc)
        return obj, errors


def wiring_note() -> str:
    """Human-readable note describing the one engine hook another agent adds."""
    return (
        "Engine hook (llm/ owner): in the JSON-producing generation paths "
        "(verify verdict, intent JSON, plan JSON), route through "
        "response_arch.structured.generate_structured(messages, schema) — or at "
        "minimum merge structured_options(schema, model_meta) into the call "
        "options and validate the result with enforce(text, schema). Gate on "
        "getattr(cfg.response_arch, 'constrained_decoding', True)."
    )


__all__ = [
    "structured_options", "enforce", "repair_prompt", "generate_structured",
    "wiring_note",
]
