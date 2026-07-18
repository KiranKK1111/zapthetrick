"""LLM intent disambiguation for low-confidence turns (gap G4).

When the embedder is genuinely uncertain about intent (cosine below the semantic
threshold), the *Claude-move* is to ask the model — not to fall back to keyword
regex. This is that async call: one cheap classifier round-trip that picks a
single intent from the taxonomy. It runs ONLY on the gray-zone turns (semantic
confidence below the primary threshold) and only when enabled, so it adds no
latency to the confident-majority path.

Gated by `semantic_intent.llm_disambiguation` (default off). Injectable
`complete_fn` for tests. Fail-open: any error returns None and the caller keeps
the semantic best-guess (or the regex net).
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_INTENTS = [
    "chitchat", "knowledge", "comparison", "debugging", "test_generation",
    "documentation", "design", "code_generation", "project_build", "archive",
]

_PROMPT = (
    "Classify the user's message into EXACTLY ONE of these intents and reply "
    "with only the label (no punctuation, no explanation):\n"
    + ", ".join(_INTENTS)
    + "\n\nMessage:\n{text}"
)


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.semantic_intent, "llm_disambiguation", False))
    except Exception:  # noqa: BLE001
        return False


def _match(raw: str) -> str | None:
    """Map a model reply to a valid intent label (tolerant of extra text)."""
    low = (raw or "").strip().lower()
    if not low:
        return None
    for label in _INTENTS:               # exact-ish first
        if low == label or low.startswith(label):
            return label
    for label in _INTENTS:               # else the first label mentioned
        if label in low:
            return label
    return None


async def _default_complete(text: str) -> str:
    from app.core.llm_client import llm
    return await llm.complete(
        [{"role": "user", "content": _PROMPT.format(text=text[:2000])}],
        options={"difficulty": "trivial", "temperature": 0.0})


async def disambiguate_intent(text: str, *, complete_fn=None) -> str | None:
    """One classifier call → a single intent label, or None on failure/no match.
    Never raises."""
    try:
        if not (text or "").strip():
            return None
        raw = await (complete_fn or _default_complete)(text)
        return _match(raw)
    except Exception as exc:  # noqa: BLE001
        log.info("intent disambiguation failed: %s", exc)
        return None


__all__ = ["disambiguate_intent", "enabled"]
