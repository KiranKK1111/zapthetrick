"""Repeat-prompt variation engine (vNext §3.9, Stage 7 Component F).

When a user asks the SAME thing again in the SAME conversation, they are not
asking for the cached answer back — they want a DIFFERENT take ("give me another
option"). §3.9 makes that explicit: an immediate same-conversation repeat is a
VARIATION request, so the engine

  * BYPASSES the answer cache (never re-serve the identical text),
  * records each answer's APPROACH in a per-fingerprint ledger,
  * feeds a DIVERGENCE directive naming the approaches already given so the next
    is genuinely different, with widened sampling + canonical-model rotation,
  * tags each sibling with its approach.

This module owns the deterministic decisions — the fingerprint, the repeat/ledger
state, the divergence directive, and the per-repeat sampling knobs. The actual
regeneration (rotated model, higher temperature, per-variant verify) is the
caller's; this tells it WHEN and HOW to diverge. Pure + fail-open. Flag-gated
(`chat.variation_engine`, default OFF → today's cache-serve on a repeat).
"""
from __future__ import annotations

import hashlib
import re
import threading
from dataclasses import dataclass, field

_LOCK = threading.RLock()
_NON_WORD = re.compile(r"[^0-9a-z]+")


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(getattr(cfg, "chat", None), "variation_engine", False))
    except Exception:  # noqa: BLE001
        return False


def fingerprint(text: str) -> str:
    """Stable fingerprint of a prompt — lowercase, collapse non-alphanumerics —
    so "reverse a string" and "Reverse a string!" repeat the same. '' for blank."""
    try:
        norm = _NON_WORD.sub(" ", (text or "").lower()).strip()
        if not norm:
            return ""
        return hashlib.sha256(norm.encode()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- #
# Per-(conversation, fingerprint) approach ledger
# --------------------------------------------------------------------------- #
_ledger: dict[str, list[str]] = {}   # f"{conv}:{fp}" -> [approach, ...]


def _key(conversation_id: str, fp: str) -> str:
    return f"{conversation_id or ''}:{fp}"


def count(conversation_id: str, fp: str) -> int:
    """How many answers this exact prompt has already produced in this chat."""
    with _LOCK:
        return len(_ledger.get(_key(conversation_id, fp), ()))


def approaches(conversation_id: str, fp: str) -> list[str]:
    with _LOCK:
        return list(_ledger.get(_key(conversation_id, fp), ()))


def record(conversation_id: str, fp: str, approach: str = "") -> None:
    """Record that an answer (with `approach`) was produced for this prompt."""
    if not fp:
        return
    with _LOCK:
        _ledger.setdefault(_key(conversation_id, fp), []).append(
            (approach or "").strip() or f"approach {count(conversation_id, fp) + 1}")


def forget_conversation(conversation_id: str) -> None:
    pref = f"{conversation_id or ''}:"
    with _LOCK:
        for k in [k for k in _ledger if k.startswith(pref)]:
            _ledger.pop(k, None)


def reset_for_tests() -> None:
    with _LOCK:
        _ledger.clear()


# --------------------------------------------------------------------------- #
# Decisions
# --------------------------------------------------------------------------- #
def is_repeat(conversation_id: str, text: str) -> bool:
    """True when this exact prompt has ALREADY been answered in this conversation
    (→ the user wants a variation, not the cache). No-op when disabled."""
    if not enabled():
        return False
    fp = fingerprint(text)
    return bool(fp) and count(conversation_id, fp) > 0


def should_bypass_cache(conversation_id: str, text: str) -> bool:
    """A repeat is a variation request → do NOT serve the cache."""
    return is_repeat(conversation_id, text)


@dataclass
class VariationParams:
    temperature: float           # widened sampling, grows with each repeat
    rotate_model: bool           # rotate the canonical model for a different take
    divergence: str              # the directive to prepend
    prior_approaches: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"temperature": round(self.temperature, 2),
                "rotate_model": self.rotate_model,
                "prior_approaches": list(self.prior_approaches)}


def divergence_directive(prior: "list[str]") -> str:
    if not prior:
        return ""
    listed = "; ".join(prior[:6])
    return ("The user asked this again — they want a DIFFERENT take. You have "
            f"already given: {listed}. Take a genuinely different approach (a "
            "different method / structure / angle), not a reworded version of the "
            "same answer.")


def variation_params(conversation_id: str, text: str, *,
                     base_temperature: float = 0.7) -> VariationParams:
    """Sampling + directive for the NEXT variation of this prompt. Temperature
    widens ~0.1 per prior answer (capped), the model rotates from the 2nd repeat
    on, and the divergence directive names the prior approaches. Never raises."""
    try:
        fp = fingerprint(text)
        prior = approaches(conversation_id, fp)
        n = len(prior)
        temp = min(1.1, base_temperature + 0.1 * n)
        return VariationParams(
            temperature=temp, rotate_model=(n >= 1),
            divergence=divergence_directive(prior), prior_approaches=prior)
    except Exception:  # noqa: BLE001
        return VariationParams(temperature=base_temperature, rotate_model=False,
                               divergence="")


__all__ = ["enabled", "fingerprint", "count", "approaches", "record",
           "forget_conversation", "reset_for_tests", "is_repeat",
           "should_bypass_cache", "VariationParams", "divergence_directive",
           "variation_params"]
