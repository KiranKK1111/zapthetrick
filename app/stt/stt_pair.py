"""STT streaming pair — partials + finalizer reconciliation (vNext §4.1, Stage 6 E).

The live caption wants two things at once: SPEED (a partial on screen <400 ms
after the interviewer speaks) and ACCURACY (the right words, especially domain
jargon). §4.1 gets both by pairing engines:

  * **Parakeet TDT** streams the PARTIAL — fast, good enough to read along;
  * **Whisper large-v3-turbo** re-scores the ENDPOINTED utterance into the
    authoritative FINAL (end-of-speech → final ≤250 ms).

This module owns the two hardware-free pieces of that pairing:
  * **`register_domain(domain)`** feeds the interview vocabulary (resume skills +
    role + JD terms — the `app/live/domain.py` DomainContext) into
    `vocabulary_boost`, which drives Parakeet's keyword-boost API so jargon
    ("kubectl", "Kafka", "gRPC") is transcribed correctly instead of phonetically;
  * **`reconcile_final(partial, final)`** decides the authoritative text once the
    finalizer lands — the Whisper final wins unless it's empty (then the partial
    stands), and it reports the partial↔final agreement so a big disagreement can
    be flagged.

`domain` is duck-typed (`.vocab` / `.role` / `.topics`, or a plain list of terms)
so this stays inside `app.stt` — no `stt → live` edge. Deterministic + fail-open.
Flag-gated (`live.stt_pair`, default OFF).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)

# Boost weight for a domain term (higher than an incidental session term so
# resume/JD jargon reliably wins the keyword bias).
_DOMAIN_WEIGHT = 2.0


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "stt_pair", False))
    except Exception:  # noqa: BLE001
        return False


def _terms_of(domain) -> list[str]:
    """Duck-type the domain into a flat term list — a DomainContext (`.vocab` +
    `.role` + `.topics`), a list/tuple of terms, or a comma/slash string."""
    try:
        if domain is None:
            return []
        if isinstance(domain, (list, tuple, set)):
            return [str(t) for t in domain]
        if isinstance(domain, str):
            import re
            return [t.strip() for t in re.split(r"[,/|;]", domain) if t.strip()]
        out: list[str] = []
        out += [str(t) for t in (getattr(domain, "vocab", None) or [])]
        out += [str(t) for t in (getattr(domain, "topics", None) or [])]
        role = getattr(domain, "role", None)
        if role:
            out.append(str(role))
        return out
    except Exception:  # noqa: BLE001
        return []


def register_domain(domain, *, weight: float = _DOMAIN_WEIGHT) -> int:
    """Feed the interview's domain vocabulary into `vocabulary_boost` (which
    drives the engine keyword-boost). Returns how many terms were registered;
    0 on any error / disabled. Idempotent per term (weights accumulate)."""
    if not enabled():
        return 0
    try:
        from app.stt import vocabulary_boost as _vb
        n = 0
        for term in _terms_of(domain):
            t = term.strip()
            if t and len(t) <= 60:
                _vb.register_term(t, weight)
                n += 1
        return n
    except Exception as exc:  # noqa: BLE001
        log.info("stt_pair.register_domain failed: %s", exc)
        return 0


def boost_terms(limit: int = 120) -> list[str]:
    """The ranked keyword-boost list for Parakeet (domain terms first). '' safe."""
    try:
        from app.stt import vocabulary_boost as _vb
        return _vb.build_boost_list(limit=limit)
    except Exception:  # noqa: BLE001
        return []


@dataclass
class FinalDecision:
    text: str
    source: str        # "final" (Whisper won) | "partial" (finalizer empty)
    agreement: float   # 0..1 partial↔final agreement (1 = identical)
    changed: bool      # the final differs from the partial


def _agreement(a: str, b: str) -> float:
    try:
        from app.stt.arbitrator import _agreement_ratio, _tokenize
        return _agreement_ratio(_tokenize(a), _tokenize(b))
    except Exception:  # noqa: BLE001
        return 1.0 if (a or "").strip() == (b or "").strip() else 0.0


def reconcile_final(partial: str, final: str) -> FinalDecision:
    """Resolve the endpointed utterance: the Whisper FINAL is authoritative and
    wins whenever it has content; if the finalizer produced nothing the Parakeet
    PARTIAL stands (never blank out a caption the user already read). Reports the
    partial↔final agreement so the caller can flag a large re-score. Never raises."""
    try:
        p = (partial or "").strip()
        f = (final or "").strip()
        if not f:
            return FinalDecision(text=p, source="partial", agreement=1.0,
                                 changed=False)
        agree = _agreement(p, f)
        return FinalDecision(text=f, source="final", agreement=round(agree, 3),
                             changed=(f != p))
    except Exception:  # noqa: BLE001
        return FinalDecision(text=(final or partial or "").strip(),
                             source="final", agreement=1.0, changed=False)


__all__ = ["enabled", "register_domain", "boost_terms", "FinalDecision",
           "reconcile_final"]
