"""Delivery tracking + true said-state (vNext §4.14, Stage 7 Component I).

The displayed answer is a SCRIPT; what the candidate actually SPOKE is the truth.
An answer shown in full but interrupted after two sentences was NOT fully said;
a candidate who improvised a claim the script never contained DID say it. §4.14
tracks that reality by fuzzy-aligning the spoken transcript against the displayed
answer:

  * a **delivery cursor** — how far through the displayed answer the candidate
    has actually read (drives the teleprompter cursor + interruption handling);
  * **improvised** additions — spoken runs that don't match the script (the
    candidate went off-book) — these enter the said-state;
  * the **said-state** = the DELIVERED text (the matched prefix) + the improvised
    additions — NOT the whole script. So the claims ledger (§4.8) only counts
    what was truly said, and "as I mentioned…" can't reference an unspoken tail.

Deterministic difflib alignment (no model) + fail-open (an alignment error →
"nothing delivered yet", the safe under-count). Flag-gated
(`live.delivery_tracking`, default OFF → today's displayed-is-said assumption).
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

_WORD = re.compile(r"\w+")
# An improvised run this many words or longer is treated as a real claim.
_MIN_IMPROV_WORDS = 4


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "delivery_tracking", False))
    except Exception:  # noqa: BLE001
        return False


@dataclass
class DeliveryState:
    displayed_words: int
    delivered_words: int              # displayed words the candidate has spoken
    delivered_ratio: float            # 0..1 through the script
    cursor: int                       # word index into the displayed answer
    delivered_text: str               # the delivered prefix of the script
    improvised: list[str] = field(default_factory=list)  # off-script spoken runs
    completed: bool = False           # ~fully delivered

    def as_dict(self) -> dict:
        return {"displayed_words": self.displayed_words,
                "delivered_words": self.delivered_words,
                "delivered_ratio": round(self.delivered_ratio, 3),
                "cursor": self.cursor, "completed": self.completed,
                "improvised": list(self.improvised)}


def _tokens_with_spans(text: str) -> "tuple[list[str], list[tuple[int,int]]]":
    toks, spans = [], []
    for m in _WORD.finditer(text or ""):
        toks.append(m.group(0).lower())
        spans.append((m.start(), m.end()))
    return toks, spans


def align_delivery(displayed: str, spoken: str, *,
                   complete_ratio: float = 0.9) -> DeliveryState:
    """Align the `spoken` transcript against the `displayed` script. Returns the
    delivery cursor (how far into the script the candidate read), the delivered
    prefix, and any improvised (off-script) spoken runs. Never raises → an empty
    delivery on error (the safe under-count)."""
    try:
        d_toks, d_spans = _tokens_with_spans(displayed or "")
        s_toks, _ = _tokens_with_spans(spoken or "")
        n = len(d_toks)
        if n == 0:
            return DeliveryState(0, 0, 0.0, 0, "", [], False)
        if not s_toks:
            return DeliveryState(n, 0, 0.0, 0, "", [], False)

        sm = difflib.SequenceMatcher(a=s_toks, b=d_toks, autojunk=False)
        matched = 0
        cursor = 0                    # furthest displayed index reached
        improvised: list[str] = []
        last_s_end = 0
        for block in sm.get_matching_blocks():
            # Spoken tokens BEFORE this match, not part of it → improvised run.
            if block.a > last_s_end and block.size:
                run = s_toks[last_s_end:block.a]
                if len(run) >= _MIN_IMPROV_WORDS:
                    improvised.append(" ".join(run))
            if block.size:
                matched += block.size
                cursor = max(cursor, block.b + block.size)
                last_s_end = block.a + block.size
        # A trailing improvised run after the last match.
        if last_s_end < len(s_toks):
            run = s_toks[last_s_end:]
            if len(run) >= _MIN_IMPROV_WORDS:
                improvised.append(" ".join(run))

        ratio = matched / n
        delivered_text = displayed[:d_spans[cursor - 1][1]] if cursor else ""
        return DeliveryState(
            displayed_words=n, delivered_words=matched, delivered_ratio=ratio,
            cursor=cursor, delivered_text=delivered_text, improvised=improvised,
            completed=ratio >= complete_ratio)
    except Exception:  # noqa: BLE001
        return DeliveryState(0, 0, 0.0, 0, "", [], False)


def said_text(state: DeliveryState) -> str:
    """The true said-state text = what was DELIVERED from the script + anything
    improvised. NOT the unspoken tail of an interrupted answer."""
    parts = [state.delivered_text.strip()] + [i.strip() for i in state.improvised]
    return " ".join(p for p in parts if p).strip()


def improvised_claims(state: DeliveryState) -> list[str]:
    """The off-script spoken runs long enough to count as claims → the said-state
    additions that enter the envelope (§4.8 claims ledger)."""
    return [i for i in state.improvised if len(_WORD.findall(i)) >= _MIN_IMPROV_WORDS]


def record_delivery(session_id: str, displayed: str, spoken: str) -> DeliveryState:
    """Align + feed the DELIVERED reality into the §4.8 claims ledger (only what
    was actually said enters the said-state). No-op when disabled. Never raises."""
    state = align_delivery(displayed, spoken)
    if not enabled():
        return state
    try:
        from app.live import canonical as _c
        # The delivered script text (if substantially delivered) + improvised runs
        # are what the candidate has genuinely "said".
        if state.delivered_ratio >= 0.5 and state.delivered_text.strip():
            _c.record_claim(session_id, state.delivered_text)
        for claim in improvised_claims(state):
            _c.record_claim(session_id, claim)
    except Exception:  # noqa: BLE001
        pass
    return state


__all__ = ["enabled", "DeliveryState", "align_delivery", "said_text",
           "improvised_claims", "record_delivery"]
