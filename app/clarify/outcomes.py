"""Clarification outcome telemetry (advanced-intent-reasoning R1).

Records what happened AFTER each clarification so later turns can learn whether
asking actually helped (the data foundation for confidence calibration, fatigue,
and trust). Like [ClarificationPreferenceStore] this is a pure, synchronous
dict-mutating store over the same `User.preferences` JSONB blob — it mutates a
SEPARATE top-level key so a single `save_store(...)` persists both.

Layout under `preferences["clarify_outcomes"]`:

    ring:    [ {b: <bucket 0..10>, i: <intent>, r: "answered"|"skipped"|
                "overridden"} ]      bounded, most-recent-last
    pending: { conversation_id: {b, i} }   an asked clarification awaiting the
                                            user's next turn to resolve
    counters: { answered, skipped, overridden, recent }  (fatigue/trust, Phase 2)

Only the intent LABEL and a numeric confidence BUCKET are stored — never raw
user text (R1.5). The ring is capped (R1.4).
"""
from __future__ import annotations

_RING_CAP = 200                       # bounded retained outcomes per user (R1.4)
_COUNTER_KEYS = ("answered", "skipped", "overridden", "recent")
_RESPONSES = ("answered", "skipped", "overridden")


def confidence_bucket(confidence: float) -> int:
    """Map a 0..1 confidence to an integer bucket 0..10 (deci-bucketing)."""
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        return 0
    if c < 0.0:
        c = 0.0
    if c > 1.0:
        c = 1.0
    return int(round(c * 10))


def _empty_outcomes() -> dict:
    return {
        "ring": [],
        "pending": {},
        "counters": {k: 0 for k in _COUNTER_KEYS},
    }


class OutcomeStore:
    """Mutates a `preferences` dict in place; the caller persists it (share the
    same root dict as [ClarificationPreferenceStore] to persist in one save)."""

    def __init__(self, prefs: dict | None):
        self.root: dict = prefs if isinstance(prefs, dict) else {}
        o = self.root.get("clarify_outcomes")
        if not isinstance(o, dict):
            o = _empty_outcomes()
            self.root["clarify_outcomes"] = o
        for k, v in _empty_outcomes().items():
            o.setdefault(k, v)
        self._o = o

    # ---- writes ----------------------------------------------------------
    def record_decision(self, conversation_id: str | None, intent: str,
                         confidence: float, asked: bool) -> None:
        """Record a clarification decision. When `asked`, mark it pending for
        the conversation so the user's next turn resolves it (R1.1)."""
        if not asked or not conversation_id:
            return
        self._o["pending"][conversation_id] = {
            "b": confidence_bucket(confidence),
            "i": (intent or "unknown")[:32],
        }

    def record_response(self, conversation_id: str | None, kind: str) -> None:
        """Resolve a pending asked clarification with the user's response
        (answered | skipped | overridden) → append to the ring (R1.2)."""
        if kind not in _RESPONSES or not conversation_id:
            return
        pending = self._o["pending"].pop(conversation_id, None)
        if not isinstance(pending, dict):
            return
        self._o["ring"].append(
            {"b": int(pending.get("b", 0)), "i": pending.get("i", "unknown"),
             "r": kind})
        if len(self._o["ring"]) > _RING_CAP:
            del self._o["ring"][:-_RING_CAP]
        self._o["counters"][kind] = int(self._o["counters"].get(kind, 0)) + 1
        # "recent" tracks fatigue volume; answered/overridden both count as a
        # clarification the user engaged with this window (Phase 2 reads it).
        self._o["counters"]["recent"] = \
            int(self._o["counters"].get("recent", 0)) + 1

    def has_pending(self, conversation_id: str | None) -> bool:
        return bool(conversation_id) and conversation_id in self._o["pending"]

    def decay_recent(self, by: int = 1) -> None:
        """Quiet turn → fatigue recovers (R3.3). Floors at 0."""
        self._o["counters"]["recent"] = \
            max(0, int(self._o["counters"].get("recent", 0)) - by)

    # ---- reads -----------------------------------------------------------
    def counters(self) -> dict:
        return dict(self._o["counters"])

    def calibration_buckets(self) -> dict:
        """Aggregate the ring into per-bucket observed answerability:
            {bucket: {"answerable": n, "needed": n}}
        where a skipped/overridden ask means we COULD have answered
        ("answerable") and an answered ask means the question was "needed"."""
        out: dict[int, dict] = {}
        for e in self._o["ring"]:
            b = int(e.get("b", 0))
            slot = out.setdefault(b, {"answerable": 0, "needed": 0})
            if e.get("r") == "answered":
                slot["needed"] += 1
            else:  # skipped / overridden
                slot["answerable"] += 1
        return out


__all__ = ["OutcomeStore", "confidence_bucket"]
