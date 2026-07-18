"""Cross-turn goal ledger + conversation state (advanced-intent-reasoning R6).

A multi-turn project accumulates decisions (language, framework, platform,
constraints) that must never be re-asked, and the *stage* of the work changes
how readily the assistant should clarify. This module:

  • accumulates confirmed slots per conversation (fed into the pre-gate's
    suppression set so a decided slot is never asked again), and
  • classifies the conversation state (discovery | planning | execution |
    review) and exposes a per-state clarification threshold (more permissive in
    discovery, stricter in execution).

It stores its state in the shared `User.preferences` JSONB root under
`clarify_ledger[conversation_id]`, so a single `save_store(...)` persists it
alongside preferences + outcomes. Slot extraction reuses the deterministic
`intent_pipeline.extract_slots`, so the ledger and pre-gate agree.
"""
from __future__ import annotations

from .intent_pipeline import extract_slots

# Conversation states.
DISCOVERY = "discovery"
PLANNING = "planning"
EXECUTION = "execution"
REVIEW = "review"

# Per-state answer band: in discovery we tolerate more questions (lower bar to
# answer), in execution we strongly prefer to keep moving (higher bar to ask =
# lower answer band). These are the "answer directly at/above" thresholds.
_STATE_BAND = {
    DISCOVERY: 0.92,   # ask a bit more freely early on
    PLANNING: 0.88,
    EXECUTION: 0.80,   # mid-build: avoid interrupting, answer more
    REVIEW: 0.85,
}
# The default band (unknown state) now comes from cfg.confidence.state_band_default
# — see threshold_for().

_SLOT_KEYS = ("language", "framework", "platform")

# Cues that the conversation has entered a given stage (checked most-specific
# first by the caller's recent text).
_PLANNING_CUES = ("plan", "design", "architecture", "approach", "structure",
                  "schema", "outline", "how should")
_EXECUTION_CUES = ("build", "create", "implement", "write the", "generate",
                  "add ", "code ", "make the", "scaffold", "now ")
_REVIEW_CUES = ("review", "fix", "bug", "error", "refactor", "optimize",
               "improve", "test", "doesn't work", "not working", "debug")


def _empty_ledger() -> dict:
    return {"slots": {}, "constraints": [], "assumptions": [],
            "provisional_slots": {}}


# Cues that the user's next turn OBJECTS to what was just assumed ("no, use
# Java", "actually make it Go", "not Python"). An objection clears provisional
# assumptions instead of promoting them.
_OBJECTION_CUES = ("no,", "no ", "not ", "actually", "instead", "don't",
                   "dont ", "wrong", "rather ", "change it", "switch to")


class GoalLedger:
    """Conversation-scoped confirmed-slot accumulation + state classification.
    Mutates the shared preferences root in place; the caller persists it."""

    def __init__(self, prefs: dict | None, conversation_id: str | None):
        self.root: dict = prefs if isinstance(prefs, dict) else {}
        self._cid = conversation_id
        all_ledgers = self.root.get("clarify_ledger")
        if not isinstance(all_ledgers, dict):
            all_ledgers = {}
            self.root["clarify_ledger"] = all_ledgers
        self._all = all_ledgers
        if conversation_id:
            led = all_ledgers.get(conversation_id)
            if not isinstance(led, dict):
                led = _empty_ledger()
                all_ledgers[conversation_id] = led
            for k, v in _empty_ledger().items():
                led.setdefault(k, v)
            self._led = led
        else:
            self._led = _empty_ledger()

    # ---- writes ----------------------------------------------------------
    def observe(self, text: str, recent: str = "") -> None:
        """Extract any newly-named slots from this turn and persist them so they
        are never re-asked (R6.1). Also settles any PROVISIONAL assumptions from
        the previous assume-mode answer (Phase-1 assumption persistence): an
        objection in this turn clears them; anything else promotes them to
        confirmed slots — silence is acceptance, matching how the assumption
        was presented ("proceeding with X — say the word to change it")."""
        if not self._cid:
            return
        self._settle_assumptions(text or "")
        slots = extract_slots(text or "", recent or "")
        for k in _SLOT_KEYS:
            v = slots.get(k)
            if v and not self._led["slots"].get(k):
                self._led["slots"][k] = v
        for c in slots.get("constraints", []) or []:
            if c not in self._led["constraints"]:
                self._led["constraints"].append(c)

    def record_assumptions(self, assumptions: list[str] | None) -> None:
        """Persist the assumptions an assume-mode answer just stated (Phase 1).

        Each assumption text is kept verbatim (for suppression context + the
        next-turn settle), and any slot value it names (extract_slots on the
        assumption text — "Assuming Python" → language=python) is stored as a
        PROVISIONAL slot: suppressed like a confirmed slot so the very next
        turn never re-asks it, but only promoted to confirmed once the user's
        next message doesn't object.
        """
        if not self._cid or not assumptions:
            return
        self._led.setdefault("assumptions", [])
        self._led.setdefault("provisional_slots", {})
        for a in assumptions:
            a = str(a or "").strip()
            if not a:
                continue
            if a not in [x.get("text") for x in self._led["assumptions"]
                         if isinstance(x, dict)]:
                self._led["assumptions"].append(
                    {"text": a, "status": "provisional"})
            try:
                s = extract_slots(a, "")
                for k in _SLOT_KEYS:
                    v = s.get(k)
                    if v and not self._led["slots"].get(k):
                        self._led["provisional_slots"][k] = v
            except Exception:  # noqa: BLE001 — verbatim text is still recorded
                pass

    def _settle_assumptions(self, user_text: str) -> None:
        """Promote or clear provisional assumptions based on the user's next
        turn: objection → clear (so the corrected value re-extracts normally);
        anything else → promote to confirmed slots + mark accepted."""
        prov = self._led.get("provisional_slots") or {}
        pending = [x for x in (self._led.get("assumptions") or [])
                   if isinstance(x, dict) and x.get("status") == "provisional"]
        if not prov and not pending:
            return
        low = (user_text or "").lower()
        objected = any(c in low for c in _OBJECTION_CUES)
        if objected:
            self._led["provisional_slots"] = {}
            for x in pending:
                x["status"] = "rejected"
            return
        for k, v in prov.items():
            if v and not self._led["slots"].get(k):
                self._led["slots"][k] = v
        self._led["provisional_slots"] = {}
        for x in pending:
            x["status"] = "accepted"

    def record_choice(self, key: str, value: str) -> None:
        """Persist an explicitly-answered choice (e.g. from a clarification)."""
        if not self._cid:
            return
        key = (key or "").strip().lower()
        value = (value or "").strip()
        if key and value:
            self._led["slots"][key] = value

    # ---- reads -----------------------------------------------------------
    def confirmed_slots(self) -> dict:
        """The slots already decided this conversation (R6.2 suppression).
        Includes PROVISIONAL (assumed-last-turn) slots so an assumption that
        was just stated is never re-asked while it awaits settlement."""
        merged = dict(self._led["slots"])
        for k, v in (self._led.get("provisional_slots") or {}).items():
            merged.setdefault(k, v)
        return merged

    def assumptions(self, status: str | None = None) -> list[dict]:
        """Recorded assumptions, optionally filtered by status
        (provisional | accepted | rejected)."""
        rows = [x for x in (self._led.get("assumptions") or [])
                if isinstance(x, dict)]
        return [x for x in rows if status is None or x.get("status") == status]

    def has_tech(self) -> bool:
        s = self._led["slots"]
        return bool(s.get("language") or s.get("framework"))


def classify_state(history: str, current: str = "") -> str:
    """Classify the conversation stage from the recent transcript (R6.3).

    Most-specific stage wins: review (fixing/optimizing existing work) →
    execution (building now) → planning (designing) → discovery (default early).
    """
    blob = f"{history}\n{current}".lower()
    if any(c in blob for c in _REVIEW_CUES):
        return REVIEW
    if any(c in blob for c in _EXECUTION_CUES):
        return EXECUTION
    if any(c in blob for c in _PLANNING_CUES):
        return PLANNING
    return DISCOVERY


def threshold_for(state: str) -> float:
    """Per-state answer band (R6.3); unknown state → the default band (from
    central config `cfg.confidence.state_band_default`, default 0.90)."""
    from app.core.config_loader import cfg
    return _STATE_BAND.get(state, cfg.confidence.state_band_default)


__all__ = ["GoalLedger", "classify_state", "threshold_for",
           "DISCOVERY", "PLANNING", "EXECUTION", "REVIEW"]
