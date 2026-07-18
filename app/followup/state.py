"""Conversation_State store (followup-context-engine R1).

``ConversationState`` extends the existing ``GoalLedger`` record — it lives in
the SAME ``User.preferences`` JSONB under ``clarify_ledger[conversation_id]`` so
one ``save_store(...)`` persists slots + outcomes + prefs + this state together
(no schema migration). It widens the ledger from slots+constraints to:

  goal / goal_initial / goal_complete, decisions (supersedable), constraints
  (positive + negative), preferences, entities (Entity_Registry, bounded LRU),
  open_questions, assumptions, enumerations (last options list).

Design contract:
  • Reuses ``GoalLedger.observe`` for slot extraction so the ledger and pre-gate
    stay in agreement — this class never duplicates that logic.
  • Mutates the shared root in place; the caller persists via ``save_store``.
  • Missing record → behaves as a fresh conversation (R1.3).
  • Bounded collections with oldest/LRU eviction (R1.5/R4.3).
  • Any error path reduces to a no-op so the turn proceeds (R1.4).
"""
from __future__ import annotations

from typing import Any

from app.clarify.goal_ledger import GoalLedger
from app.clarify.intent_pipeline import extract_slots


def _cfg_bounds() -> dict:
    """Read the configured collection caps (fail-safe defaults)."""
    try:
        from app.core.config_loader import cfg
        f = cfg.followup
        return {
            "decisions": int(getattr(f, "max_decisions", 32)),
            "constraints": int(getattr(f, "max_constraints", 32)),
            "entities": int(getattr(f, "max_entities", 64)),
            "open_questions": int(getattr(f, "max_open_questions", 16)),
            "enumerations": int(getattr(f, "max_enumerations", 12)),
        }
    except Exception:  # noqa: BLE001
        return {"decisions": 32, "constraints": 32, "entities": 64,
                "open_questions": 16, "enumerations": 12}


# Widened keys layered on top of the GoalLedger record. Absent keys default
# empty so legacy `clarify_ledger` records still load (R1.2 back-compat).
_WIDE_DEFAULTS: dict[str, Any] = {
    "goal": None,
    "goal_initial": None,
    "goal_complete": False,
    "decisions": {},          # key -> chosen value (supersedable)
    "fu_constraints": [],     # [{text, negative}] (distinct from ledger slots' constraints)
    "preferences": {},        # concise/detailed, example-language, ...
    "entities": [],           # Entity_Registry (LRU; most-recent last)
    "open_questions": [],
    "assumptions": [],
    "enumerations": [],       # last enumerated options list
}


class ConversationState:
    """Per-conversation structured state extending ``GoalLedger``. Mutates the
    shared preferences root in place; the caller persists it."""

    def __init__(self, prefs: dict | None, conversation_id: str | None):
        # GoalLedger ensures root["clarify_ledger"][cid] exists + seeds slots.
        self._ledger = GoalLedger(prefs, conversation_id)
        self.root: dict = self._ledger.root
        self._cid = conversation_id
        self._bounds = _cfg_bounds()
        if conversation_id:
            led = self.root["clarify_ledger"].get(conversation_id)
            if not isinstance(led, dict):
                led = {}
                self.root["clarify_ledger"][conversation_id] = led
        else:
            # No conversation id → ephemeral record (not persisted), fresh.
            led = {}
        # Backfill widened keys onto the (possibly legacy) record.
        for k, v in _WIDE_DEFAULTS.items():
            if k not in led:
                led[k] = type(v)() if v is not None and not isinstance(v, bool) else v
        self._led = led

    # ---- reads -----------------------------------------------------------
    def goal(self) -> str | None:
        return self._led.get("goal")

    def goal_initial(self) -> str | None:
        return self._led.get("goal_initial")

    def is_goal_complete(self) -> bool:
        return bool(self._led.get("goal_complete"))

    def decisions(self) -> dict[str, str]:
        d = self._led.get("decisions")
        return dict(d) if isinstance(d, dict) else {}

    def constraints(self) -> list[dict]:
        """Positive + negative constraints recorded by the engine."""
        c = self._led.get("fu_constraints")
        return list(c) if isinstance(c, list) else []

    def preferences(self) -> dict[str, str]:
        p = self._led.get("preferences")
        return dict(p) if isinstance(p, dict) else {}

    def entities(self) -> list[str]:
        e = self._led.get("entities")
        return list(e) if isinstance(e, list) else []

    def open_questions(self) -> list[str]:
        q = self._led.get("open_questions")
        return list(q) if isinstance(q, list) else []

    def assumptions(self) -> list[str]:
        a = self._led.get("assumptions")
        return list(a) if isinstance(a, list) else []

    def enumerations(self) -> list[str]:
        """The most recent enumerated options (for selection references)."""
        e = self._led.get("enumerations")
        return list(e) if isinstance(e, list) else []

    def current_artifact(self) -> dict | None:
        """The artifact this conversation is actively working on (workspace-and-
        artifacts R7) — exposed so a follow-up ("add X to it") targets it."""
        a = self._led.get("current_artifact")
        return dict(a) if isinstance(a, dict) else None

    def confirmed_slots(self) -> dict:
        """The GoalLedger slots (language/framework/platform) — still suppressed
        by the clarifier so they're never re-asked (R1.2)."""
        return self._ledger.confirmed_slots()

    def summary(self) -> str:
        """A compact, prompt-ready summary of the state (always included in
        follow-up context per R8.2). Empty when nothing has accumulated."""
        parts: list[str] = []
        goal = self.goal()
        if goal:
            done = " (completed)" if self.is_goal_complete() else ""
            parts.append(f"Goal: {goal}{done}")
        slots = {k: v for k, v in self.confirmed_slots().items() if v}
        if slots:
            parts.append("Decided: " + ", ".join(f"{k}={v}" for k, v in slots.items()))
        dec = self.decisions()
        if dec:
            parts.append("Decisions: " + ", ".join(f"{k}={v}" for k, v in dec.items()))
        cons = self.constraints()
        if cons:
            rendered = [("no " + c["text"]) if c.get("negative") else c["text"]
                        for c in cons]
            parts.append("Constraints: " + ", ".join(rendered))
        prefs = self.preferences()
        if prefs:
            parts.append("Preferences: " + ", ".join(f"{k}={v}" for k, v in prefs.items()))
        ents = self.entities()
        if ents:
            parts.append("Entities: " + ", ".join(ents[-8:]))
        oqs = self.open_questions()
        if oqs:
            parts.append("Open questions: " + "; ".join(oqs))
        return "\n".join(parts)

    # ---- writes (mutate the shared root; caller persists) ----------------
    def observe(self, text: str, recent: str = "") -> None:
        """Record what this turn establishes. Delegates slot extraction to the
        GoalLedger (no duplication), then widens with entities + a first goal."""
        if not self._cid:
            return
        try:
            self._ledger.observe(text, recent)
            self._register_entities_from(text)
            # Seed the goal from the first substantive turn (kept until shifted).
            if not self._led.get("goal"):
                g = self._derive_goal(text)
                if g:
                    self._led["goal"] = g
                    self._led["goal_initial"] = g
        except Exception:  # noqa: BLE001 — never break a turn (R1.4)
            pass

    def reset_topic(self) -> None:
        """Drop the topic-scoped narrative state on an explicit topic shift so a
        new subject starts clean: goal, entities, decisions, constraints,
        enumerations, and open questions are cleared. The GoalLedger's CONFIRMED
        slots (language/framework/platform) are deliberately KEPT so the user is
        not re-asked their standing preferences across a switch. Fail-open."""
        if not self._cid:
            return
        try:
            for k in ("goal", "goal_initial", "goal_complete", "entities",
                      "decisions", "fu_constraints", "enumerations",
                      "open_questions", "assumptions", "current_artifact"):
                if k in self._led:
                    v = self._led[k]
                    if isinstance(v, list):
                        self._led[k] = []
                    elif isinstance(v, dict):
                        self._led[k] = {}
                    elif isinstance(v, bool):
                        self._led[k] = False
                    else:
                        self._led[k] = None
        except Exception:  # noqa: BLE001 — never break a turn
            pass

    def set_goal(self, goal: str, complete: bool = False) -> None:
        goal = (goal or "").strip()
        if not goal:
            return
        if not self._led.get("goal_initial"):
            self._led["goal_initial"] = goal
        self._led["goal"] = goal
        self._led["goal_complete"] = bool(complete)

    def mark_goal_complete(self, complete: bool = True) -> None:
        self._led["goal_complete"] = bool(complete)

    def set_decision(self, key: str, value: str) -> None:
        """Record/supersede a decision (R6.1) — the latest value wins."""
        key = (key or "").strip().lower()
        value = (value or "").strip()
        if not key or not value:
            return
        d = self._led.setdefault("decisions", {})
        d[key] = value
        self._bound_dict(d, self._bounds["decisions"])

    def remove_decision(self, key: str) -> None:
        """Reverse a decision (R6.2)."""
        key = (key or "").strip().lower()
        d = self._led.get("decisions")
        if isinstance(d, dict):
            d.pop(key, None)

    def add_constraint(self, text: str, negative: bool = False) -> None:
        """Record a positive or negative constraint (R6.3)."""
        text = (text or "").strip()
        if not text:
            return
        c = self._led.setdefault("fu_constraints", [])
        # De-dupe on (text, negative).
        for item in c:
            if item.get("text") == text and bool(item.get("negative")) == bool(negative):
                return
        c.append({"text": text, "negative": bool(negative)})
        self._bound_list(c, self._bounds["constraints"])

    def set_preference(self, key: str, value: str) -> None:
        key = (key or "").strip().lower()
        value = (value or "").strip()
        if key and value:
            self._led.setdefault("preferences", {})[key] = value

    def add_entity(self, name: str) -> None:
        """Register a named entity (LRU: re-adding moves it to most-recent)."""
        name = (name or "").strip()
        if not name:
            return
        e = self._led.setdefault("entities", [])
        if name in e:
            e.remove(name)
        e.append(name)
        self._bound_list(e, self._bounds["entities"])

    def add_open_question(self, q: str) -> None:
        q = (q or "").strip()
        if not q:
            return
        oq = self._led.setdefault("open_questions", [])
        if q not in oq:
            oq.append(q)
            self._bound_list(oq, self._bounds["open_questions"])

    def clear_open_question(self, q: str) -> None:
        q = (q or "").strip()
        oq = self._led.get("open_questions")
        if isinstance(oq, list) and q in oq:
            oq.remove(q)

    def set_enumerations(self, options: list[str]) -> None:
        """Replace the last enumerated options list (for selection refs, R3.2)."""
        opts = [str(o).strip() for o in (options or []) if str(o).strip()]
        if not opts:
            return
        self._led["enumerations"] = opts[: self._bounds["enumerations"]]

    def set_current_artifact(self, artifact_id: str, title: str | None = None,
                             kind: str | None = None) -> None:
        """Mark the conversation's Current_Artifact + register it (and a generic
        kind word like "diagram"/"document") as resolvable entities so a
        follow-up reference binds to it (workspace-and-artifacts R7.1)."""
        if not artifact_id:
            return
        self._led["current_artifact"] = {
            "id": artifact_id,
            "title": (title or "").strip(),
            "kind": (kind or "").strip(),
        }
        if title:
            self.add_entity(title.strip())
        if kind:
            self.add_entity(kind.strip())

    def add_assumption(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        a = self._led.setdefault("assumptions", [])
        if text not in a:
            a.append(text)
            self._bound_list(a, self._bounds["open_questions"])

    # ---- helpers ---------------------------------------------------------
    def _register_entities_from(self, text: str) -> None:
        """Pull named technologies/frameworks/platforms from a turn into the
        Entity_Registry, reusing the deterministic slot extractor."""
        slots = extract_slots(text or "", "")
        for key in ("language", "framework", "platform"):
            v = slots.get(key)
            if v:
                self.add_entity(str(v))

    @staticmethod
    def _derive_goal(text: str) -> str | None:
        """A coarse goal label from the first substantive turn: the first
        sentence, trimmed. Deterministic, no LLM."""
        t = " ".join((text or "").split())
        if len(t) < 8:
            return None
        # First sentence / clause, capped.
        for sep in (". ", "? ", "\n"):
            i = t.find(sep)
            if 0 < i < 140:
                t = t[:i]
                break
        return t[:140]

    @staticmethod
    def _bound_list(lst: list, cap: int) -> None:
        cap = max(1, int(cap))
        while len(lst) > cap:
            lst.pop(0)  # oldest-first eviction

    @staticmethod
    def _bound_dict(d: dict, cap: int) -> None:
        cap = max(1, int(cap))
        while len(d) > cap:
            # Dicts preserve insertion order → drop the oldest key.
            oldest = next(iter(d))
            d.pop(oldest, None)


__all__ = ["ConversationState"]
