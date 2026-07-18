"""State updater + continuation planning (followup-context-engine R6/R7/R9/R10).

Applies a turn's effects to the ``ConversationState``:
  • corrections / contradictions supersede a Decision (R6.1),
  • reversals remove it (R6.2),
  • negative constraints are recorded (R6.3),
  • ``commit(turn, answer)`` registers new entities + enumerated options from the
    answer and clears an Open_Question the user just answered (R9.3),
  • goal shifts / completions are tracked (R10).

Plus ``continuation_directive`` (R7) for resuming an incomplete / enumerated /
complete prior answer without repetition.

All deterministic; every entry point is wrapped so an error is a no-op and the
turn proceeds (Property 1).
"""
from __future__ import annotations

import re

from app.followup import acts as A
from app.clarify.intent_pipeline import extract_slots

# "use X instead", "change it to X", "actually X", "should be X".
_DECISION_VERB_RE = re.compile(
    r"\b(?:use|switch to|change (?:it )?to|make it|go with|should be|"
    r"actually(?: use)?)\s+(.+)$", re.IGNORECASE)
# Reversal / removal cues.
_REVERSAL_RE = re.compile(
    r"\b(?:undo|revert|never ?mind|scratch that|forget (?:that|the)|"
    r"no longer|remove|drop)\b", re.IGNORECASE)
# Negative constraint: "don't use X", "no X", "without X", "avoid X".
_NEGATIVE_RE = re.compile(
    r"\b(?:don'?t use|do not use|no|without|avoid|never use)\s+([A-Za-z0-9][\w .+\-]{1,40})",
    re.IGNORECASE)

# Enumerated-options capture from an answer (markdown list / numbered list).
_NUM_ITEM_RE = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+(.+?)\s*$")


_COMPLETION_RE = re.compile(
    r"\b(?:that'?s all|we'?re done|all done|looks? complete|finished|"
    r"that completes|nothing else|good to go|ship it)\b", re.IGNORECASE)


def apply_turn(turn: str, act: str, resolution, state) -> None:
    """Mutate `state` for this turn's act. No-op on error."""
    try:
        _apply_turn(turn, act, resolution, state)
    except Exception:  # noqa: BLE001
        pass


def _apply_turn(turn: str, act: str, resolution, state) -> None:
    t = (turn or "").strip()
    if not t:
        return

    # Goal completion cue — mark the current goal complete so a later
    # continuation advances rather than re-doing it (R10.2).
    if _COMPLETION_RE.search(t):
        state.mark_goal_complete(True)

    if act == A.CORRECTION:
        # Reversal first (it removes), else supersede a decision.
        if _REVERSAL_RE.search(t):
            _remove_for(turn, resolution, state)
            return
        _supersede_decision(turn, state)
        # A correction can also state a negative constraint ("no longer use X").
        _record_negatives(turn, state)
        return

    if act == A.REJECTION:
        # The user rejected the last direction — record nothing positive; a
        # negative constraint may still be present.
        _record_negatives(turn, state)
        return

    # A genuinely new topic mid-conversation is a goal shift: adopt the new
    # current goal while RETAINING the initial one for reference (R10.1).
    if act == A.NEW_TOPIC:
        try:
            existing = state.goal()
        except Exception:  # noqa: BLE001
            existing = None
        if existing and len(t.split()) >= 4:
            new_goal = ConversationState_derive(t)
            if new_goal and new_goal.lower() != (existing or "").lower():
                state.set_goal(new_goal)

    # Any turn may introduce a negative constraint ("don't use Firebase").
    _record_negatives(turn, state)


def ConversationState_derive(text: str) -> str | None:
    """Coarse goal label from a turn (mirror of ConversationState._derive_goal)
    without importing private state internals."""
    t = " ".join((text or "").split())
    if len(t) < 8:
        return None
    for sep in (". ", "? ", "\n"):
        i = t.find(sep)
        if 0 < i < 140:
            t = t[:i]
            break
    return t[:140]


def _supersede_decision(turn: str, state) -> None:
    """Record the corrected choice as the current decision (latest wins)."""
    m = _DECISION_VERB_RE.search(turn or "")
    value = (m.group(1).strip() if m else "").rstrip(".!")
    # Prefer a recognized tech slot for a clean key; else a generic "choice".
    slots = extract_slots(value or turn or "", "")
    keyed = False
    for key in ("language", "framework", "platform"):
        v = slots.get(key)
        if v:
            state.set_decision(key, str(v))
            state.add_entity(str(v))
            keyed = True
    if not keyed and value:
        state.set_decision("choice", value)


def _remove_for(turn: str, resolution, state) -> None:
    """Reverse: remove the decision the user is undoing. Prefer a resolved
    antecedent / named slot; else clear the generic 'choice'."""
    slots = extract_slots(turn or "", "")
    removed = False
    for key in ("language", "framework", "platform"):
        if slots.get(key):
            state.remove_decision(key)
            removed = True
    if not removed and resolution is not None and getattr(resolution, "antecedents", None):
        # Antecedent might match a decision value — drop any decision with it.
        ant = resolution.antecedents[0].lower()
        for k, v in list(state.decisions().items()):
            if v.lower() == ant:
                state.remove_decision(k)
                removed = True
    if not removed:
        state.remove_decision("choice")


def _record_negatives(turn: str, state) -> None:
    for m in _NEGATIVE_RE.finditer(turn or ""):
        tok = m.group(1).strip().rstrip(".!,")
        # Skip trivial stopword captures.
        if tok and len(tok) >= 2 and tok.lower() not in ("the", "a", "an", "it"):
            state.add_constraint(tok, negative=True)


def commit(turn: str, answer: str, state, *, asked_question: str | None = None) -> None:
    """After an answer streams: register entities + enumerations from it, and
    resolve any open question the user just answered. No-op on error (R1.4)."""
    try:
        _commit(turn, answer, state, asked_question)
    except Exception:  # noqa: BLE001
        pass


def _commit(turn: str, answer: str, state, asked_question: str | None) -> None:
    # 1) Entities named in the answer (reuse the deterministic extractor).
    slots = extract_slots(answer or "", "")
    for key in ("language", "framework", "platform"):
        v = slots.get(key)
        if v:
            state.add_entity(str(v))

    # 2) Enumerated options in the answer → selection-reference antecedents.
    options = _extract_enumerations(answer or "")
    if options:
        state.set_enumerations(options)

    # 3) Open-question lifecycle: a turn that answered a pending question clears
    #    it; a new assistant question that goes unanswered is recorded.
    if asked_question:
        state.add_open_question(asked_question)
    else:
        # If the user's turn looks like it answered an open question, clear it.
        for q in state.open_questions():
            state.clear_open_question(q)
            break


def _extract_enumerations(answer: str) -> list[str]:
    """Pull a list of enumerated options from a markdown / numbered list, in
    order. Caps to a small head so selection references stay meaningful."""
    items: list[str] = []
    for line in (answer or "").splitlines():
        m = _NUM_ITEM_RE.match(line)
        if m:
            # Strip leading markdown emphasis / bold and trailing colons.
            opt = re.sub(r"^[*_`]+|[*_`:]+$", "", m.group(1).strip()).strip()
            # Take the option label before a ':' or ' - ' description.
            opt = re.split(r"\s[:\-–]\s", opt, maxsplit=1)[0].strip()
            if opt and len(opt) <= 60:
                items.append(opt)
        if len(items) >= 12:
            break
    return items


def continuation_directive(state) -> str:
    """A directive for a continuation turn (R7). If the prior answer enumerated
    items, ask for the NEXT ones; otherwise resume without repeating."""
    try:
        enums = state.enumerations()
    except Exception:  # noqa: BLE001
        enums = []
    if enums:
        return ("Continue from the previous list — produce the NEXT items, do "
                "not repeat any already given.")
    return ("Continue the previous answer from where it ended; do not repeat "
            "content already provided. If it was complete, extend or deepen it.")


__all__ = ["apply_turn", "commit", "continuation_directive"]
