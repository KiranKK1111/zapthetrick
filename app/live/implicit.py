"""
Implicit-question / semantic-completion detection
(live-conversational-intelligence R48).

Interviewers often probe WITHOUT a syntactic question: "Walk me through your
approach.", "I'm curious about your reasoning here.", "Talk to me about
scaling." These are implicit questions that warrant an answer even though they
lack a '?' or a wh-word. `detect_implicit` flags such utterances so the live
module doesn't miss them. Deterministic + fail-open (errs toward NOT inventing a
question).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.core import lexicons

# Imperative / probing cues that signal an implicit request to respond.
_IMPERATIVE_CUES = lexicons.LIVE_IMPLICIT_IMPERATIVE_CUES

# Trailing-cue: an interviewer prompt that hangs expecting completion.
_TRAILING_CUES = lexicons.LIVE_IMPLICIT_TRAILING_CUES

# Hypothetical / assumption scenario probes.
_HYPO_SINGLE = lexicons.LIVE_HYPOTHETICAL_SINGLE_CUES
_HYPO_PHRASES = lexicons.LIVE_HYPOTHETICAL_PHRASE_CUES

# "I suppose that's fine" / "we assume standard latency" are HEDGES the
# speaker makes about themselves — not scenario probes. A single-word cue
# preceded by one of these pronouns is ignored.
_PRONOUN_GUARD = {"i", "we", "you", "they", "he", "she", "it"}

_WORD_RE = re.compile(r"[a-z']+")


@dataclass
class ImplicitSignal:
    is_implicit_question: bool = False
    confidence: float = 0.0
    cue: str = ""

    def to_dict(self) -> dict:
        return {"is_implicit_question": self.is_implicit_question,
                "confidence": round(self.confidence, 3), "cue": self.cue}


def detect_implicit(utterance: str) -> ImplicitSignal:
    """Detect an implicit/semantic-completion question. Never raises → negative."""
    sig = ImplicitSignal()
    try:
        t = (utterance or "").strip().lower()
        if not t:
            return sig
        # An explicit question is handled elsewhere; this is the implicit layer.
        if "?" in t:
            return sig
        for cue in _IMPERATIVE_CUES:
            if cue in t:
                # The cue lists match SUBSTRINGS, so the speaker holding the
                # floor trips them: "give me one moment" hits "give me",
                # "I want to describe the format" hits "describe". Answering
                # there talks OVER the interviewer. A semantic veto separates
                # direction (asking the listener to talk vs offering to talk)
                # in a way no literal list can. Fail-open → cue wins, as before.
                if holds_floor(t):
                    return sig
                sig.is_implicit_question = True
                sig.cue = cue
                sig.confidence = 0.75
                return sig
        for cue in _TRAILING_CUES:
            if t.endswith(cue):
                if holds_floor(t):
                    return sig
                sig.is_implicit_question = True
                sig.cue = cue
                sig.confidence = 0.55
                return sig
        # SEMANTIC gate (2026-07-09): probing phrasings the cue lists can't
        # anticipate. Similarity IS the confidence, so the promotion
        # thresholds downstream apply unchanged. Fail-open → negative.
        return _semantic_signal("implicit_request", t) or sig
    except Exception:  # noqa: BLE001
        return sig


def holds_floor(t: str) -> bool:
    """True when the utterance semantically reads as the SPEAKER taking/holding
    the floor (self-introduction, agenda, logistics, thinking aloud) rather than
    asking the listener to speak. Vetoes a literal cue match. Fail-open → False
    when the embedder is unavailable, so cue behaviour is unchanged."""
    try:
        from app.semantics import gates as _gates
        return _gates.matches("speaker_holds_floor", t) is True
    except Exception:  # noqa: BLE001
        return False


def _semantic_signal(gate: str, t: str) -> ImplicitSignal | None:
    """ImplicitSignal from an exemplar-embedding gate; None when the gate has
    no opinion (embedder warming/absent) or says no."""
    try:
        from app.semantics import gates as _gates
        s = _gates.score(gate, t)
        if s is None or s < _gates.threshold_for(gate):
            return None
        return ImplicitSignal(is_implicit_question=True,
                              confidence=min(0.95, float(s)),
                              cue="semantic")
    except Exception:  # noqa: BLE001
        return None


def detect_hypothetical(utterance: str) -> ImplicitSignal:
    """Detect a hypothetical / assumption SCENARIO probe: "Suppose one
    service goes down.", "Let's say we have a million users.", "Imagine your
    API is slow." — the interviewer expects a response even though there is
    no wh-word and often no '?'.

    Unlike `detect_implicit` this does NOT bail on '?' ("What if the cache
    dies?" is hypothetical AND interrogative). Single-word cues are
    pronoun-guarded so "I suppose that's fine" (a hedge) never fires.
    Never raises → negative."""
    sig = ImplicitSignal()
    try:
        t = (utterance or "").strip().lower()
        if not t:
            return sig
        for phrase in _HYPO_PHRASES:
            pos = t.find(phrase)
            if pos >= 0:
                sig.is_implicit_question = True
                sig.cue = phrase
                sig.confidence = 0.8 if pos == 0 else 0.65
                return sig
        words = _WORD_RE.findall(t)
        for cue in _HYPO_SINGLE:
            if cue not in words:
                continue
            idx = words.index(cue)
            prev = words[idx - 1] if idx > 0 else ""
            if prev in _PRONOUN_GUARD:
                continue  # "I suppose…" — a hedge, not a probe
            sig.is_implicit_question = True
            sig.cue = cue
            sig.confidence = 0.8 if idx == 0 else 0.65
            return sig
        # Semantic tail — scenario probes phrased without any cue word. The
        # floor veto applies here too: "give me one moment, my screen froze"
        # reads as a mini-scenario to the embedder, but the speaker is handling
        # their own problem, not posing one to the candidate.
        if holds_floor(t):
            return sig
        return _semantic_signal("hypothetical_scenario", t) or sig
    except Exception:  # noqa: BLE001
        return sig
