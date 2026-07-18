"""
Question-hypothesis buffer + human-like turn-taking
(live-conversational-intelligence R3).

The segmenter finalizes an utterance on endpoint silence. Turn-taking adds a
short *settle window* on top: a finalized utterance is held briefly; if the
speaker continues within the window, the continuation is **merged** into the
same hypothesis instead of being answered as a second question. Only when the
settle window elapses (end-of-turn) is the merged question confirmed.

This module is **pure logic** — time is passed in, so it is deterministic and
testable without timers. The async debounce that drives it lives at the wiring
seam (`routes_ws`), gated by `cfg.live.turn_taking` / `cfg.live.turn_settle_ms`.
With turn-taking off, the segmenter's finalized utterances are answered exactly
as today.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Dynamic endpointing (semantic completeness) ─────────────────────────────
# A fixed silence gap splits a question the moment the speaker pauses to think
# ("So tell me… <2s> …how would you scale Kafka?"). Instead we make the
# end-of-turn wait ADAPTIVE: if the transcript so far looks INCOMPLETE (ends
# on a word that grammatically demands a continuation), we wait longer for the
# speaker to finish; if it looks COMPLETE (ends with '?' or a closed clause),
# we settle fast. This is the "smart endpointing" real ASR stacks use.

# Words that, when the utterance ends on them, signal MORE is coming.
_TRAILING_INCOMPLETE = {
    # coordinating / subordinating conjunctions
    "and", "or", "but", "so", "because", "if", "when", "while", "since",
    "although", "though", "unless", "until", "whereas", "that", "which",
    # prepositions
    "with", "for", "to", "in", "on", "of", "at", "by", "from", "as", "than",
    "about", "into", "over", "under", "between", "through", "against",
    # articles / determiners / possessives
    "the", "a", "an", "my", "your", "our", "their", "its", "this", "these",
    "those", "some", "any",
    # bare auxiliaries / modals (expect a predicate)
    "is", "are", "was", "were", "be", "been", "being", "do", "does", "did",
    "can", "could", "would", "should", "will", "shall", "may", "might", "must",
    "have", "has", "had",
    # a dangling wh-word ("how", "what" …) with nothing after it yet
    "how", "what", "why", "when", "where", "who", "whom", "whose", "which",
}

_WORD_RE = re.compile(r"[A-Za-z0-9']+")


def completeness(text: str) -> str:
    """Classify the utterance tail as 'complete' | 'incomplete' | 'neutral'."""
    t = (text or "").strip()
    if not t:
        return "neutral"
    words = _WORD_RE.findall(t.lower())
    last = words[-1] if words else ""
    prev = words[-2] if len(words) >= 2 else ""
    # Does the WORD tail read mid-thought, regardless of punctuation?
    # * a dangling function word (conjunction / preposition / article / bare
    #   auxiliary / bare wh) → "What is", "…heavily and", "what is the";
    # * a transitive verb that still needs its object → "can you explain";
    # * a pronoun after an opener/modal/verb that expects an object AFTER it
    #   → "how would you", "so tell me" ("what motivates you" ends on a
    #   content verb, so it stays complete).
    dangling = bool(words) and (
        last in _TRAILING_INCOMPLETE
        or last in _TRAILING_EXPECTS_OBJECT
        or (last in {"me", "us", "you"}
            and prev in _EXPECTS_OBJECT_BEFORE_PRONOUN)
    )
    # A terminal question/point mark usually means the thought closed — but
    # ASR punctuation is a GUESS: Parakeet happily appends '?' to a dangling
    # stem ("Can you tell me?"). Trust the mark only when the words don't
    # read mid-thought; otherwise the fake '?' would make every premature
    # endpoint look like a finished question.
    if t[-1] in "?!":
        return "incomplete" if dangling else "complete"
    if not words:
        return "neutral"
    if dangling:
        return "incomplete"
    # Ends on a comma/colon/dash → mid-thought.
    if t[-1] in ",:;-–—":
        return "incomplete"
    # A closed declarative/interrogative with enough words reads as complete.
    if t[-1] == "." and len(words) >= 3:
        return "complete"
    return "neutral"


# Transitive verbs that, when the utterance ENDS on them, still expect an
# object → the speaker is mid-question ("can you explain", "how do you handle").
_TRAILING_EXPECTS_OBJECT = {
    "explain", "describe", "define", "compare", "implement", "handle",
    "design", "discuss", "build", "create", "use", "make", "list", "name",
    "mention", "cover", "optimize", "improve", "scale", "debug", "review",
    "walk", "tell", "show", "give",
}
# Openers/modals/verbs that, followed by a bare "me/us/you", still expect the
# object after the pronoun ("how would you …", "so tell me …").
_EXPECTS_OBJECT_BEFORE_PRONOUN = {
    "would", "could", "will", "can", "should", "do", "does", "did", "tell",
    "give", "show", "walk", "let", "help", "ask", "have", "get",
}


@dataclass
class HypothesisBuffer:
    """Accumulates utterance fragments into one pending question until the
    end-of-turn settle window elapses."""

    settle_ms: int = 600
    _parts: list[str] = field(default_factory=list)
    _audio_present: bool = False
    _last_at: float | None = None
    _generation: int = 0

    def add(self, text: str, now: float, *, has_audio: bool = False) -> int:
        """Append a fragment to the pending hypothesis. Returns the current
        generation id (bumps each time new speech extends the hypothesis) so a
        scheduled settle check can detect that it was superseded."""
        t = (text or "").strip()
        if t:
            self._parts.append(t)
        self._audio_present = self._audio_present or has_audio
        self._last_at = now
        self._generation += 1
        return self._generation

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def has_audio(self) -> bool:
        return self._audio_present

    def pending(self) -> bool:
        return bool(self._parts)

    def required_settle_ms(self) -> int:
        """DYNAMIC end-of-turn wait for the current hypothesis: shorter when the
        question already reads as complete, longer when it ends mid-thought so
        a pausing speaker isn't cut off. Scales the configured `settle_ms`.

        Audio-derived hypotheses (`has_audio`) already sit behind the
        segmenter's VAD endpoint — a real silence gap was confirmed in AUDIO
        time before the utterance finalized. Waiting the full settle window
        again on top of that is a double-wait for the same silence, so those
        collapse: a complete tail answers immediately, a neutral tail keeps
        only a short merge window. Text-injected fragments (no VAD behind
        them) keep the full window. An incomplete tail always waits long —
        the speaker paused mid-thought, and VAD silence can't tell us the
        thought is finished."""
        base = self.settle_ms
        if base <= 0:
            return 0
        kind = completeness(self.merged())
        if kind == "complete":
            if self._audio_present:
                return 0
            # Closed thought → settle fast for a snappy answer.
            return max(250, int(base * 0.6))
        if kind == "incomplete":
            # Grammatically dangling ("What is …", "how would you …") → wait
            # for the speaker to finish. Capped at 2.6s so a real thinking
            # pause mid-question doesn't split "What is <pause> Kafka?" into
            # two questions, while a genuine end-of-turn still resolves.
            return min(2600, int(base * 3.5))
        if self._audio_present:
            # A multi-fragment turn means the speaker is drip-feeding the
            # question between thinking pauses — keep a wider merge window
            # for the next piece instead of committing on the first lull.
            if len(self._parts) > 1:
                return min(2600, base * 2)
            return min(base, 250)
        return base

    def settle_due(self, now: float) -> bool:
        """True once the DYNAMIC settle window has elapsed since the last
        fragment (end of turn). A non-positive settle window is always due
        (turn-taking off)."""
        if not self._parts or self._last_at is None:
            return False
        if self.settle_ms <= 0:
            return True
        return (now - self._last_at) * 1000.0 >= self.required_settle_ms()

    def merged(self) -> str:
        """The merged hypothesis text so far (continuation appended to the
        original), de-duplicating a trivial exact repeat.

        Multi-fragment turns are stitched into ONE sentence: terminal '?'/'.'
        on earlier fragments is an endpointing artifact, not real punctuation
        ("Can you tell me?" + "various annotations" + "in Spring Boot" must
        read "Can you tell me various annotations in Spring Boot?"), and a
        word duplicated across the boundary ("…tell me" + "me about…") is
        dropped once."""
        out: list[str] = []
        for p in self._parts:
            if not out or out[-1] != p:
                out.append(p)
        if not out:
            return ""
        if len(out) == 1:
            return out[0].strip()
        # Only the LAST fragment's own punctuation survives: a '?' on an
        # earlier fragment is the ASR's guess on a premature endpoint — the
        # speaker kept going, so it was wrong. Re-appending it would launder
        # a fake '?' into a "complete question" verdict downstream.
        cleaned: list[str] = []
        for i, p in enumerate(out):
            p = p.strip()
            if i < len(out) - 1:
                p = p.rstrip(" ?.!,;:")
            if cleaned and p:
                prev_last = cleaned[-1].split()[-1].lower().strip("?.!,;:")
                first = p.split()[0].lower().strip("?.!,;:")
                if prev_last == first and len(first) >= 2:
                    p = p.split(None, 1)[1] if " " in p else ""
            if p:
                cleaned.append(p)
        return " ".join(cleaned).strip()

    def take(self) -> tuple[str, bool]:
        """Return (merged_text, had_audio) and clear the buffer."""
        text = self.merged()
        had_audio = self._audio_present
        self._parts = []
        self._audio_present = False
        self._last_at = None
        return text, had_audio
