"""
Semantic barge-in classification (VoiceModeArchitecture.md §"Semantic Barge-In",
§2 Intent Prediction, §3/§4 Acknowledgement + Backchannel Detection).

When the user speaks WHILE the assistant is talking, the question is not "was
there sound" but **"is the user trying to take the conversational turn?"**. The
client previously answered that with hardcoded word lists ("stop"/"wait" vs
"yeah"/"mm-hmm"), which cannot generalize to paraphrases ("hang on a second",
"that's not what I meant", "no I follow you").

This classifies the utterance SEMANTICALLY against exemplar sets — the same
embedding-gate machinery the rest of the orchestration uses (app/semantics/gates
`classify`) — into:

    "interrupt"    — taking the floor: a stop request, a correction, a new
                     question, a redirect. The assistant should stop.
    "backchannel"  — following along: acknowledgement, agreement, filler. The
                     assistant should KEEP speaking.

Fail-open by design: returns None when semantic gates are off or the embedder
isn't ready, so the caller keeps its deterministic word-list fallback (cold
start / slim deploys behave exactly as before).
"""
from __future__ import annotations

# Exemplars are DATA, not rules — extend freely; the classifier generalizes to
# paraphrases by embedding similarity rather than literal matching.
_CLASSES: dict[str, list[str]] = {
    "interrupt": [
        # explicit stop / floor-taking
        "stop", "wait", "hold on", "hang on a second", "wait a minute",
        "let me stop you there", "pause for a second", "that's enough",
        # correction / disagreement
        "no that's not what i meant", "actually i meant something else",
        "that's wrong", "no i was asking about something different",
        "you misunderstood my question", "that's not right",
        # redirect / new question
        "actually let's talk about something else",
        "can you explain it differently instead",
        "what about redis instead", "let me ask you something else",
        "skip that and tell me about the other one",
        "go back to the previous point",
        "can you make it shorter", "just give me the short answer",
    ],
    "backchannel": [
        # acknowledgement / agreement — the user is FOLLOWING, not interrupting
        "yeah", "yes", "okay", "ok got it", "right", "sure", "mm hmm",
        "uh huh", "i see", "makes sense", "that makes sense", "understood",
        "got it thanks", "yeah exactly", "right right", "true",
        "yeah that's what i thought", "okay okay", "yep understood",
        "cool", "nice", "perfect", "great", "fair enough", "yeah go on",
        "okay keep going", "mhm continue",
    ],
}

_CACHE_KEY = "voice_barge_in_v1"
# Nearest-exemplar similarity required to trust the class at all. Below this the
# utterance resembles neither set → None → the caller's fallback decides.
_THRESHOLD = 0.55


def classify_utterance(text: str, *, embed_fn=None) -> str | None:
    """Return "interrupt" | "backchannel", or None when undecided/unavailable.
    Never raises."""
    t = (text or "").strip()
    if not t:
        return None
    try:
        from app.semantics.gates import classify
        return classify(t, _CLASSES, embed_fn=embed_fn,
                        threshold=_THRESHOLD, cache_key=_CACHE_KEY)
    except Exception:  # noqa: BLE001 — fail-open; the caller has a fallback
        return None


__all__ = ["classify_utterance"]
