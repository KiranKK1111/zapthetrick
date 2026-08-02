"""First-real-word gate — moved out of the Live file (design §L1).

This is voice-only logic that happened to live in `app/api/routes_ws.py`
alongside the Live endpoint. It moved here verbatim; behaviour is unchanged.

A lexical floor, deliberately literal: a genuine closed set like number/date
normalization, NOT intent classification. Only clear non-lexical vocalizations
are listed — real short replies ("yes", "no", "stop", a single technical term)
must pass, because a one-word answer is a legitimate conversational turn.
"""
from __future__ import annotations

import re

# Backchannels that must NOT take a conversational turn on their own
# (VoiceModeArchitecture.md §"First Real Word Rule"): a cough, throat-clear or
# "uh" the recognizer rendered as a lone token should be ignored, not answered.
VOICE_FILLERS = frozenset({
    "uh", "um", "umm", "uhh", "uhm", "hmm", "hm", "hmmm", "mm", "mmm",
    "mhm", "mmhm", "mmhmm", "mm-hmm", "uh-huh", "uhhuh",
    "ah", "ahh", "aha", "huh", "eh", "er", "err", "erm",
})

# Letter-runs in ANY script (re.UNICODE) so Hindi/Telugu/etc. words count as
# words; digits/punctuation are excluded so "." or "?" alone is not "a word".
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def is_meaningful(text: str) -> bool:
    """True when a final transcript is worth taking a conversational turn.

    Rejects empty / punctuation-only strings and lone backchannels so a cough or
    an "uh" cannot launch a turn. Any multi-word utterance passes — a filler can
    legitimately open a real sentence ("uh, what is Kafka?").
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    words = _WORD_RE.findall(t)
    if not words:
        return False  # punctuation / digits only (".", "?")
    if len(words) == 1:
        w = words[0]
        if w in VOICE_FILLERS:
            return False
        # A lone stray ASCII letter ("a", "o") is STT noise, not a turn; a
        # one-character non-Latin word (CJK / Indic base glyph) IS a real word,
        # so the length floor is ASCII-only — never penalize Hindi/Telugu/etc.
        return not (len(w) == 1 and w.isascii())
    # Two+ words: ignore only if EVERY token is a filler ("uh um", "mm hmm").
    return not all(w in VOICE_FILLERS for w in words)


__all__ = ["VOICE_FILLERS", "is_meaningful"]
