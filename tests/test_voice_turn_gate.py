"""First-real-word gate for the chat voice-mode WS (/ws/voice).

VoiceModeArchitecture.md §"First Real Word Rule" / §"recognized_word_count >= 1":
a lone cough / "uh" the recognizer renders as a token must NOT take a
conversational turn, while any real utterance (including short single words and
non-Latin scripts) must pass. This locks that boundary so background sounds stop
launching turns ("interrupts on small sounds").
"""
import pytest

from app.api.routes_ws import _voice_is_meaningful as meaningful


@pytest.mark.parametrize("text", [
    "",            # nothing
    "   ",         # whitespace
    ".",           # punctuation only
    "?",           # punctuation only
    "uh",          # filler
    "um",          # filler
    "hmm",         # filler
    "mm-hmm",      # backchannel
    "uh um",       # all-filler multiword
    "mm hmm",      # all-filler multiword
    "a",           # stray single ASCII letter (STT noise)
    "o",           # stray single ASCII letter
])
def test_ignored_utterances(text):
    assert meaningful(text) is False


@pytest.mark.parametrize("text", [
    "Kafka?",                 # single real word
    "no",                     # short real reply
    "stop",                   # interrupt keyword
    "what is recursion",      # full question
    "uh, what is Kafka?",     # filler opener + real question
    "yeah okay sure",         # multiword, not all fillers set
    "क्या है",                 # Hindi (multi-word)
    "हाँ",                     # Hindi single word (combining marks)
    "क",                      # single non-Latin glyph is a real word
])
def test_meaningful_utterances(text):
    assert meaningful(text) is True
