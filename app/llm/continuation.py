"""Mid-stream continuation contract (Architecture §15 — "always finishes").

A stream can end before the answer is complete in two ways:

  * the model stops with `finish_reason == "length"` — it hit the output-token
    ceiling mid-sentence (a *truncation*, not a real stop), or
  * the transport drops after some tokens were already shown (socket reset,
    inter-chunk timeout) — a *cut-off*.

In both cases we don't want to leave the user with a half answer. The fix is a
**continuation contract**: re-prompt the next-best model with the conversation so
far *plus the partial answer* and an instruction to continue seamlessly without
repeating. The seam at the join is de-duplicated so the two halves read as one
continuous answer. Detection relies on the real `finish_reason` recorded by the
adapters (§14); the wiring lives in `engine.stream_with_continuation`.

Everything here is pure and side-effect-free so it is trivially testable.
"""
from __future__ import annotations

CONTINUE_INSTRUCTION = (
    "You are continuing your own previous reply, which was cut off. Resume from "
    "exactly where it stopped: do not repeat, restate, or summarize any text "
    "already written, and do not add a greeting or preamble. Output only the "
    "continuation so it joins seamlessly onto the existing text."
)

# finish_reason values that mean "cut off by an output limit", i.e. worth
# continuing. A clean "stop" (or tool_calls / content_filter) is left alone.
_CUTOFF_REASONS = {"length", "max_tokens", "model_length", "max_output_tokens"}


def is_cutoff(finish_reason: str | None) -> bool:
    """True when the model stopped because it hit a length/token limit — a
    truncation we should continue — rather than finishing cleanly."""
    return (finish_reason or "").strip().lower() in _CUTOFF_REASONS


def build_continuation_messages(
    messages: list[dict], partial_text: str, *, tail_chars: int = 2000
) -> list[dict]:
    """The conversation so far + the partial answer as an assistant turn + a
    'continue seamlessly' user turn.

    The partial is carried as a real assistant message so the model treats it as
    its own prior output; a trimmed tail is also quoted in the instruction so
    providers that under-weight the assistant turn still see where to resume.
    """
    partial = (partial_text or "").strip()
    tail = partial[-tail_chars:]
    instruction = CONTINUE_INSTRUCTION
    if tail:
        instruction += "\n\nThe reply so far ends with:\n" + tail
    return [
        *messages,
        {"role": "assistant", "content": partial},
        {"role": "user", "content": instruction},
    ]


def dedupe_seam(prev_tail: str, new_head: str, *, max_overlap: int = 200) -> str:
    """Trim from `new_head` any leading run that repeats the end of `prev_tail`,
    so a continuation joins without duplicating text at the seam.

    Finds the longest suffix of `prev_tail` that is also a prefix of `new_head`
    (bounded by `max_overlap`) and drops it from the head. Returns `new_head`
    unchanged when there is no overlap.
    """
    if not prev_tail or not new_head:
        return new_head
    a = prev_tail[-max_overlap:]
    limit = min(len(a), len(new_head))
    for k in range(limit, 0, -1):
        if a.endswith(new_head[:k]):
            return new_head[k:]
    return new_head


class SeamDeduper:
    """Stateful seam trimmer for a streamed continuation.

    A continuation's first tokens are the most likely to repeat the tail of what
    was already shown, but the overlap can span several small chunks — so we
    buffer the first `buffer` characters, de-dupe the whole head against the
    prior tail once, then pass everything through untouched.

    Usage per continuation attempt:
        seam = SeamDeduper(prev_tail, active=is_continuation)
        for chunk in stream:
            out = seam.feed(chunk)
            if out: emit(out)
        out = seam.flush()          # leftover if the stream was shorter than buffer
        if out: emit(out)
    """

    def __init__(self, prev_tail: str, *, active: bool, buffer: int = 200):
        self._tail = prev_tail or ""
        self._buffer = buffer
        self._buf: list[str] = []
        self._n = 0
        self._done = not active

    def feed(self, chunk: str) -> str:
        if self._done:
            return chunk
        self._buf.append(chunk)
        self._n += len(chunk)
        if self._n < self._buffer:
            return ""
        return self._release()

    def flush(self) -> str:
        """Emit any still-buffered head (stream ended before the buffer filled)."""
        if self._done:
            return ""
        return self._release()

    def _release(self) -> str:
        joined = dedupe_seam(self._tail, "".join(self._buf))
        self._buf = []
        self._done = True
        return joined


__all__ = [
    "CONTINUE_INSTRUCTION",
    "is_cutoff",
    "build_continuation_messages",
    "dedupe_seam",
    "SeamDeduper",
]
