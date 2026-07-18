"""Mid-stream output guards — kill "messy, never-ending" model output LIVE.

Two failure modes reach users despite the post-stream sanitizer
(`app/response_arch/sanitize.strip_reasoning` cleans only the PERSISTED copy):

* Inline reasoning junk streamed raw: `<think>…</think>` blocks (DeepSeek/Qwen)
  and GPT-OSS harmony channel tokens (`<|channel|>analysis<|message|>…`) show
  up token-by-token in the visible answer.
* Degenerate loops: a model repeating the same well-formed sentence forever
  passes every gibberish check and streams until the provider timeout.

`StreamScrubber` is an incremental version of `strip_reasoning` (state machine
across chunk boundaries); `RepetitionGuard` stops a stream after the same
normalized sentence repeats N times in a row, or after a hard char ceiling.
Both are wired once in `LLMClient.stream_chat`, so chat SSE, live WS, and the
speculative path are all covered at the source.
"""
from __future__ import annotations

import re

# Markers of interest: think-style tags (open/close) + harmony control tokens.
_MARK = re.compile(
    r"<\s*(/?)\s*(think|thinking|reasoning|scratchpad)\s*>|<\|([a-z_]+)\|>",
    re.IGNORECASE,
)
# How much text to hold back when a marker may be split across chunks.
_TAIL = 48


class StreamScrubber:
    """Incrementally remove reasoning blocks / harmony tokens from a stream.

    feed(chunk) returns the clean text that is SAFE to show now; flush()
    returns any held-back tail once the stream ends. Content inside an
    unterminated think block is dropped (matches strip_reasoning's dangling-
    open behavior)."""

    def __init__(self) -> None:
        self._tail = ""
        # state: "normal" | "think" | "harmony_header" | "harmony_skip"
        self._state = "normal"
        self._think_tag = ""

    def feed(self, chunk: str) -> str:
        text = self._tail + (chunk or "")
        self._tail = ""
        out: list[str] = []
        pos = 0
        while pos <= len(text):
            if self._state == "normal":
                m = _MARK.search(text, pos)
                if not m:
                    # Emit everything except a possible partial marker tail.
                    cut = len(text)
                    lt = text.rfind("<", max(pos, len(text) - _TAIL))
                    if lt >= pos:
                        cut = lt
                    out.append(text[pos:cut])
                    self._tail = text[cut:]
                    return "".join(out)
                out.append(text[pos:m.start()])
                pos = m.end()
                if m.group(2):                      # think-style tag
                    if not m.group(1):              # open tag
                        self._state = "think"
                        self._think_tag = m.group(2).lower()
                    # stray close tag → just drop it
                elif (m.group(3) or "").lower() == "channel":
                    self._state = "harmony_header"
                # other harmony tokens (<|end|>, <|start|>, …) → drop
            elif self._state == "think":
                m = re.compile(
                    rf"<\s*/\s*{self._think_tag}\s*>", re.IGNORECASE
                ).search(text, pos)
                if not m:
                    # Drop the reasoning content, keep a tail for a split
                    # close tag.
                    self._tail = text[max(pos, len(text) - _TAIL):]
                    return "".join(out)
                pos = m.end()
                self._state = "normal"
            elif self._state == "harmony_header":
                m = re.compile(r"<\|message\|>", re.IGNORECASE).search(text, pos)
                if not m:
                    self._tail = text[max(pos - 32, 0):]
                    # keep the channel name we've seen so far in the tail
                    return "".join(out)
                channel = text[pos:m.start()].strip().lower()
                pos = m.end()
                self._state = "normal" if channel == "final" else "harmony_skip"
            elif self._state == "harmony_skip":
                m = _MARK.search(text, pos)
                if not m:
                    self._tail = text[max(pos, len(text) - _TAIL):]
                    return "".join(out)
                pos = m.end()
                tok = (m.group(3) or "").lower()
                if m.group(2):
                    continue                        # think tag inside skip — stay
                if tok == "channel":
                    self._state = "harmony_header"
                elif tok in ("end", "start", "return"):
                    self._state = "normal"
            else:  # pragma: no cover — unreachable
                break
        return "".join(out)

    def flush(self) -> str:
        tail, self._tail = self._tail, ""
        if self._state != "normal":
            return ""           # unterminated reasoning → drop
        # The held tail may still contain bare harmony tokens; strip them.
        return _MARK.sub("", tail)


_SENT_SPLIT = re.compile(r"(?<=[.!?\n])\s+")
_NORM = re.compile(r"[^a-z0-9]+")


class RepetitionGuard:
    """Stop degenerate loops: the same normalized sentence N times in a row,
    or total output beyond a hard char ceiling. feed() returns True when the
    stream should be killed."""

    def __init__(self, max_repeats: int = 3, max_chars: int = 120_000,
                 min_unit_len: int = 12) -> None:
        self.max_repeats = max(2, int(max_repeats))
        self.max_chars = int(max_chars)
        self.min_unit_len = int(min_unit_len)
        self._buf = ""
        self._last = ""
        self._count = 0
        self._chars = 0

    def feed(self, chunk: str) -> bool:
        self._chars += len(chunk)
        if self.max_chars and self._chars > self.max_chars:
            return True
        self._buf += chunk
        if len(self._buf) > 20_000:                 # runaway single sentence
            self._buf = self._buf[-20_000:]
        parts = _SENT_SPLIT.split(self._buf)
        # Last part may be an incomplete sentence — keep it buffered.
        self._buf = parts[-1]
        for unit in parts[:-1]:
            n = _NORM.sub(" ", unit.lower()).strip()
            if len(n) < self.min_unit_len:
                continue
            if n == self._last:
                self._count += 1
                if self._count >= self.max_repeats:
                    return True
            else:
                self._last = n
                self._count = 1
        return False
