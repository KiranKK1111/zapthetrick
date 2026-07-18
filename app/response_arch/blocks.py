"""Semantic / block streaming — atomic logical-block emission (Phase 6 #6, #18).

The FE renders progressively, but the BACKEND never told it where the logical
seams were: it emitted an undifferentiated token stream and the client had to
re-guess block boundaries every frame. This module gives the backend a
:class:`BlockAssembler` that consumes token deltas and releases a **block** only
when it is *complete and well-formed*:

* a fenced code block is held until its closing ``` — never a half-open fence;
* a paragraph / heading / table / list is held until the blank line that ends it.

Each released block carries a stable monotonic id, a type, and (for code) its
language + any inferred filename — so the client can render, anchor, and diff by
block instead of by raw offset.

**Progressive artifact delivery (#18)** rides the same seam: when
``emit_artifacts`` is on and a *closed* fenced block looks like a file, the
assembler surfaces it as an :class:`~app.response_arch.artifacts.Artifact` the
moment the fence closes — artifacts land as they complete, not only at ``done``.

Deterministic + fail-open: a parser hiccup degrades to emitting the buffered
text as a single paragraph block, never an exception.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

from .artifacts import Artifact, _infer_filename

# A fence line: column-0 ``` optionally followed by an info string.
_FENCE_OPEN = re.compile(r"^```([A-Za-z0-9+\-_]*)([^\n]*)\n", re.DOTALL)
# The next closing fence somewhere later in the buffer (its own line).
_FENCE_CLOSE = re.compile(r"\n[ \t]*```[ \t]*(?:\n|$)")
# The next fence-open line after position 0 (paragraph → code seam).
_NEXT_FENCE = re.compile(r"\n```")

# Sentinel: "no block terminator yet — keep buffering".
_WAIT = object()


@dataclass
class Block:
    id: int
    type: str            # paragraph | heading | code | table | list
    text: str
    closed: bool = True
    language: str = ""
    filename: str = ""

    def as_frame(self) -> dict:
        d = {"id": self.id, "type": self.type, "text": self.text,
             "closed": self.closed}
        if self.language:
            d["language"] = self.language
        if self.filename:
            d["filename"] = self.filename
        return d


def classify_block(text: str) -> str:
    """Cheap block classification for a non-code chunk."""
    s = (text or "").lstrip()
    if s.startswith("#"):
        return "heading"
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if len(lines) >= 2 and all("|" in ln for ln in lines[:2]) \
            and re.search(r"\|?\s*:?-{2,}", lines[1] if len(lines) > 1 else ""):
        return "table"
    if lines and re.match(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", lines[0]):
        return "list"
    return "paragraph"


@dataclass
class BlockAssembler:
    """Incrementally turn a token stream into complete logical blocks."""

    emit_artifacts: bool = False
    _buf: str = ""
    _ids: "itertools.count" = field(default_factory=lambda: itertools.count(1))
    artifacts: list[Artifact] = field(default_factory=list)

    def feed(self, delta: str) -> list[Block]:
        """Add a token delta; return blocks that are now complete."""
        if delta:
            self._buf += delta
        try:
            return self._drain(final=False)
        except Exception:  # noqa: BLE001 — never break the stream on a parse bug
            return []

    def flush(self) -> list[Block]:
        """Emit whatever remains (the trailing, possibly-open block)."""
        try:
            return self._drain(final=True)
        except Exception:  # noqa: BLE001
            rem, self._buf = self._buf, ""
            return [Block(next(self._ids), "paragraph", rem.strip())] \
                if rem.strip() else []

    # -- internals ----------------------------------------------------------
    def _drain(self, *, final: bool) -> list[Block]:
        out: list[Block] = []
        while True:
            buf = self._buf
            stripped = buf.lstrip("\n")
            if not stripped:
                self._buf = "" if final else buf
                break

            fence_m = _FENCE_OPEN.match(stripped)
            if fence_m:
                blk = self._take_code(stripped, fence_m, final=final)
                if blk is _WAIT:
                    self._buf = buf          # incomplete fence — wait
                    break
                out.append(blk)
                continue

            blk = self._take_prose(stripped, final=final)
            if blk is _WAIT:
                self._buf = buf              # no terminator yet — wait
                break
            if blk is not None:              # skip empty paragraphs
                out.append(blk)
        return out

    def _take_code(self, stripped: str, fence_m: "re.Match", *,
                   final: bool):
        lang = (fence_m.group(1) or "").lower()
        info = fence_m.group(2) or ""
        body_start = fence_m.end()
        close = _FENCE_CLOSE.search(stripped, body_start - 1)
        if close:
            end = close.end()
            block_text = stripped[:end].rstrip("\n")
            content = stripped[body_start:close.start()]
            self._buf = stripped[end:]
            return self._make_code(block_text, lang, info, content, closed=True)
        if final:
            self._buf = ""
            content = stripped[body_start:]
            return self._make_code(stripped.rstrip("\n"), lang, info,
                                   content, closed=False)
        return _WAIT

    def _make_code(self, block_text: str, lang: str, info: str,
                   content: str, *, closed: bool) -> Block:
        filename = ""
        try:
            filename = _infer_filename(content, info, lang or "txt",
                                       preamble="", idx=len(self.artifacts) + 1)
        except Exception:  # noqa: BLE001
            filename = ""
        blk = Block(next(self._ids), "code", block_text, closed=closed,
                    language=lang or "text", filename=filename)
        # Progressive artifact delivery (#18): a CLOSED file-ish code block
        # becomes an artifact the moment it lands.
        if closed and self.emit_artifacts and _is_fileish(filename, content):
            self.artifacts.append(
                Artifact(filename=filename or f"snippet-{len(self.artifacts)+1}",
                         language=lang or "txt", content=content.rstrip("\n")))
        return blk

    def _take_prose(self, stripped: str, *, final: bool):
        # A fence may start before the next blank line — emit the prose first.
        fence_at = _NEXT_FENCE.search(stripped)
        para_end = stripped.find("\n\n")
        if fence_at is not None and (para_end == -1 or fence_at.start() < para_end):
            cut = fence_at.start() + 1     # keep the newline out of the prose
            para = stripped[:cut].rstrip("\n")
            self._buf = stripped[cut:]
            return self._prose_block(para)
        if para_end != -1:
            para = stripped[:para_end]
            self._buf = stripped[para_end + 2:]
            return self._prose_block(para)
        if final:
            self._buf = ""
            return self._prose_block(stripped.rstrip("\n"))
        return _WAIT

    def _prose_block(self, text: str) -> Block | None:
        if not text.strip():
            return None                    # consumed blank run — emit nothing
        return Block(next(self._ids), classify_block(text), text.strip())


def _is_fileish(filename: str, content: str) -> bool:
    """Whether a closed code block is substantial enough to be an artifact."""
    if filename and not filename.startswith("snippet-"):
        return True
    return len((content or "").strip()) >= 40


__all__ = ["Block", "BlockAssembler", "classify_block"]
