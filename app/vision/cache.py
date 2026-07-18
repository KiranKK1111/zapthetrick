"""Per-image vision cache (VisionAnalysis.md "cache the vision output").

The expensive vision parse runs ONCE per distinct image; a follow-up question
about the same screenshot/document reuses the cached structured text and skips
the vision stage entirely. Keyed by a hash of the image bytes (+ the prompt, so
a different extraction instruction re-parses). Small bounded LRU — vision text
is a few KB, and we only need the current session's images hot.

This is process-local and best-effort: a miss just re-parses. It complements
(does not replace) the per-message `sources.image_analysis` reuse the chat
route already does within one conversation.
"""
from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from typing import Sequence


def image_key(images: Sequence[bytes], prompt: str) -> str:
    """Content-addressed key for a (images, prompt) parse."""
    h = hashlib.sha256()
    h.update((prompt or "").encode("utf-8", "ignore"))
    h.update(b"\x00")
    for img in images:
        # Hash a length prefix + the bytes so [a,b] and [ab] never collide.
        h.update(len(img).to_bytes(8, "big"))
        h.update(img)
    return h.hexdigest()


class VisionCache:
    """Thread-safe bounded LRU of image-hash -> extracted text."""

    def __init__(self, max_entries: int = 128) -> None:
        self._max = max(1, int(max_entries))
        self._data: "OrderedDict[str, str]" = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            val = self._data.get(key)
            if val is not None:
                self._data.move_to_end(key)  # mark recently used
            return val

    def put(self, key: str, value: str) -> None:
        if not value:
            return
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)  # evict least-recently-used

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


# Process-wide singleton (sized from config on first factory use).
_CACHE: VisionCache | None = None
_CACHE_LOCK = threading.Lock()


def get_cache(max_entries: int = 128) -> VisionCache:
    global _CACHE
    with _CACHE_LOCK:
        if _CACHE is None:
            _CACHE = VisionCache(max_entries)
        return _CACHE
