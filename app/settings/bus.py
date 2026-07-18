"""Tiny in-process pub/sub for live config changes.

The bus is deliberately single-process and asyncio-only — the app is
device-local per Architecture2.md, so we don't need a real broker.

Lifecycle:
    1. At startup, each subsystem registers a subscriber:
           bus.subscribe("llm",       _rebuild_llm_client)
           bus.subscribe("embeddings", _rebuild_embedder)
           ...
    2. `POST /api/settings` calls `bus.publish(section, diff, full_cfg)`
       once per top-level section the user changed.
    3. Subscribers swap their in-process state. Failures are logged
       but don't propagate — one bad subscriber shouldn't break the
       save.

Subscribers may be sync or async. They get:
    handler(section: str, diff: dict, full_cfg: dict) -> None | Awaitable[None]
"""
from __future__ import annotations

import asyncio
import inspect
import logging
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Union


log = logging.getLogger(__name__)


Handler = Callable[[str, dict, dict], Union[None, Awaitable[None]]]


@dataclass
class ConfigEvent:
    """A single config-change notification. Carries both the diff
    (only the keys that changed) and the full new config so subscribers
    don't have to compose them themselves."""
    section: str
    diff: dict
    full_cfg: dict


class ConfigBus:
    def __init__(self) -> None:
        self._subs: dict[str, list[Handler]] = {}

    def subscribe(self, section: str, handler: Handler) -> None:
        """Register a callback for `section`. Replaces nothing — every
        registered handler fires on every publish."""
        self._subs.setdefault(section, []).append(handler)
        log.debug("config bus: +subscriber for %r (n=%d)", section, len(self._subs[section]))

    def unsubscribe(self, section: str, handler: Handler) -> bool:
        subs = self._subs.get(section)
        if not subs:
            return False
        try:
            subs.remove(handler)
            return True
        except ValueError:
            return False

    async def publish(self, section: str, diff: dict, full_cfg: dict) -> None:
        """Fire every subscriber for `section`. Sync handlers run
        inline; async handlers are awaited (sequentially — keeps
        ordering predictable, e.g. embeddings before reranker)."""
        subs = list(self._subs.get(section, []))
        if not subs:
            return
        log.info("config bus: publish %r (subscribers=%d)", section, len(subs))
        for h in subs:
            try:
                result = h(section, diff, full_cfg)
                if inspect.isawaitable(result):
                    await result
            except Exception as exc:  # noqa: BLE001 — never break the save
                log.exception("config bus: subscriber for %r raised: %s", section, exc)


bus = ConfigBus()


def diff_paths(old: Any, new: Any, prefix: str = "") -> dict[str, dict]:
    """Return `{section_name: diff_subtree}` for every top-level key
    in `new` that differs from `old`. Deep comparison.

    "Top-level" here means the first level of the config dict —
    `llm`, `embeddings`, `database`, etc. The whole sub-dict is
    returned as the diff so subscribers see the full new state of
    their section.
    """
    if not isinstance(old, dict) or not isinstance(new, dict):
        return {prefix: new} if old != new else {}
    out: dict[str, dict] = {}
    for key, new_val in new.items():
        old_val = old.get(key)
        if isinstance(new_val, dict) and isinstance(old_val, dict):
            if _deep_diff(old_val, new_val):
                out[key] = new_val
        elif old_val != new_val:
            # Scalar changed at the top level — wrap so the diff is
            # still a dict for the handler signature.
            out[key] = {key: new_val}
    return out


def _deep_diff(a: dict, b: dict) -> bool:
    """True if `a` and `b` differ at any depth."""
    if a.keys() != b.keys():
        return True
    for k, v_b in b.items():
        v_a = a[k]
        if isinstance(v_a, dict) and isinstance(v_b, dict):
            if _deep_diff(v_a, v_b):
                return True
        elif v_a != v_b:
            return True
    return False


__all__ = ["ConfigBus", "ConfigEvent", "bus", "diff_paths"]


# Quiet flake8 — `asyncio` is reserved for sub-classes that may want to
# spawn background work from a subscriber.
_ = asyncio
