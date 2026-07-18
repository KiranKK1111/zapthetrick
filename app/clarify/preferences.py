"""Clarification preference memory (Phase 4).

A small, dict-based store over the `User.preferences` JSONB blob. It is pure and
synchronous so it can be unit-tested without a database: the route loads the
blob, mutates it through this store, then persists by reassigning
`user.preferences` and committing (see [load_store] / [save_store]).

Layout under `preferences["clarify"]`:

    durable:   {choice: value}                  cross-session prefs (R17)
    sessions:  {conversation_id: {choice: value}} in-conversation memory (R16)
    counts:    {choice: {value: n}}              repeat tracking for promotion
    mode:      str | None                        active Clarification_Mode (R18)
    contract:  {setting: value}                  collaboration contract (R19/R30)
    analytics: {asked, answered, skipped, modified}  (R32)

Choices are keyed by their question `header` (e.g. "Language"); values are the
chosen option label(s). Everything is scoped to one device user; [clear] wipes
all of it (R36).
"""
from __future__ import annotations

import re

_PROMOTE_THRESHOLD = 3  # same value chosen this many times → durable (R16.3)

_ANALYTICS_KEYS = ("asked", "answered", "skipped", "modified")


def _empty_clarify() -> dict:
    return {
        "durable": {},
        "sessions": {},
        "counts": {},
        "mode": None,
        "contract": {},
        "analytics": {k: 0 for k in _ANALYTICS_KEYS},
    }


class ClarificationPreferenceStore:
    """Mutates a `preferences` dict in place; the caller persists it."""

    def __init__(self, prefs: dict | None, *, conversation_id: str | None = None):
        self.root: dict = prefs if isinstance(prefs, dict) else {}
        self._cid = conversation_id
        c = self.root.get("clarify")
        if not isinstance(c, dict):
            c = _empty_clarify()
            self.root["clarify"] = c
        # Backfill any missing sub-keys (older blobs).
        for k, v in _empty_clarify().items():
            c.setdefault(k, v)
        self._c = c

    # ---- reads -----------------------------------------------------------
    def durable_prefs(self) -> dict:
        return dict(self._c["durable"])

    def session_prefs(self) -> dict:
        if not self._cid:
            return {}
        return dict(self._c["sessions"].get(self._cid, {}))

    def known_choices(self) -> dict:
        """Durable prefs overlaid by this conversation's answers — the set of
        choices the gate should treat as ALREADY DECIDED (R16.2, R17)."""
        merged = dict(self._c["durable"])
        merged.update(self.session_prefs())
        return merged

    def mode(self) -> str | None:
        return self._c.get("mode")

    def contract(self) -> dict:
        return dict(self._c.get("contract", {}))

    def analytics(self) -> dict:
        return dict(self._c["analytics"])

    # ---- writes ----------------------------------------------------------
    def record_answer(self, choice: str, value: str) -> None:
        """Record one answered choice for this conversation; promote to durable
        once the same value recurs `_PROMOTE_THRESHOLD` times (R16.1/16.3)."""
        choice = (choice or "").strip()
        value = (value or "").strip()
        if not choice or not value:
            return
        if self._cid:
            self._c["sessions"].setdefault(self._cid, {})[choice] = value
        counts = self._c["counts"].setdefault(choice, {})
        counts[value] = int(counts.get(value, 0)) + 1
        if counts[value] >= _PROMOTE_THRESHOLD:
            self._c["durable"][choice] = value
        self._c["analytics"]["answered"] = \
            int(self._c["analytics"].get("answered", 0)) + 1

    def record_answers(self, answers: dict) -> None:
        for choice, value in (answers or {}).items():
            self.record_answer(str(choice), str(value))

    def set_durable(self, choice: str, value: str) -> None:
        choice = (choice or "").strip()
        if choice:
            self._c["durable"][choice] = (value or "").strip()

    def set_mode(self, mode: str | None) -> None:
        self._c["mode"] = mode

    def set_contract(self, settings: dict) -> None:
        if isinstance(settings, dict):
            self._c["contract"] = {**self._c.get("contract", {}), **settings}

    def analytics_record(self, event: str, n: int = 1) -> None:
        if event in _ANALYTICS_KEYS:
            self._c["analytics"][event] = \
                int(self._c["analytics"].get(event, 0)) + n

    def clear(self) -> None:
        """Forget everything — preferences, sessions, counts, contract,
        analytics (R36)."""
        self.root["clarify"] = _empty_clarify()
        self._c = self.root["clarify"]


# ---- persistence helpers (used by the route) -----------------------------
async def load_store(session, user_id, *, conversation_id: str | None = None):
    """Load the device user's preferences into a store. Returns (store, user)
    or (None, None) if the user can't be loaded (anonymous / db down)."""
    if user_id is None:
        return None, None
    try:
        from storage.models import User
        user = await session.get(User, user_id)
    except Exception:  # noqa: BLE001
        return None, None
    if user is None:
        return None, None
    store = ClarificationPreferenceStore(
        dict(user.preferences or {}), conversation_id=conversation_id)
    return store, user


async def save_store(session, user, store) -> None:
    """Persist the mutated blob (reassign for JSONB change detection)."""
    if user is None or store is None:
        return
    try:
        user.preferences = dict(store.root)
        await session.commit()
    except Exception:  # noqa: BLE001 — never let preference IO break a turn
        pass


# Lines like "Language: Python" / "Features: A, B" produced by the
# clarification panel. Parsed back into {choice: value} so the gate can remember
# them (Phase 4). Headers are short labels (<=3 words); prose / code lines with a
# long left-hand side are ignored.
_PRIORITY_SUFFIX = re.compile(r"\s+by priority$", re.IGNORECASE)
# A header is a short human label — letters/digits/spaces and a few separators.
# This rejects code (parentheses, quotes, operators) and prose.
_LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 /&+.\-]*$")


def parse_answer_lines(text: str) -> dict:
    out: dict = {}
    for raw in (text or "").splitlines():
        line = raw.strip()
        if ":" not in line:
            continue
        head, _, val = line.partition(":")
        head = _PRIORITY_SUFFIX.sub("", head.strip())
        val = val.strip()
        if (head and val and len(head.split()) <= 3 and len(head) <= 40
                and _LABEL_RE.fullmatch(head)):
            out[head] = val
    return out
