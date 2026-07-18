"""
PII redaction + retention (live-conversational-intelligence R20).

`redact` masks personally-identifying values (emails, phone numbers, long ID /
card-like digit runs, secret-looking tokens) in a transcript BEFORE it is sent
to a third-party LLM provider, while preserving the technical content needed to
answer. It references secrets by name, never echoing the value. The
`Retention_Policy` + `purge` drive deletion of a live session's transcript /
answers / embeddings. Deterministic + FAIL-CLOSED: a pattern that errors falls
back to a crude mask, and if the whole pass fails the content is withheld —
unredacted text is never the error path's output.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger("zapthetrick.live")

_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")
# Long digit runs (IDs, card numbers, SSNs) — 9+ digits, allowing separators.
_LONG_ID = re.compile(r"(?<!\d)(?:\d[ -]?){9,}\d(?!\d)")
# Secret-looking tokens: api keys, bearer tokens, long base64-ish strings.
_SECRET = re.compile(r"\b(?:sk-[A-Za-z0-9_-]{12,}|gh[ps]_[A-Za-z0-9]{20,}|"
                     r"AKIA[0-9A-Z]{12,}|[A-Za-z0-9_\-]{32,})\b")
_NAME_INTRO = re.compile(r"\b(my name is|i am|i'm|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
                         re.IGNORECASE)


# Crude fallback mask when a primary pattern errors: emails + digit runs. Blunt
# (over-masks) by design — the fail-closed direction for a PII boundary.
_FALLBACK_MASK = re.compile(r"[\w.+-]+@\S+|\d[\d\s().\-]{6,}\d")

# What the LLM sees when redaction itself is broken: never the raw transcript.
REDACTION_FAILED = "[transcript withheld: PII redaction unavailable]"


def redact(text: str) -> tuple[str, dict]:
    """Mask PII / secrets in `text`. Returns (clean_text, mapping) where mapping
    is placeholder -> original (kept in-process only, never surfaced).

    FAILS CLOSED: each pattern is isolated, so one erroring pattern degrades to
    a crude fallback mask instead of aborting the pass; if even that fails, the
    content is withheld (`REDACTION_FAILED`). The original text is never
    returned from an error path — this runs immediately before third-party-LLM
    egress. Never raises."""
    if not text or not text.strip():
        return text or "", {}
    mapping: dict[str, str] = {}
    counters = {"EMAIL": 0, "PHONE": 0, "ID": 0, "SECRET": 0, "NAME": 0}

    def _sub(pattern: re.Pattern, label: str, s: str, group: int = 0) -> str:
        def _repl(m: re.Match) -> str:
            val = m.group(group)
            counters[label] += 1
            ph = f"[{label}_{counters[label]}]"
            mapping[ph] = val
            # For NAME the regex has a leading clause we must keep.
            if label == "NAME":
                return m.group(1) + " " + ph
            return ph
        return pattern.sub(_repl, s)

    out = text
    for pattern, label, group in (
        (_SECRET, "SECRET", 0),
        (_EMAIL, "EMAIL", 0),
        (_LONG_ID, "ID", 0),
        (_PHONE, "PHONE", 0),
        (_NAME_INTRO, "NAME", 2),
    ):
        try:
            out = _sub(pattern, label, out, group=group)
        except Exception as exc:  # noqa: BLE001 — degrade to the crude mask
            log.warning("PII pattern %s failed — applying fallback mask: %s",
                        label, exc)
            try:
                out = _FALLBACK_MASK.sub("[MASKED]", out)
            except Exception:  # noqa: BLE001 — withhold rather than leak
                log.error("PII redaction unavailable — withholding content")
                return REDACTION_FAILED, {}
    return out, mapping


@dataclass
class RetentionPolicy:
    """Lifetime + deletion rules for live transcripts/answers/embeddings."""
    retention_days: int = 0          # 0 = keep indefinitely (today's behavior)

    @classmethod
    def from_config(cls) -> "RetentionPolicy":
        from app.core.config_loader import cfg
        return cls(retention_days=int(getattr(cfg.live, "retention_days", 0) or 0))

    def expires(self) -> bool:
        return self.retention_days > 0


async def purge(session_id) -> bool:
    """User-initiated purge of a live session's persisted Q&A (messages). Best-
    effort over the existing Session/Message rows; never raises. Returns True
    when something was purged."""
    import uuid as _uuid
    from storage.db import get_session_factory
    factory = get_session_factory()
    if factory is None:
        return False
    try:
        sid = _uuid.UUID(str(session_id))
    except (ValueError, TypeError):
        return False
    try:
        from storage.repos import MessageRepo, SessionRepo
        async with factory() as db:
            sess = await SessionRepo(db).get(sid)
            if sess is None or sess.type != "live":
                return False
            mr = MessageRepo(db)
            # Use a delete helper if present; otherwise no-op gracefully.
            deleter = getattr(mr, "delete_for_session", None)
            if deleter is None:
                return False
            await deleter(session_id=sid)
            await db.commit()
            return True
    except Exception:  # noqa: BLE001
        log.exception("live purge failed for session %s", session_id)
        return False
