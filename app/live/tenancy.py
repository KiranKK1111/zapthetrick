"""
Enterprise_Readiness: per-user scoping + shared libraries + team analytics
(live-conversational-intelligence R62).

Adds multi-tenant readiness ON TOP of the existing `User`/`Session` ownership:
sessions / profiles / assets / event logs are SCOPED to their owning user so no
cross-user leakage occurs; resume / interview-knowledge libraries can be shared
read-only; team analytics are aggregated WITHOUT any PII. Disabled by default →
single-user behavior is byte-for-byte unchanged. Composes with privacy/retention
(R20) + consent (R17). Deterministic + fail-open.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Fields considered PII — excluded from any cross-user aggregation.
_PII_FIELDS = frozenset({
    "name", "full_name", "email", "phone", "address", "linkedin", "github",
    "user_id", "owner", "resume_text", "raw_text", "dob", "ssn",
})


def is_enabled() -> bool:
    """Whether enterprise readiness is on. Off by default → single-user."""
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.live, "enterprise_readiness", False))
    except Exception:  # noqa: BLE001
        return False


def owns(resource_owner_id, requesting_user_id) -> bool:
    """Per-user scoping check: a user may only access resources they own. When
    enterprise mode is OFF this is a no-op (single-user → always True). Never
    raises → False (deny) on error when enabled."""
    try:
        if not is_enabled():
            return True
        if resource_owner_id is None or requesting_user_id is None:
            return False
        return str(resource_owner_id) == str(requesting_user_id)
    except Exception:  # noqa: BLE001
        return False


def scope_query(records: list[dict], requesting_user_id, owner_key: str = "owner") -> list[dict]:
    """Filter a list of records to those owned by the requesting user (when
    enterprise mode is on). Off → returns the list unchanged. Never raises."""
    try:
        if not is_enabled():
            return list(records or [])
        return [r for r in (records or [])
                if owns(r.get(owner_key), requesting_user_id)]
    except Exception:  # noqa: BLE001
        return []


def strip_pii(record: dict) -> dict:
    """Drop PII fields from a record (for shared libraries / aggregation). Never
    raises."""
    try:
        return {k: v for k, v in (record or {}).items() if k.lower() not in _PII_FIELDS}
    except Exception:  # noqa: BLE001
        return {}


@dataclass
class TeamAnalytics:
    sessions: int = 0
    answers: int = 0
    avg_answers_per_session: float = 0.0
    top_topics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"sessions": self.sessions, "answers": self.answers,
                "avg_answers_per_session": round(self.avg_answers_per_session, 2),
                "top_topics": self.top_topics, "pii_free": True}


def aggregate_team(sessions: list[dict]) -> TeamAnalytics:
    """Aggregate read-only team analytics from per-session summaries WITHOUT any
    PII. Each session dict may carry {answers, topics:[...], + PII fields that
    are ignored}. Never raises."""
    ta = TeamAnalytics()
    try:
        clean = [strip_pii(s) for s in (sessions or [])]
        ta.sessions = len(clean)
        topic_counts: dict[str, int] = {}
        total_answers = 0
        for s in clean:
            total_answers += int(s.get("answers", 0) or 0)
            for t in (s.get("topics") or []):
                key = str(t).lower()
                topic_counts[key] = topic_counts.get(key, 0) + 1
        ta.answers = total_answers
        ta.avg_answers_per_session = (total_answers / ta.sessions) if ta.sessions else 0.0
        ta.top_topics = sorted(topic_counts, key=topic_counts.get, reverse=True)[:5]
        return ta
    except Exception:  # noqa: BLE001
        return ta
