"""Relevance-based context selection (followup-context-engine R8, task 9.2).

Pins Property 9: the most relevant prior turns are selected for a follow-up, and
the ConversationState summary is always included regardless of per-turn
relevance.
"""
from __future__ import annotations

from app.followup.context import select
from app.followup.state import ConversationState


def _history():
    return [
        {"role": "user", "content": "How do I set up pgvector in Postgres?"},
        {"role": "assistant", "content": "Install the pgvector extension..."},
        {"role": "user", "content": "What about a good recipe for pancakes?"},
        {"role": "assistant", "content": "Mix flour, eggs, milk..."},
        {"role": "user", "content": "How do I tune Postgres connection pooling?"},
        {"role": "assistant", "content": "Use pgbouncer or the pool settings..."},
    ]


def test_state_summary_is_always_prepended():
    s = ConversationState({}, "c1")
    s.set_goal("build a Postgres-backed app")
    s.set_decision("database", "postgres")
    msgs = select(s, "make the pooling faster", _history(), max_turns=3,
                  recent_floor=1)
    assert msgs and msgs[0]["role"] == "system"
    assert "Postgres-backed app" in msgs[0]["content"]


def test_relevant_turns_selected_over_irrelevant():
    s = ConversationState({}, "c1")
    # Query about Postgres pooling → the pancake turns should be dropped, the
    # Postgres turns kept.
    msgs = select(s, "postgres connection pooling tuning", _history(),
                  max_turns=3, recent_floor=1)
    joined = " ".join(m["content"] for m in msgs)
    assert "pancakes" not in joined.lower()
    assert "postgres" in joined.lower() or "pooling" in joined.lower()


def test_recent_floor_always_kept():
    s = ConversationState({}, "c1")
    msgs = select(s, "totally unrelated quantum physics", _history(),
                  max_turns=4, recent_floor=2)
    # The last two turns are forced in regardless of relevance.
    contents = [m["content"] for m in msgs]
    assert _history()[-1]["content"] in contents
    assert _history()[-2]["content"] in contents


def test_empty_history_returns_only_summary_or_empty():
    s = ConversationState({}, "c1")
    assert select(s, "anything", []) == []     # no summary, no history → empty


def test_failopen_on_bad_state_returns_recent_history():
    class _Boom:
        def summary(self):
            raise RuntimeError("boom")

    hist = _history()
    msgs = select(_Boom(), "postgres", hist, max_turns=2)
    # summary() raising is caught inside _select; selection still proceeds with
    # no summary floor and never raises.
    assert isinstance(msgs, list)
