"""Session budget / economics (live-conversational-intelligence R22; task 21.2).

Pins Property 22: concurrency cap acquire/shed/release, total-answer cap,
per-session registry, and unbounded behavior at a high cap.
"""
from __future__ import annotations

from app.live.budget import SessionBudget, get_budget


def test_concurrency_cap_sheds_then_releases():
    b = SessionBudget(max_concurrent=2)
    assert b.acquire() is True       # 1
    assert b.acquire() is True       # 2
    assert b.acquire() is False      # at cap → shed
    b.release()
    assert b.acquire() is True       # slot freed


def test_total_answer_cap():
    b = SessionBudget(max_concurrent=10, max_answers=2)
    assert b.acquire() is True
    b.release()
    assert b.acquire() is True
    b.release()
    assert b.acquire() is False      # total cap reached


def test_unbounded_when_zero_caps():
    b = SessionBudget(max_concurrent=0, max_answers=0)
    assert all(b.acquire() for _ in range(20))   # never sheds


def test_snapshot_shape():
    b = SessionBudget(max_concurrent=3)
    b.acquire()
    s = b.snapshot()
    assert s["active"] == 1 and s["max_concurrent"] == 3


def test_registry_per_session():
    a = get_budget("s1")
    assert get_budget("s1") is a
    assert get_budget("s2") is not a
