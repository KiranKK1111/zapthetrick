"""Context budgeting + conversation compression (perceived-speed R12/R13, task 13.3).

Pins Property 9 (allocation invariants + lowest-relevance drop + reserve) and
Property 10 (recent-verbatim + decision recall).
"""
from __future__ import annotations

from app.perceived.context_budget import (
    ConversationCompressor,
    allocate,
    fit,
)


def test_allocation_sums_to_window_and_keeps_reserve():
    b = allocate(200_000)
    assert b.total == 200_000          # exact split (R12.1)
    assert b.reserve > 0               # reserve always kept (R12.3)


def test_allocation_pathological_weights_keep_reserve():
    b = allocate(1000, {"memory": 5, "documents": 5, "conversation": 5, "reserve": 1})
    assert b.total == 1000
    assert b.reserve >= 0


def test_fit_drops_lowest_relevance_first():
    items = [
        {"id": "a", "rel": 0.9, "sz": 40},
        {"id": "b", "rel": 0.2, "sz": 40},   # lowest relevance
        {"id": "c", "rel": 0.7, "sz": 40},
    ]
    kept = fit(items, 80, relevance=lambda x: x["rel"], size=lambda x: x["sz"])
    ids = {x["id"] for x in kept}
    assert ids == {"a", "c"}             # 'b' (lowest relevance) dropped


def test_compress_keeps_recent_verbatim_and_recalls_decisions():
    history = [
        {"role": "user", "content": "We will use PostgreSQL for storage."},
        {"role": "assistant", "content": "Sounds good."},
        {"role": "user", "content": "Also, must support offline mode."},
        {"role": "assistant", "content": "Noted."},
        {"role": "user", "content": "Now build the login screen."},
        {"role": "assistant", "content": "Here is the login screen…"},
        {"role": "user", "content": "Add a logout button."},
        {"role": "assistant", "content": "Added the logout button."},
    ]
    comp = ConversationCompressor(recent_keep=4).compress(history)
    # Old turns collapsed to a leading summary + the last 4 verbatim.
    assert comp[0]["role"] == "system"
    assert comp[1:] == history[-4:]
    summary = comp[0]["content"]
    assert "PostgreSQL" in summary       # decision recall (R13.3)
    assert "offline" in summary.lower()  # constraint recall


def test_compress_short_history_unchanged():
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    assert ConversationCompressor(recent_keep=4).compress(history) == history


def test_compress_falls_back_to_truncation_on_summarizer_error():
    history = [{"role": "user", "content": f"turn {i}"} for i in range(10)]

    def _boom(_old):
        raise RuntimeError("summarizer down")

    comp = ConversationCompressor(recent_keep=3).compress(history, summarize=_boom)
    assert comp[0]["role"] == "system"   # truncation fallback still produced a head
    assert comp[1:] == history[-3:]
