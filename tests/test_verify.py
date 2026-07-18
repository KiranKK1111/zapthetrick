"""Tests for the self-refine helpers (app/chat/verify) — pure parts only
(the draft→verify→revise loop itself needs the live LLM)."""
from __future__ import annotations

from app.chat.verify import _last_user, _parse_verdict, chunk_text


def test_parse_verdict_structured():
    assert _parse_verdict('{"correct": true, "problems": []}') == (True, [])
    ok, probs = _parse_verdict('{"correct": false, "problems": ["off-by-one", "wrong O()"]}')
    assert ok is False and probs == ["off-by-one", "wrong O()"]
    # correct=false but no problems listed → treat as correct (nothing to fix)
    assert _parse_verdict('{"correct": false, "problems": []}') == (True, [])
    # tolerates fences/prose
    assert _parse_verdict('```json\n{"correct": true}\n```') == (True, [])
    # unparseable → default correct (don't revise a good draft on a format slip)
    assert _parse_verdict("garbage") == (True, [])
    assert _parse_verdict("") == (True, [])
    assert _parse_verdict("[1,2]") == (True, [])  # not an object


def test_last_user_returns_latest_user_content():
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "the real question"},
    ]
    assert _last_user(msgs) == "the real question"
    assert _last_user([{"role": "system", "content": "x"}]) == ""
    assert _last_user([]) == ""


def test_chunk_text_reassembles_exactly():
    text = "line one\n" + "x" * 500 + "\nline three\n"
    pieces = list(chunk_text(text, size=80))
    assert "".join(pieces) == text          # lossless
    assert len(pieces) > 1                   # actually chunked
    assert all(len(p) <= 80 + 1 for p in pieces) or any("\n" in p for p in pieces)


def test_chunk_text_small_input_single_piece():
    assert list(chunk_text("hi", size=160)) == ["hi"]
    assert list(chunk_text("", size=160)) == []
