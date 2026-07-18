"""Semantic / block streaming + progressive artifacts (P6 #6/#18)."""
from __future__ import annotations

from app.response_arch.blocks import BlockAssembler, classify_block


def _drip(asm, text, step=7):
    """Feed text in small chunks to simulate token streaming."""
    out = []
    for i in range(0, len(text), step):
        out += asm.feed(text[i:i + step])
    out += asm.flush()
    return out


def test_paragraphs_split_on_blank_line():
    asm = BlockAssembler()
    blocks = _drip(asm, "First para.\n\nSecond para.\n\nThird.")
    texts = [b.text for b in blocks]
    assert texts == ["First para.", "Second para.", "Third."]
    assert all(b.type == "paragraph" for b in blocks)
    # ids are stable + monotonic
    assert [b.id for b in blocks] == [1, 2, 3]


def test_open_paragraph_not_emitted_until_terminated():
    asm = BlockAssembler()
    got = asm.feed("still typing without a blank line")
    assert got == []                      # nothing complete yet
    got = asm.flush()
    assert len(got) == 1 and got[0].text.startswith("still typing")


def test_code_block_is_atomic_and_closed():
    asm = BlockAssembler()
    text = "Here is code:\n\n```python\nprint('hi')\n```\n\nDone."
    blocks = _drip(asm, text)
    kinds = [(b.type, b.closed) for b in blocks]
    assert ("code", True) in kinds
    code = [b for b in blocks if b.type == "code"][0]
    assert code.language == "python"
    assert "print('hi')" in code.text


def test_half_open_fence_waits_then_flushes_unclosed():
    asm = BlockAssembler()
    # opening fence but never closed within feed
    got = asm.feed("```js\nconst x = 1;\n")
    assert got == []                      # held — no closing fence yet
    flushed = asm.flush()
    assert len(flushed) == 1
    assert flushed[0].type == "code" and flushed[0].closed is False


def test_progressive_artifact_on_code_close():
    asm = BlockAssembler(emit_artifacts=True)
    text = ("```python name=sort.py\n"
            "def sort(x):\n    return sorted(x)\n```\n\nExplanation here.")
    _drip(asm, text)
    assert len(asm.artifacts) == 1
    art = asm.artifacts[0]
    assert art.filename == "sort.py"
    assert "def sort" in art.content


def test_no_artifact_when_disabled():
    asm = BlockAssembler(emit_artifacts=False)
    _drip(asm, "```python name=a.py\nx=1\n```\n\n")
    assert asm.artifacts == []


def test_classify_block():
    assert classify_block("# Title") == "heading"
    assert classify_block("- a\n- b") == "list"
    assert classify_block("| a | b |\n| --- | --- |\n| 1 | 2 |") == "table"
    assert classify_block("just words") == "paragraph"


def test_feed_and_flush_never_raise():
    asm = BlockAssembler()
    assert asm.feed("") == []
    assert asm.flush() == []
