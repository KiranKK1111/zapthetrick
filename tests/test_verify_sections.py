"""Per-section incremental verification while streaming (P6 #9)."""
from __future__ import annotations

from app.chat.verify import assess_partial_sections
from app.quality.stream_controller import CONTINUE, REGENERATE


def test_clean_multi_section_answer_continues():
    text = ("Kafka is a distributed log.\n\n"
            "It uses partitions for parallelism.\n\n"
            "Consumers track offsets.")
    r = assess_partial_sections(text)
    assert r["action"] == CONTINUE
    assert len(r["sections"]) == 3


def test_one_bad_section_flags_whole_turn():
    good = "Here is a solid explanation of the concept and how it works well."
    # a degenerate repeated-token section hides inside an otherwise fine answer
    bad = ("loop " * 30).strip()
    r = assess_partial_sections(f"{good}\n\n{bad}")
    assert r["action"] == REGENERATE
    # the worst section is attributed
    worst = [s for s in r["sections"] if s["action"] == REGENERATE]
    assert worst and any("repetition" in x for s in worst for x in s["reasons"])


def test_code_section_not_flagged_for_error_words():
    text = ("Handle failures gracefully.\n\n"
            "```python\ntry:\n    run()\nexcept Exception as e:\n"
            "    log.error('failed')\n```")
    r = assess_partial_sections(text)
    # the code block trips 'error'/'exception'/'failed' words but must not flag
    assert r["action"] == CONTINUE
    code = [s for s in r["sections"] if s["type"] == "code"]
    assert code and code[0]["action"] == CONTINUE


def test_empty_text_fails_open():
    r = assess_partial_sections("")
    assert "action" in r and "sections" in r
