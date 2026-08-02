"""The verifier must not call a well-formatted answer gibberish.

From a real deployed session: a correct, well-structured answer comparing
`@Controller` and `@RestController` was scored

    verify: {relevance: 0.0, hallucination_risk: 1.0, gibberish: true,
             issue: garbled / incoherent output}

and regenerated. The user saw the same answer twice, and the turn cost two full
generations — most of the 42 s latency recorded on it.

The cause was not semantic. A markdown table separator,
`|--------|---------------------------|------------------|`, is 57 characters
with no spaces, which tripped a "word >= 50 chars means mush" heuristic. So
every answer containing a comparison table was condemned for being well
formatted.

Both directions matter: tables must pass, and genuine garbage must still be
caught — a verifier that never fires is as useless as one that always does.
"""
from __future__ import annotations

import pytest

from app.live.verify import looks_incoherent

TABLE_ANSWER = """RestController composes Controller and ResponseBody.

| Aspect | Controller (Spring MVC) | RestController |
|--------|---------------------------|------------------|
| Meta-annotation | @Controller + @RequestMapping | @Controller + @ResponseBody |
| Response handling | Must add @ResponseBody per method | Written straight to the body |
| Typical use-case | Server-rendered pages | JSON REST APIs |

Use RestController for JSON-first APIs and keep Controller for view rendering.
"""


# ── Well-formatted answers are not garbage ──────────────────────────────────

def test_the_exact_answer_from_the_live_log_is_not_gibberish():
    assert looks_incoherent(TABLE_ANSWER) is False


@pytest.mark.parametrize("answer", [
    "A hash map is O(1) average.\n\n| k | v |\n|---|---|\n| a | 1 |\n\nThat is the shape.",
    "Here is the code:\n\n```java\n@RestController\nclass C {}\n```\n\nIt returns JSON.",
    "First point.\n\n---\n\nSecond point, separated by a horizontal rule.",
    "See https://docs.spring.io/spring-framework/reference/web/webmvc.html for the details of view resolution.",
    "Use `@RequestMapping(\"/very/long/path/segment/that/is/lengthy\")` on the class.",
])
def test_legitimate_markdown_structure_passes(answer):
    """Tables, fences, rules, URLs and inline code all contain long space-free
    runs by nature. None of them says anything about coherence."""
    assert looks_incoherent(answer) is False


# ── Genuine garbage is still caught ─────────────────────────────────────────

@pytest.mark.parametrize("answer,why", [
    ("", "empty"),
    ("   ", "whitespace only"),
    ("the <unk> <unk> answer here", "unknown tokens"),
    ("bad \ufffd\ufffd\ufffd output", "replacement characters"),
    (" ".join(["spring"] * 40), "runaway repetition"),
    ("asdkjfhaskdjfhaskjdfhaskjdhfaskjdhfaskjdhfaskjdhfaskjdfhaksjdhfkasjdhf",
     "one enormous mashed token"),
])
def test_real_garbage_is_still_flagged(answer, why):
    assert looks_incoherent(answer) is True, f"missed: {why}"


def test_structure_with_no_prose_at_all_is_still_flagged():
    """A bare table with no sentence around it IS a broken answer — stripping
    structure must not become a way to smuggle emptiness through."""
    assert looks_incoherent("|---|---|\n|---|---|\n---\n```\n```") is True


def test_a_dense_wall_of_text_is_still_flagged():
    """The whitespace-ratio check must survive on prose."""
    assert looks_incoherent("word" * 200) is True


def test_a_long_well_spaced_answer_passes():
    """Length alone is not a defect — the fix must not simply stop firing."""
    body = ("Spring resolves the return value through message converters. " * 12)
    assert looks_incoherent(body) is False
