"""Reliability guard for prompt templating.

Prompts that embed a JSON example used to break under `str.format()` (literal
braces parsed as placeholders → KeyError → silent classifier fallback, which
killed difficulty-aware routing until a live test caught it). The fix is
`app.core.prompt.fill`, which substitutes only `{name}` tokens and leaves every
other brace untouched. These tests pin that behavior AND confirm each
JSON-bearing prompt renders correctly.
"""
from __future__ import annotations

from app.core.prompt import fill


# --- the fill primitive ---------------------------------------------------

def test_fill_substitutes_named_only():
    assert fill("hi {name}", name="Ada") == "hi Ada"


def test_fill_leaves_literal_json_untouched():
    t = 'Reply JSON: {"k": "v"} for {who}'
    assert fill(t, who="you") == 'Reply JSON: {"k": "v"} for you'


def test_fill_nested_json_untouched():
    t = '{"calls": [{"name": "x", "arguments": {...}}]} -> {q}'
    assert fill(t, q="?") == '{"calls": [{"name": "x", "arguments": {...}}]} -> ?'


def test_fill_value_with_braces_not_resubstituted():
    # A value containing another key's token must NOT be re-substituted (single pass).
    assert fill("{a} {b}", a="{b}", b="X") == "{b} X"


def test_fill_missing_key_never_raises():
    assert fill("{present} {absent}", present="ok") == "ok {absent}"


def test_fill_no_values_is_identity():
    assert fill('{"json": 1} and {x}') == '{"json": 1} and {x}'


# --- the real prompts render with single (real) JSON braces ---------------

def _has_real_json(s: str) -> bool:
    # The example survived as REAL braces (not doubled), proving no escaping is
    # needed anymore.
    return '{"' in s and "{{" not in s


def test_difficulty_prompt_renders():
    from app.chat.difficulty import _PROMPT
    out = fill(_PROMPT, text="some question")
    assert "some question" in out and '"difficulty"' in out and _has_real_json(out)


def test_verify_prompts_render():
    from app.chat.verify import _REVISE, _VERIFY
    v = fill(_VERIFY, q="q", a="draft")
    assert "draft" in v and '"correct"' in v and _has_real_json(v)
    assert "- bug" in fill(_REVISE, q="q", a="draft", c="- bug")


def test_grounder_prompt_renders():
    from app.agents.grounder import _PROMPT
    out = fill(_PROMPT, evidence="E", draft="D")
    assert "E" in out and "D" in out and '"unverified"' in out and _has_real_json(out)


def test_suggester_prompt_renders():
    from app.agents.suggester import _PROMPT
    out = fill(_PROMPT, turns="- a")
    assert "- a" in out and '"suggestions"' in out and _has_real_json(out)


def test_executor_prompt_renders():
    from app.tools.executor import _PROMPT
    out = fill(_PROMPT, catalog="- web_search — search", question="q")
    assert "web_search" in out and '"calls"' in out and _has_real_json(out)
