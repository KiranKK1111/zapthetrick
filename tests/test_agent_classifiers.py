"""Tests for the LLM-output PARSERS of the de-hardcoded classifiers.

The classifiers themselves make a model call (not unit-testable offline), but the
parse + tolerance + safe-fallback logic is pure and is exactly where a flaky
model reply would otherwise cause a crash or a wrong flag. These lock that down.
"""
from __future__ import annotations

import pytest

from app.agents.grounder import _parse_unverified
from app.agents.suggester import _parse_suggestions
from app.documents.detect import _parse


# --- grounder: unverified-claims parsing ---------------------------------

def test_grounder_parses_list():
    assert _parse_unverified('{"unverified": ["a", "b"]}') == ["a", "b"]


def test_grounder_tolerates_fences_and_prose():
    raw = 'Here you go:\n```json\n{"unverified": ["x"]}\n```'
    assert _parse_unverified(raw) == ["x"]


def test_grounder_empty_and_garbage_are_safe():
    assert _parse_unverified("") == []
    assert _parse_unverified("not json at all") == []
    assert _parse_unverified('{"unverified": "oops not a list"}') == []
    assert _parse_unverified('{"other": 1}') == []


def test_grounder_drops_blank_items():
    assert _parse_unverified('{"unverified": ["a", "", "  ", "b"]}') == ["a", "b"]


# --- document intent parsing ---------------------------------------------

def test_detect_parse_basic_and_fenced():
    assert _parse('{"document": true, "format": "pdf"}') == {
        "document": True, "format": "pdf"}
    assert _parse('```\n{"document": false}\n```') == {"document": False}


def test_detect_parse_garbage_is_empty_dict():
    assert _parse("") == {}
    assert _parse("nope") == {}
    assert _parse("[1,2,3]") == {}  # not an object


# --- explicit (deterministic) document-request detection -----------------

def test_explicit_doc_request_positive():
    from app.documents.detect import explicit_doc_request as d
    assert d("give me a pdf document") == (True, "pdf")
    assert d("give me a pdf") == (True, "pdf")
    assert d("can you give me a pdf document") == (True, "pdf")
    assert d("export this as excel") == (True, "xlsx")
    assert d("make me a word document") == (True, "docx")
    assert d("I want a word doc") == (True, "docx")
    assert d("put it in pdf format") == (True, "pdf")
    assert d("save as docx") == (True, "docx")
    assert d("create an excel spreadsheet") == (True, "xlsx")
    assert d("give me a json file") == (True, "json")


def test_explicit_doc_request_zip():
    from app.documents.detect import explicit_doc_request as d
    assert d("zip the whole project") == (True, "zip")
    assert d("get me this project in a zip file") == (True, "zip")
    assert d("download the whole codebase") == (True, "zip")


def test_explicit_doc_request_negatives_defer_to_llm():
    # No deterministic opinion → (False, None) so the LLM classifier decides.
    from app.documents.detect import explicit_doc_request as d
    assert d("how do I parse a PDF in python") == (False, None)
    assert d("what is the difference between json and yaml") == (False, None)
    assert d("explain how excel formulas work") == (False, None)
    assert d("write a function to read a csv") == (False, None)
    assert d("give me an example of a python class") == (False, None)
    assert d("") == (False, None)


def test_explicit_doc_formats_multiple():
    from app.documents.detect import explicit_doc_formats as f
    assert f("get me a text and markdown documents") == ["txt", "md"]
    assert f("give me a pdf and a word doc") == ["pdf", "docx"]
    assert f("export as excel and csv") == ["xlsx", "csv"]
    assert f("give me a markdown and pdf") == ["md", "pdf"]
    # Single format → single-element list.
    assert f("give me a pdf document") == ["pdf"]
    # Not a document request → empty (defer to LLM).
    assert f("how do I parse a pdf in python") == []
    assert f("summarize this text") == []


def test_explicit_code_file_request():
    from app.documents.detect import explicit_doc_request as d
    assert d("give me a python file") == (True, "py")
    assert d("give me a java file") == (True, "java")
    assert d("write me a rust file") == (True, "rs")
    assert d("create a .py file") == (True, "py")
    # No language named → "code" (resolved from the answer later).
    assert d("give me this code file") == (True, "code")
    # Code questions that are NOT file requests.
    assert d("give me an example of a python class") == (False, None)
    assert d("write python code to sort a list") == (False, None)
    assert d("the code file is broken") == (False, None)


def test_infer_code_ext():
    from app.documents.detect import infer_code_ext as e
    assert e("```python\nprint(1)\n```") == "py"
    assert e("```java\nclass X{}\n```") == "java"
    assert e("no code here") == "txt"


# --- suggester parsing ----------------------------------------------------

def test_suggester_parses_and_caps_at_three():
    raw = '{"suggestions": ["a", "b", "c", "d", "e"]}'
    assert _parse_suggestions(raw) == ["a", "b", "c"]


def test_suggester_empty_and_garbage_are_safe():
    assert _parse_suggestions("") == []
    assert _parse_suggestions("garbage") == []
    assert _parse_suggestions('{"suggestions": "not a list"}') == []


def test_suggester_strips_blanks():
    assert _parse_suggestions('{"suggestions": [" hi ", "", "  "]}') == ["hi"]


# --- tool executor parsing ------------------------------------------------

def test_tool_executor_parse_calls():
    from app.tools.executor import _parse_calls
    raw = '{"calls": [{"name": "web_search", "arguments": {"query": "x"}}]}'
    assert _parse_calls(raw) == [{"name": "web_search", "arguments": {"query": "x"}}]
    # tolerates fences/prose, missing args, garbage
    assert _parse_calls('```json\n{"calls": [{"name": "code_search"}]}\n```') == [
        {"name": "code_search", "arguments": {}}]
    assert _parse_calls("") == []
    assert _parse_calls("not json") == []
    assert _parse_calls('{"calls": "nope"}') == []
    assert _parse_calls('{"calls": [{"no_name": 1}]}') == []


def test_tool_executor_arg_filtering():
    from app.tools.executor import _resolve_args
    # web_search-like schema (no conversation_id) must NOT receive it.
    web = {"properties": {"query": {}, "max_results": {}}}
    assert _resolve_args(web, {"query": "x"}, {"conversation_id": "c1"}) == {"query": "x"}
    # code-tool schema declares conversation_id → it's injected (context wins).
    code = {"properties": {"conversation_id": {}, "symbol": {}}}
    assert _resolve_args(code, {"symbol": "foo", "conversation_id": "wrong"},
                         {"conversation_id": "c1"}) == {"symbol": "foo", "conversation_id": "c1"}
    # undeclared model args are dropped (handlers take no **kwargs).
    assert _resolve_args(web, {"query": "x", "bogus": 1}, None) == {"query": "x"}
