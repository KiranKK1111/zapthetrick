"""Live code verification helpers (app.live.code_run) + sandbox Java/Go."""
from __future__ import annotations

from app.live import code_run as cr
from app.sandbox.executor import _java_class, _lang_plan, run_code


# --------------------------------------------------------------------------- #
# Language selection
# --------------------------------------------------------------------------- #
def test_language_named_in_question_wins():
    assert cr.pick_language("write a java program to reverse a string", {}) == ("java", True)
    assert cr.pick_language("solve this in python", {"skills": ["Java"]}) == ("python", True)
    assert cr.language_in_question("implement it in Go") == "go"
    assert cr.language_in_question("reverse a string") == ""


def test_resume_language_when_question_silent():
    # first RUNNABLE resume language
    assert cr.pick_language("reverse a string", {"skills": ["Rust", "Python"]})[0] == "python"
    # only non-runnable resume langs → returned but not runnable
    lang, runnable = cr.pick_language("reverse a string", {"skills": ["Rust"]})
    assert lang == "rust" and runnable is False
    # nothing → default python
    assert cr.pick_language("reverse a string", {}) == ("python", True)


def test_resume_languages_from_skills_and_projects():
    prof = {"skills": ["Java", "SQL", "JavaScript"],
            "projects": [{"tech": ["Go", "React"]}]}
    langs = cr.resume_languages(prof)
    assert "java" in langs and "javascript" in langs and "go" in langs
    assert "sql" not in langs and "react" not in langs   # not languages we run


def test_normalize_aliases():
    assert cr.normalize_lang("py") == "python"
    assert cr.normalize_lang("node") == "javascript"
    assert cr.normalize_lang("golang") == "go"


# --------------------------------------------------------------------------- #
# Code extraction
# --------------------------------------------------------------------------- #
def test_extract_prefers_matching_language():
    ans = ("intro\n```python\nprint('py')\n```\nmid\n"
           "```java\nclass Main {}\n```\n")
    code, lang = cr.extract_code(ans, prefer_lang="java")
    assert lang == "java" and "class Main" in code
    code2, lang2 = cr.extract_code(ans, prefer_lang="python")
    assert lang2 == "python" and "print" in code2


def test_extract_none_when_no_fence():
    assert cr.extract_code("just prose, no code") == ("", "")


# --------------------------------------------------------------------------- #
# Sandbox compiled-language support (executor)
# --------------------------------------------------------------------------- #
def test_java_class_extraction():
    assert _java_class("public class PalindromeChecker {}") == "PalindromeChecker"
    assert _java_class("public final class Foo {}") == "Foo"
    assert _java_class("System.out.println(1);") == "Main"   # no class keyword


def test_lang_plans():
    assert _lang_plan("java", "public class Foo {}") == ("Foo.java",
                                                         [["javac", "Foo.java"], ["java", "Foo"]])
    assert _lang_plan("go", "package main") == ("main.go", [["go", "run", "main.go"]])
    assert _lang_plan("python", "x")[1] == [["python", "-I", "-B", "main.py"]] \
        or _lang_plan("python", "x")[0] == "main.py"   # interpreter path
    # cobol/racket/scala are now in the registry; an unknown language is None.
    assert _lang_plan("brainfuck", "x") is None


def test_python_runs_and_bad_python_fails():
    ok = run_code("print(2+2)", "python")
    assert ok.ok and ok.stdout.strip() == "4"
    bad = run_code("print(", "python")   # syntax error
    assert not bad.ok
