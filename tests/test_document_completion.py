"""Phase 6 — goal-completion detector (deterministic next-step suggestions)."""
from __future__ import annotations

import pytest

from app.documents.completion import (
    completion_report, detect_deliverables, detect_project_type,
    suggest_next_deliverables,
)

REST_API = """Here is a Spring Boot REST API.
```java
@RestController
class UserController { @GetMapping("/users") List<User> all() {} }
```
See the README to run it.
"""


@pytest.fixture(autouse=True)
def _deterministic_classifiers(monkeypatch):
    """Pin the DETERMINISTIC fallback. detect_project_type is SEMANTIC-first
    (gates.classify), warm in the full suite → valid-but-different classes than
    the regex these tests pin. The semantic mechanism is covered in
    test_semantic_gates; here we test the fallback, so disable the classifier."""
    import app.semantics.gates as _g
    monkeypatch.setattr(_g, "classify", lambda *a, **k: None)


class TestProjectType:
    @pytest.mark.parametrize("text,kind", [
        ("a @RestController spring boot endpoint", "rest_api"),
        ("a fastapi app with @app.get routes", "rest_api"),
        ("a react front-end web app", "web_app"),
        ("a CLI tool using argparse", "cli"),
        ("publish this library to pypi", "library"),
        ("explain how recursion works", "generic"),
    ])
    def test_detect(self, text, kind):
        assert detect_project_type(text) == kind


class TestDeliverableDetection:
    def test_detects_present(self):
        got = detect_deliverables(REST_API)
        assert "source_code" in got and "readme" in got

    def test_detects_tests_and_docker(self):
        txt = "```python\ndef test_it(): assert True\n```\nFROM python:3.12"
        got = detect_deliverables(txt)
        assert "tests" in got and "dockerfile" in got


class TestSuggestions:
    def test_rest_api_missing_deliverables(self):
        sugg = suggest_next_deliverables(REST_API)
        keys = {s.key for s in sugg}
        assert "tests" in keys and "dockerfile" in keys and "api_docs" in keys
        # source_code + readme are present → not suggested.
        assert "source_code" not in keys and "readme" not in keys

    def test_no_code_no_suggestions(self):
        # A plain Q&A must not nag about Dockerfiles.
        assert suggest_next_deliverables("what is a monad?") == []

    def test_complete_project_has_no_suggestions(self):
        full = (REST_API + "\n```python\ndef test_x(): assert 1\n```\n"
                "swagger openapi docs\nrequirements.txt\nFROM python:3.12")
        assert suggest_next_deliverables(full) == []

    def test_suggestions_carry_a_reason(self):
        for s in suggest_next_deliverables(REST_API):
            assert s.label and s.reason and s.key


class TestReport:
    def test_shape_and_percentage(self):
        rep = completion_report(REST_API)
        assert rep["project_type"] == "rest_api"
        assert "source_code" in rep["present"] and "tests" in rep["missing"]
        assert 0 < rep["completion_pct"] < 100
        assert isinstance(rep["suggestions"], list)

    def test_generic_qa_report(self):
        rep = completion_report("define entropy")
        assert rep["suggestions"] == []
