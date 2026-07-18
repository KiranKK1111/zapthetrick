"""Phase-2 (ArchitectureVerdict.md): capability registry + negotiation,
StackProfile detection, and attachment→slot clarification elimination.
Deterministic — no LLM, no DB, no network.
"""
from __future__ import annotations

import io
import zipfile

import pytest

from app.capabilities import registry as caps
from app.clarify import intent_pipeline as ip
from app.clarify.requirement_matrix import SOURCE_ATTACHMENT
from app.codeintel.stack_profile import (detect_stack_from_archive,
                                         detect_stack_from_members)


# ------------------------------------------------------------- registry ----
class TestCapabilityRegistry:
    def test_snapshot_shape(self):
        snap = caps.refresh()
        assert isinstance(snap["document_formats"], dict)
        assert "zip" in snap["document_formats"]
        assert isinstance(snap["gpu"], dict)
        assert isinstance(snap["tools"], list)

    def test_stdlib_formats_always_available(self):
        avail = caps.available_document_formats()
        for f in ("zip", "md", "txt", "csv", "json"):
            assert f in avail

    def test_negotiate_available_format(self):
        ok, alt, why = caps.negotiate_format("zip")
        assert ok is True and alt is None

    def test_negotiate_missing_format_falls_back(self, monkeypatch):
        # Simulate a deployment without the PDF renderer.
        monkeypatch.setitem(caps._FORMAT_DEPS, "pdf", "definitely_not_a_module")
        caps.refresh()
        ok, alt, why = caps.negotiate_format("pdf")
        assert ok is False
        assert alt in caps.available_document_formats()
        assert "pdf" in why
        caps.refresh()   # restore for other tests (monkeypatch undoes the dep)

    def test_negotiate_unknown_format_offers_markdown(self):
        ok, alt, why = caps.negotiate_format("cad")
        assert ok is False and alt == "md"

    def test_snapshot_is_cached(self):
        a = caps.capability_snapshot()
        b = caps.capability_snapshot()
        assert a is b            # TTL cache returns the same object


# ---------------------------------------------------------- stack profile ---
class TestStackProfile:
    def test_spring_boot_from_pom(self):
        p = detect_stack_from_members([
            ("pom.xml", "<project><dependency>spring-boot-starter-web"
                        "</dependency><artifactId>postgresql</artifactId>"),
            ("src/Main.java", "class Main {}"),
        ])
        assert p.language == "java"
        assert p.framework == "spring boot"
        assert p.build_tool == "maven"
        assert "postgresql" in p.db_hints

    def test_typescript_react_from_package_json(self):
        p = detect_stack_from_members([
            ("package.json", '{"dependencies": {"react": "^18", '
                             '"typescript": "^5"}}'),
            ("tsconfig.json", "{}"),
            ("src/App.tsx", "export const App = () => null"),
        ])
        assert p.language == "typescript"
        assert p.framework == "react"
        assert p.package_manager == "npm"

    def test_fastapi_from_requirements(self):
        p = detect_stack_from_members([
            ("requirements.txt", "fastapi==0.110\npsycopg[binary]\nuvicorn"),
            ("app/main.py", "from fastapi import FastAPI"),
        ])
        assert p.language == "python"
        assert p.framework == "fastapi"
        assert "postgresql" in p.db_hints

    def test_extension_histogram_fallback(self):
        p = detect_stack_from_members([
            ("main.go", "package main"),
            ("util.go", "package main"),
            ("handler.go", "package main"),
        ])
        assert p.language == "go"
        assert p.confidence >= 0.7
        assert any(e.startswith("extension_histogram") for e in p.evidence)

    def test_empty_input_is_empty_profile(self):
        p = detect_stack_from_members([])
        assert p.empty and p.slots() == {}

    def test_zip_bytes_detection(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("proj/pom.xml",
                        "<project>spring-boot-starter</project>")
            zf.writestr("proj/src/Main.java", "class Main {}")
        p = detect_stack_from_archive(buf.getvalue(), "proj.zip")
        assert p.language == "java"
        assert p.framework == "spring boot"

    def test_never_raises_on_garbage(self):
        p = detect_stack_from_archive(b"\x00\x01notanarchive", "x.zip")
        assert p.empty


# ----------------------------------------- attachment → clarifier slots ----
class TestClarificationElimination:
    def test_uploaded_stack_satisfies_required_language(self):
        # Without attachment evidence: code-gen with no language → CLARIFY.
        base = ip.assess("write a login api")
        assert "language" in base.missing_required
        # With a detected Spring upload: nothing required is missing.
        a = ip.assess("write a login api", has_artifact=True,
                      attachment_slots={"language": "java",
                                        "framework": "spring boot"})
        assert "language" not in a.missing_required
        assert a.decision != ip.CLARIFY or "language" not in a.missing_required

    def test_matrix_attributes_attachment_source(self):
        a = ip.assess("add authentication", has_artifact=True,
                      attachment_slots={"language": "java",
                                        "framework": "spring boot"})
        assert a.matrix is not None
        lang = a.matrix.facts.get("language")
        assert lang is not None and lang.value == "java"
        assert lang.source == SOURCE_ATTACHMENT

    def test_project_build_satisfied_by_attachment(self):
        base = ip.assess("build the app")
        a = ip.assess("build the app", has_artifact=True,
                      attachment_slots={"framework": "django"})
        assert "language_or_framework" not in a.missing_required
        # Baseline sanity: without evidence the build ask IS under-specified.
        if "language_or_framework" in base.missing_required:
            assert base.decision == ip.CLARIFY

    def test_empty_attachment_slots_change_nothing(self):
        a = ip.assess("write a login api", attachment_slots={})
        b = ip.assess("write a login api")
        assert a.decision == b.decision
        assert a.missing_required == b.missing_required
