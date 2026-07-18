"""End-to-end project loop (app/verify): build artifacts are actually
verified + tested in the sandbox, repaired by the model on failure, and ship
with an honest VERIFICATION.txt. Real sandbox execution; model stubbed.
"""
from __future__ import annotations

import asyncio
import io
import zipfile

import pytest

from app.verify.project_loop import (parse_fenced_files,
                                     verify_and_repair_archive)
from app.verify.project_verify import files_from_zip, verify_project_files

_GOOD = {
    "app/main.py": "def add(a, b):\n    return a + b\n",
    "app/__init__.py": "",
    "config.json": '{"debug": false}\n',
    "tests/test_main.py": ("import sys; sys.path.insert(0, '.')\n"
                           "from app.main import add\n"
                           "def test_add():\n    assert add(2, 3) == 5\n"),
}
_BROKEN = {
    "app/main.py": "def add(a, b:\n    return a + b\n",     # SyntaxError
    "config.json": '{"debug": nope}\n',                      # bad JSON
}


def _zip_of(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


class TestVerifyProjectFiles:
    def test_good_project_verifies_and_tests_pass(self):
        v = verify_project_files(_GOOD)
        assert v.status == "verified", v.as_dict()
        assert v.checked >= 3
        assert v.tests == "passed", v.test_output
        assert v.ok

    def test_broken_project_fails_with_feedback(self):
        v = verify_project_files(_BROKEN)
        assert v.status == "failed"
        assert len(v.failures) == 2
        fb = v.repair_feedback()
        assert "app/main.py" in fb.replace("\\", "/")
        assert "config.json" in fb.replace("\\", "/")

    def test_failing_tests_fail_the_project(self):
        files = dict(_GOOD)
        files["tests/test_main.py"] = (
            "import sys; sys.path.insert(0, '.')\n"
            "from app.main import add\n"
            "def test_add():\n    assert add(2, 3) == 99\n")
        v = verify_project_files(files)
        assert v.tests == "failed" and v.status == "failed"

    def test_empty_input_skipped(self):
        assert verify_project_files({}).status == "skipped"

    def test_report_text_is_honest(self):
        v = verify_project_files(_BROKEN)
        rpt = v.report_text()
        assert "status : failed" in rpt
        assert "FAIL" in rpt


class TestArchiveLoop:
    def test_zip_roundtrip_adds_verification_report(self):
        data, meta = asyncio.run(verify_and_repair_archive(_zip_of(_GOOD)))
        assert meta is not None and meta["status"] == "verified"
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            names = zf.namelist()
            assert "VERIFICATION.txt" in names
            assert "app/main.py" in names
            assert b"status : verified" in zf.read("VERIFICATION.txt")

    def test_repair_round_fixes_broken_project(self, monkeypatch):
        # Stub the model: it returns the corrected files as fenced blocks.
        # Patch the module-level `llm` object itself (the loop imports it at
        # call time), which is robust against test-order effects.
        from types import SimpleNamespace

        async def fake_repair(messages, options=None, **kw):
            return ("Here you go:\n"
                    "```app/main.py\ndef add(a, b):\n    return a + b\n```\n"
                    "```config.json\n{\"debug\": false}\n```")

        monkeypatch.setattr("app.core.llm_client.llm",
                            SimpleNamespace(complete_routed=fake_repair))
        data, meta = asyncio.run(verify_and_repair_archive(_zip_of(_BROKEN)))
        assert meta is not None
        assert meta["repaired"] is True
        assert meta["status"] == "verified"
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            assert b"def add(a, b):" in zf.read("app/main.py")
            assert b"repair round(s) were applied" in zf.read("VERIFICATION.txt")

    def test_unrepairable_ships_honest_failure(self, monkeypatch):
        from types import SimpleNamespace

        async def no_help(messages, options=None, **kw):
            return "sorry, no fences here"

        monkeypatch.setattr("app.core.llm_client.llm",
                            SimpleNamespace(complete_routed=no_help))
        data, meta = asyncio.run(verify_and_repair_archive(_zip_of(_BROKEN)))
        assert meta is not None and meta["status"] == "failed"
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            assert b"status : failed" in zf.read("VERIFICATION.txt")

    def test_flag_off_is_passthrough(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.artifact_validation, "verify_projects", False)
        raw = _zip_of(_GOOD)
        data, meta = asyncio.run(verify_and_repair_archive(raw))
        assert data == raw and meta is None

    def test_non_project_zip_passthrough(self):
        raw = _zip_of({})            # empty archive → nothing to verify
        data, meta = asyncio.run(verify_and_repair_archive(raw))
        assert meta is None


class TestHelpers:
    def test_parse_fenced_files(self):
        text = ("```app/x.py\nprint(1)\n```\nnoise\n"
                "```lib/util.js\nlet a = 1\n```\n"
                "```../evil.py\nboom\n```")          # traversal rejected
        files = parse_fenced_files(text)
        assert set(files) == {"app/x.py", "lib/util.js"}

    def test_files_from_zip_skips_binary(self):
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("a.py", "print(1)")
            zf.writestr("img.png", b"\x89PNG\x00\xff\xfe binary")
        files = files_from_zip(buf.getvalue())
        assert "a.py" in files and "img.png" not in files
