"""Sandbox-power batch (user asks 2026-07-09, items 1-4): entrypoint smoke
run, node syntax checks, sandbox-script archive CREATION, sandbox-script
archive EXTRACTION on upload, memory-safe 7z reading, and 7z export verify."""
from __future__ import annotations

import io
import zipfile

from app.verify.project_verify import (ProjectVerification, _entrypoint,
                                       _module_in_project,
                                       verify_project_files)


class TestSmokeRun:
    def test_entrypoint_selection(self):
        assert _entrypoint({"main.py": "", "util.py": ""}) == "main.py"
        assert _entrypoint({"src/app.py": "", "src/lib.py": "",
                            "app.py": ""}) == "app.py"      # shallower wins
        assert _entrypoint({"solo.py": ""}) == "solo.py"    # single .py
        assert _entrypoint({"a.py": "", "b.py": ""}) is None

    def test_module_in_project(self):
        files = {"mypkg/__init__.py": "", "util.py": ""}
        assert _module_in_project("mypkg", files)
        assert _module_in_project("util", files)
        assert not _module_in_project("flask", files)

    def test_runtime_crash_fails_project(self):
        v = verify_project_files({
            "main.py": "x = undefined_name  # NameError at import\n"})
        if v.backend == "" or "sandbox unavailable" in " ".join(v.skipped):
            return                                     # host has no sandbox
        assert v.smoke == "failed"
        assert v.status == "failed"
        assert any("runtime error" in f["error"] for f in v.failures)

    def test_clean_entrypoint_passes_smoke(self):
        v = verify_project_files({"main.py": "print('hello')\n"})
        if v.backend == "":
            return
        assert v.smoke in ("passed", "skipped")
        assert v.status == "verified"

    def test_missing_third_party_dep_is_skip_not_failure(self):
        v = verify_project_files({
            "main.py": "import flask_nonexistent_dep\nprint('up')\n"})
        if v.backend == "":
            return
        assert v.smoke == "skipped"
        assert v.status == "verified"
        assert any("third-party dependency" in s for s in v.skipped)

    def test_smoke_in_report(self):
        v = ProjectVerification(status="verified", smoke="passed")
        assert "smoke  : passed" in v.report_text()
        assert v.as_dict()["smoke"] == "passed"

    def test_smoke_failure_breaks_ok(self):
        v = ProjectVerification(status="failed", smoke="failed")
        assert not v.ok
        v2 = ProjectVerification(status="verified", smoke="long_running")
        assert v2.ok


class TestSandboxArchiveBuild:
    def test_zip_built_in_sandbox(self):
        from app.verify.archive_build import build_archive_sandboxed
        data = build_archive_sandboxed(
            "# Readme", [("src/main.py", "print(1)"), ("cfg.json", "{}")],
            "zip")
        if data is None:
            return                                     # sandbox unavailable
        names = set(zipfile.ZipFile(io.BytesIO(data)).namelist())
        assert {"README.md", "src/main.py", "cfg.json"} <= names

    def test_path_escape_not_staged(self):
        from app.verify.archive_build import build_archive_sandboxed
        data = build_archive_sandboxed(
            "", [("../../evil.py", "x"), ("ok.py", "y")], "zip")
        if data is None:
            return
        names = set(zipfile.ZipFile(io.BytesIO(data)).namelist())
        assert "ok.py" in names
        assert not any(".." in n for n in names)


class TestSandboxExtraction:
    def _zip(self, members: dict[str, str]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            for name, content in members.items():
                zf.writestr(name, content)
        return buf.getvalue()

    def test_upload_extracted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ZAPTHETRICK_WS_ROOT", str(tmp_path))
        from app.agent_workspace.materialize import materialize_archive
        res = materialize_archive(
            "conv-sbx-1", self._zip({"app.py": "print(1)",
                                     "docs/readme.md": "hi"}), "proj.zip")
        assert res.ok
        assert res.files == 2
        assert (tmp_path / "conv-sbx-1" / "app.py").is_file()
        assert (tmp_path / "conv-sbx-1" / "docs" / "readme.md").is_file()

    def test_zip_slip_still_blocked(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ZAPTHETRICK_WS_ROOT", str(tmp_path))
        from app.agent_workspace.materialize import materialize_archive
        res = materialize_archive(
            "conv-sbx-2",
            self._zip({"../evil.txt": "x", "good.txt": "y"}), "p.zip")
        assert res.files == 1
        assert res.skipped >= 1
        assert not (tmp_path / "evil.txt").exists()
        assert (tmp_path / "conv-sbx-2" / "good.txt").is_file()

    def test_driver_and_blob_not_left_behind(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ZAPTHETRICK_WS_ROOT", str(tmp_path))
        from app.agent_workspace.materialize import materialize_archive
        materialize_archive("conv-sbx-3", self._zip({"a.txt": "x"}), "p.zip")
        left = {p.name for p in (tmp_path / "conv-sbx-3").iterdir()}
        assert "_dtt_extract.py" not in left
        assert "_dtt_upload_archive.bin" not in left


class TestSevenZipSafety:
    def test_7z_verify_roundtrip(self):
        import py7zr
        from app.verify.project_loop import files_from_7z
        buf = io.BytesIO()
        with py7zr.SevenZipFile(buf, "w") as z:
            z.writestr("print('ok')\n", "main.py")
            z.writestr("{}", "cfg.json")
        files = files_from_7z(buf.getvalue())
        assert files == {"main.py": "print('ok')\n", "cfg.json": "{}"}

    def test_materialize_7z_preflight(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ZAPTHETRICK_WS_ROOT", str(tmp_path))
        import py7zr
        from app.agent_workspace.materialize import materialize_archive
        buf = io.BytesIO()
        with py7zr.SevenZipFile(buf, "w") as z:
            z.writestr("data", "keep.txt")
        res = materialize_archive("conv-7z", buf.getvalue(), "u.7z")
        assert res.ok and res.files == 1
