"""Dedicated execution sandbox (app/sandbox) — layered isolation.

Real execution runs at whatever level this host offers (subprocess on the
Windows dev box, rlimit/namespace on Linux CI/VPS); the bubblewrap argv is a
pure function so the namespace level is verified structurally everywhere.
"""
from __future__ import annotations

import asyncio
import os

import pytest

from app.sandbox import executor as sbx


class TestIsolationProbe:
    def test_level_is_known_value(self):
        assert sbx.isolation_level(refresh=True) in (
            "namespace", "rlimit", "subprocess")

    def test_windows_is_subprocess(self):
        if os.name == "nt":
            assert sbx.isolation_level(refresh=True) == "subprocess"

    def test_capability_registry_reports_isolation(self):
        from app.capabilities import registry as caps
        snap = caps.refresh()
        assert snap["sandbox"]["available"] is True
        assert snap["sandbox"]["isolation"] == sbx.isolation_level()


class TestBwrapArgv:
    def test_namespace_shape(self):
        argv = sbx.build_bwrap_argv("/tmp/ws", ["python3", "main.py"])
        s = " ".join(argv)
        assert "--unshare-all" in s          # includes network unshare
        assert "--clearenv" in s
        assert "--die-with-parent" in s
        assert ["--bind", "/tmp/ws", "/work"] == argv[
            argv.index("--bind"):argv.index("--bind") + 3]
        assert argv[-2:] == ["python3", "main.py"][-2:]
        assert "--chdir" in argv and "/work" in argv


class TestRunCode:
    def test_python_ok(self):
        r = sbx.run_code("print('hello sandbox')", "python")
        assert r.ok and r.exit_code == 0
        assert "hello sandbox" in r.stdout
        assert r.backend in ("namespace", "rlimit", "subprocess")

    def test_python_error_propagates(self):
        r = sbx.run_code("raise SystemExit(3)", "python")
        assert r.status == "error" and r.exit_code == 3

    def test_python_exception_is_error(self):
        r = sbx.run_code("boom(", "python")
        assert r.status == "error"
        assert "SyntaxError" in r.stderr

    def test_timeout_kills(self):
        r = sbx.run_code("while True:\n    pass",
                         "python",
                         limits=sbx.SandboxLimits(timeout_s=1.5))
        assert r.status == "timeout"
        assert r.duration_ms < 15_000       # tree-kill actually worked

    def test_staged_files_visible(self):
        r = sbx.run_code(
            "print(open('data/input.txt').read().strip())", "python",
            files={"data/input.txt": "42"})
        assert r.ok and r.stdout.strip() == "42"

    def test_env_is_scrubbed(self):
        os.environ["DTT_SECRET_PROBE"] = "leak-me"
        try:
            r = sbx.run_code(
                "import os; print(os.environ.get('DTT_SECRET_PROBE'))",
                "python")
            assert r.ok and r.stdout.strip() == "None"
        finally:
            os.environ.pop("DTT_SECRET_PROBE", None)

    def test_output_capped(self):
        r = sbx.run_code("print('x' * 1_000_000)", "python",
                         limits=sbx.SandboxLimits(output_kb=8))
        assert len(r.stdout) <= 8 * 1024 + 16

    def test_unknown_language_unavailable(self):
        r = sbx.run_code("puts 'hi'", "cobol")
        assert r.status == "unavailable"

    def test_disabled_by_config(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.sandbox, "enabled", False)
        r = sbx.run_code("print(1)", "python")
        assert r.status == "unavailable"


class TestVerifyScript:
    def test_verified_with_expected_output(self):
        r = sbx.verify_script("print(sum(range(5)))", "python",
                              expected_stdout="10")
        assert r.ok

    def test_output_mismatch_fails(self):
        r = sbx.verify_script("print(11)", "python", expected_stdout="10")
        assert r.status == "error" and "mismatch" in r.reason

    def test_orchestration_bridge(self):
        from app.orchestration.sandbox import verify_snippet
        res = asyncio.run(verify_snippet("print('ok')", "python"))
        assert res.is_verified and res.status == "verified"
        bad = asyncio.run(verify_snippet("raise ValueError('nope')", "python"))
        assert not bad.is_verified and bad.status == "failed"
        assert "ValueError" in bad.repair_feedback
