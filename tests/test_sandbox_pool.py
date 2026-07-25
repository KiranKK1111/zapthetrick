"""Stage-4 §3.1 Component A — warm sandbox pool + local compile-once.

Two independent wins:
  * `app/sandbox/pool.py` hands out pre-created workspaces and DESTROYS a used
    one (never reused) — the throw-the-world-away guarantee is preserved;
  * `executor.run_batch` on the LOCAL backend now compiles ONCE and runs many
    (previously it looped `run_code`, recompiling per stdin).

Both are additive + fail-open: pool OFF → cold `mkdtemp`, byte-identical.
"""
from __future__ import annotations

import os

import pytest

from app.sandbox import executor as sbx
from app.sandbox import pool as sbxpool


@pytest.fixture
def local_backend(monkeypatch):
    """Force the local backend (config.yaml ships backend: docker)."""
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.sandbox, "backend", "local", raising=False)
    monkeypatch.setattr(cfg.sandbox, "enabled", True, raising=False)
    monkeypatch.setattr(cfg.sandbox, "languages", [], raising=False)
    yield


@pytest.fixture(autouse=True)
def _fresh_pool():
    """Each test gets a clean singleton (and no leaked warm dirs)."""
    sbxpool.reset_pool()
    yield
    sbxpool.reset_pool()


# --------------------------------------------------------------------------- #
# The pool
# --------------------------------------------------------------------------- #
class TestSandboxPool:
    def test_acquire_returns_ready_workspace(self):
        p = sbxpool.SandboxPool(size=2)
        try:
            ws = p.acquire()
            assert os.path.isdir(ws)
        finally:
            p.close()

    def test_release_destroys_and_never_reuses(self):
        p = sbxpool.SandboxPool(size=2)
        try:
            handed_out: list[str] = []
            for _ in range(10):
                ws = p.acquire()
                handed_out.append(ws)
                assert os.path.isdir(ws)
                p.release(ws)
                assert not os.path.exists(ws)     # destroyed, thrown away
            # No workspace path was ever handed out twice (never reused).
            assert len(set(handed_out)) == len(handed_out)
            assert p.stats()["destroyed"] == 10
        finally:
            p.close()

    def test_refills_in_background(self):
        import time
        p = sbxpool.SandboxPool(size=3)
        try:
            # Drain the ready set, then let the daemon refiller top it back up.
            for _ in range(3):
                p.release(p.acquire())
            deadline = time.monotonic() + 3.0
            while p.ready() < 3 and time.monotonic() < deadline:
                time.sleep(0.05)
            assert p.ready() >= 3
        finally:
            p.close()

    def test_size_zero_is_functional(self):
        p = sbxpool.SandboxPool(size=0)
        try:
            ws = p.acquire()          # creates on the spot (pool miss)
            assert os.path.isdir(ws)
            p.release(ws)
            assert not os.path.exists(ws)
        finally:
            p.close()

    def test_close_is_idempotent_and_clears_ready(self):
        p = sbxpool.SandboxPool(size=2)
        p.close()
        p.close()                      # no raise
        assert p.ready() == 0

    def test_pool_disabled_by_default(self):
        assert sbxpool.pool_enabled() is False   # config default OFF


# --------------------------------------------------------------------------- #
# Local run_batch: compile once, run many
# --------------------------------------------------------------------------- #
class TestLocalRunBatchCompileOnce:
    def test_compiles_once_runs_many(self, local_backend, monkeypatch):
        """A fake 2-command plan proves the compile step fires ONCE while the run
        step fires per stdin — the core compile-once-run-many win."""
        calls: list[tuple[list[str], str | None]] = []

        def fake_plan(lang, code):
            return ("main.x", [["cc", "main.x"], ["run", "main.x"]])

        def fake_run_command(argv, **kw):
            calls.append((list(argv), kw.get("stdin_data")))
            return sbx.SandboxResult(status="ok", exit_code=0, stdout="out")

        monkeypatch.setattr(sbx, "_lang_plan", fake_plan)
        monkeypatch.setattr(sbx, "run_command", fake_run_command)

        results = sbx.run_batch("print(1)", "fakelang",
                                stdins=["a", "b", "c"])

        assert len(results) == 3
        assert all(r.ok for r in results)
        compiles = [a for a, _ in calls if a == ["cc", "main.x"]]
        runs = [(a, s) for a, s in calls if a == ["run", "main.x"]]
        assert len(compiles) == 1                      # compiled ONCE
        assert [s for _, s in runs] == ["a", "b", "c"]  # ran per stdin, in order

    def test_compile_failure_broadcasts_distinct_copies(self, local_backend,
                                                         monkeypatch):
        def fake_plan(lang, code):
            return ("main.x", [["cc", "main.x"], ["run", "main.x"]])

        def fake_run_command(argv, **kw):
            if argv == ["cc", "main.x"]:
                return sbx.SandboxResult(status="error", exit_code=1,
                                         stderr="compile boom")
            raise AssertionError("run must not fire after a compile failure")

        monkeypatch.setattr(sbx, "_lang_plan", fake_plan)
        monkeypatch.setattr(sbx, "run_command", fake_run_command)

        results = sbx.run_batch("print(1)", "fakelang", stdins=["a", "b"])
        assert len(results) == 2
        assert all(r.status == "error" and "compile boom" in r.stderr
                   for r in results)
        # Distinct objects — mutating one must not alias the rest.
        results[0].reason = "touched"
        assert results[1].reason != "touched"

    def test_correct_results_real_python(self, local_backend):
        """End-to-end on the always-present python runtime: each stdin yields its
        own upper-cased echo (proves the shared workspace runs each input)."""
        code = "import sys; sys.stdout.write(sys.stdin.read().strip().upper())"
        results = sbx.run_batch(code, "python", stdins=["ab", "cd", "ef"])
        assert len(results) == 3
        assert [r.stdout.strip() for r in results] == ["AB", "CD", "EF"]

    def test_empty_stdins_returns_empty(self, local_backend):
        assert sbx.run_batch("print(1)", "python", stdins=[]) == []


# --------------------------------------------------------------------------- #
# Pool wired into the executor
# --------------------------------------------------------------------------- #
class TestPoolWiredIntoExecutor:
    def test_warm_pool_run_code_python(self, local_backend, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.sandbox, "warm_pool", True, raising=False)
        monkeypatch.setattr(cfg.sandbox, "pool_size", 2, raising=False)
        res = sbx.run_code("print('hi from pool')", "python")
        assert res.ok
        assert "hi from pool" in res.stdout

    def test_acquire_ws_reports_pool_source(self, monkeypatch):
        from app.core.config_loader import cfg
        # OFF → cold mkdtemp
        monkeypatch.setattr(cfg.sandbox, "warm_pool", False, raising=False)
        ws, from_pool = sbx._acquire_ws()
        assert from_pool is False and os.path.isdir(ws)
        sbx._release_ws(ws, from_pool)
        assert not os.path.exists(ws)
        # ON → pooled
        monkeypatch.setattr(cfg.sandbox, "warm_pool", True, raising=False)
        ws2, from_pool2 = sbx._acquire_ws()
        assert from_pool2 is True and os.path.isdir(ws2)
        sbx._release_ws(ws2, from_pool2)
        assert not os.path.exists(ws2)
