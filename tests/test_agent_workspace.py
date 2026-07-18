"""Phase 0 — build-system detection (#22) + constrained runner (#7).

Pure/offline: detection is deterministic over marker files; the runner uses the
current Python interpreter so it works on any OS without extra toolchains.
"""
from __future__ import annotations

import asyncio
import sys

from app.agent_workspace import (
    detect_build_system,
    detect_build_systems,
    run_in_workspace,
)


# --------------------------------------------------------------------------
# build-system detection
# --------------------------------------------------------------------------
def test_detect_python_requirements(tmp_path):
    (tmp_path / "requirements.txt").write_text("pytest\n")
    bs = detect_build_system(str(tmp_path))
    assert bs is not None
    assert bs.name == "python"
    assert bs.test == "pytest -q"
    assert bs.install == "pip install -r requirements.txt"


def test_detect_python_pyproject_with_ruff(tmp_path):
    (tmp_path / "pyproject.toml").write_text(
        "[tool.ruff]\nline-length = 88\n")
    bs = detect_build_system(str(tmp_path))
    assert bs.name == "python"
    assert bs.install == "pip install -e ."
    assert bs.lint == "ruff check ."


def test_detect_node_with_scripts(tmp_path):
    (tmp_path / "package.json").write_text(
        '{"scripts": {"build": "tsc", "test": "jest", "lint": "eslint ."}}')
    bs = detect_build_system(str(tmp_path))
    assert bs.name == "node"
    assert bs.install == "npm install"
    assert bs.build == "npm run build"
    assert bs.test == "npm test"
    assert bs.lint == "npm run lint"


def test_detect_node_typescript_no_build_script(tmp_path):
    (tmp_path / "package.json").write_text('{"scripts": {}}')
    (tmp_path / "tsconfig.json").write_text("{}")
    bs = detect_build_system(str(tmp_path))
    assert bs.name == "node"
    assert bs.build == "npx tsc --noEmit"
    assert bs.test is None  # no test script


def test_detect_node_pnpm_lock(tmp_path):
    (tmp_path / "package.json").write_text('{"scripts": {"build": "x"}}')
    (tmp_path / "pnpm-lock.yaml").write_text("")
    bs = detect_build_system(str(tmp_path))
    assert bs.install == "pnpm install"
    assert bs.build == "pnpm build"


def test_detect_maven_cargo_go(tmp_path):
    (tmp_path / "pom.xml").write_text("<project/>")
    assert detect_build_system(str(tmp_path)).name == "maven"

    (tmp_path / "pom.xml").unlink()
    (tmp_path / "Cargo.toml").write_text("[package]\n")
    bs = detect_build_system(str(tmp_path))
    assert bs.name == "cargo" and bs.test == "cargo test"

    (tmp_path / "Cargo.toml").unlink()
    (tmp_path / "go.mod").write_text("module x\n")
    bs = detect_build_system(str(tmp_path))
    assert bs.name == "go" and bs.test == "go test ./..."


def test_detect_priority_python_over_make(tmp_path):
    (tmp_path / "requirements.txt").write_text("")
    (tmp_path / "Makefile").write_text("all:\n\techo hi\n")
    systems = detect_build_systems(str(tmp_path))
    names = [s.name for s in systems]
    assert names[0] == "python"          # language toolchain wins
    assert "make" in names               # but make is still reported


def test_detect_none_for_plain_dir(tmp_path):
    (tmp_path / "notes.txt").write_text("hello")
    assert detect_build_system(str(tmp_path)) is None
    assert detect_build_systems(str(tmp_path)) == []


# --------------------------------------------------------------------------
# constrained runner
# --------------------------------------------------------------------------
def _run(cmd, **kw):
    return asyncio.run(run_in_workspace(cmd, **kw))


def test_runner_captures_stdout_and_exit(tmp_path):
    py = sys.executable
    r = _run(f'"{py}" -c "print(\'hello-out\')"', cwd=str(tmp_path))
    assert r.ok
    assert r.exit_code == 0
    assert "hello-out" in r.stdout


def test_runner_nonzero_exit(tmp_path):
    py = sys.executable
    r = _run(f'"{py}" -c "import sys; sys.exit(3)"', cwd=str(tmp_path))
    assert r.exit_code == 3
    assert not r.ok


def test_runner_timeout(tmp_path):
    py = sys.executable
    r = _run(f'"{py}" -c "import time; time.sleep(10)"',
             cwd=str(tmp_path), timeout=1)
    assert r.timed_out
    assert not r.ok


def test_runner_denies_catastrophic(tmp_path):
    r = _run("rm -rf /", cwd=str(tmp_path))
    assert r.denied
    assert not r.ok
    assert "deny" in r.reason.lower() or "destructive" in r.reason.lower()


def test_runner_missing_workspace():
    r = _run("echo hi", cwd="/no/such/dir/xyz123")
    assert r.denied
    assert "not found" in r.reason.lower()


def test_runner_output_is_capped(tmp_path):
    py = sys.executable
    r = _run(f'"{py}" -c "print(\'A\'*100000)"',
             cwd=str(tmp_path), max_output=5000)
    assert len(r.stdout) <= 5200  # cap + truncation marker
    assert "truncated" in r.stdout


# --------------------------------------------------------------------------
# verification loop (#32) + wiring into the agent toolset
# --------------------------------------------------------------------------
from app.agent_workspace import verify_workspace  # noqa: E402


def _verify(root, **kw):
    return asyncio.run(verify_workspace(root, **kw))


def _pytest_on_path(monkeypatch):
    """Put this interpreter's Scripts/bin dir on PATH so the workspace subprocess
    resolves `pytest` (the venv isn't 'activated' when we launch python.exe). On
    a real VPS pytest/python/node are already on PATH."""
    import os
    scripts = os.path.dirname(sys.executable)
    monkeypatch.setenv("PATH", scripts + os.pathsep + os.environ.get("PATH", ""))


def test_verify_python_pass(tmp_path, monkeypatch):
    _pytest_on_path(monkeypatch)
    (tmp_path / "requirements.txt").write_text("")
    (tmp_path / "test_sample.py").write_text(
        "def test_ok():\n    assert 1 + 1 == 2\n")
    rep = _verify(str(tmp_path), steps=("test",))
    assert rep.system == "python"
    assert rep.attempted
    assert rep.ok
    assert rep.feedback() == ""


def test_verify_python_fail(tmp_path, monkeypatch):
    _pytest_on_path(monkeypatch)
    (tmp_path / "requirements.txt").write_text("")
    (tmp_path / "test_bad.py").write_text(
        "def test_bad():\n    assert False, 'boom'\n")
    rep = _verify(str(tmp_path), steps=("test",))
    assert rep.attempted
    assert not rep.ok
    fb = rep.feedback()
    assert "Verification failed" in fb
    assert "test" in fb


def test_verify_no_build_system(tmp_path):
    (tmp_path / "notes.txt").write_text("hi")
    rep = _verify(str(tmp_path))
    assert rep.system is None
    assert not rep.attempted
    assert rep.ok  # nothing to verify → not a failure
    assert "nothing to verify" in rep.summary.lower()


def test_verify_tool_is_registered_and_gated():
    from app.agent.tools import HANDLERS, SPEC_BY_NAME
    from app.agent import permissions

    assert "verify" in HANDLERS
    spec = SPEC_BY_NAME["verify"]
    assert spec.runs is True
    # plan mode is read-only → verify (runs) is denied; acceptEdits allows.
    assert permissions.decide("verify", {}, "plan")[0] == "deny"
    assert permissions.decide("verify", {}, "acceptEdits")[0] == "allow"


def test_verify_tool_handler_runs(tmp_path, monkeypatch):
    from app.agent.tools import HANDLERS

    _pytest_on_path(monkeypatch)
    (tmp_path / "requirements.txt").write_text("")
    (tmp_path / "test_h.py").write_text("def test_h():\n    assert True\n")
    out = asyncio.run(HANDLERS["verify"](str(tmp_path), steps=["test"]))
    assert "python" in out
    assert "test: PASS" in out


# --------------------------------------------------------------------------
# Phase 2 — workspace materializer (safe archive extraction)
# --------------------------------------------------------------------------
import io  # noqa: E402
import os  # noqa: E402
import zipfile  # noqa: E402

from app.agent_workspace import (  # noqa: E402
    materialize_archive,
    package_workspace,
    workspace_exists,
    workspace_path,
)


def _ws_env(monkeypatch, tmp_path):
    """Isolate workspace root to a temp dir for the duration of a test."""
    root = tmp_path / "ws_root"
    monkeypatch.setenv("ZAPTHETRICK_WS_ROOT", str(root))
    return root


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_materialize_normal_zip(monkeypatch, tmp_path):
    _ws_env(monkeypatch, tmp_path)
    data = _zip_bytes({
        "src/main.py": b"print('hi')\n",
        "README.md": b"# project\n",
    })
    res = materialize_archive("conv1", data, "project.zip")
    assert res.ok
    assert res.error is None
    assert res.files == 2
    assert workspace_exists("conv1")
    p = workspace_path("conv1")
    assert os.path.isfile(os.path.join(p, "src", "main.py"))
    assert os.path.isfile(os.path.join(p, "README.md"))


def test_materialize_blocks_zip_slip(monkeypatch, tmp_path):
    _ws_env(monkeypatch, tmp_path)
    data = _zip_bytes({
        "ok.txt": b"fine\n",
        "../evil.txt": b"escaped\n",
        "../../also_evil.txt": b"escaped\n",
    })
    res = materialize_archive("conv2", data, "evil.zip")
    # Good member written, traversal members dropped (skipped), nothing escapes.
    assert res.files == 1
    assert res.skipped >= 2
    p = workspace_path("conv2")
    root_parent = os.path.dirname(os.path.realpath(p))
    assert not os.path.exists(os.path.join(root_parent, "evil.txt"))
    assert os.path.isfile(os.path.join(p, "ok.txt"))


def test_materialize_caps_truncate(monkeypatch, tmp_path):
    _ws_env(monkeypatch, tmp_path)
    import app.agent_workspace.materialize as mz
    monkeypatch.setattr(mz, "MAX_FILES", 3)
    members = {f"f{i}.txt": b"x" for i in range(10)}
    res = materialize_archive("conv3", data=_zip_bytes(members),
                              filename="many.zip")
    assert res.truncated
    assert res.files <= 3


def test_materialize_unsupported_type(monkeypatch, tmp_path):
    _ws_env(monkeypatch, tmp_path)
    res = materialize_archive("conv4", b"not an archive", "notes.txt")
    assert not res.ok
    assert res.error is not None
    assert "unsupported" in res.error.lower()


def test_package_workspace_roundtrips(monkeypatch, tmp_path):
    _ws_env(monkeypatch, tmp_path)
    data = _zip_bytes({
        "a.py": b"a\n",
        "pkg/b.py": b"b\n",
        ".git/config": b"should be skipped\n",
        "node_modules/dep.js": b"skip\n",
    })
    materialize_archive("conv5", data, "p.zip")
    out = package_workspace("conv5")
    with zipfile.ZipFile(io.BytesIO(out)) as z:
        names = set(z.namelist())
    assert "a.py" in names
    assert "pkg/b.py" in names
    assert not any(n.startswith(".git/") for n in names)
    assert not any(n.startswith("node_modules/") for n in names)


def test_materialize_replaces_prior_contents(monkeypatch, tmp_path):
    _ws_env(monkeypatch, tmp_path)
    materialize_archive("conv6", _zip_bytes({"old.py": b"old\n"}), "v1.zip")
    materialize_archive("conv6", _zip_bytes({"new.py": b"new\n"}), "v2.zip")
    p = workspace_path("conv6")
    assert os.path.isfile(os.path.join(p, "new.py"))
    assert not os.path.exists(os.path.join(p, "old.py"))
