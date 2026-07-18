"""Phase 4.5 — security & abuse hardening.

Adversarially exercises the workspace extractor + constrained runner, the
secret-redaction layer, and the chat agent-run concurrency cap. All offline /
deterministic (no network, no real toolchain).
"""
from __future__ import annotations

import asyncio
import io
import os
import tarfile
import zipfile

import pytest

from app.agent_workspace import (
    materialize_archive,
    redact_event,
    redact_secrets,
    run_in_workspace,
    workspace_path,
)
import app.agent_workspace.materialize as mz


def _ws_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ZAPTHETRICK_WS_ROOT", str(tmp_path / "ws_root"))


def _zip_bytes(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buf.getvalue()


def _run(cmd, **kw):
    return asyncio.run(run_in_workspace(cmd, **kw))


# ── extractor: path traversal / zip-slip variants ─────────────────────────
@pytest.mark.parametrize("evil", [
    "../evil.txt",
    "../../evil.txt",
    "a/../../evil.txt",
    "..\\..\\evil.txt",
    "/etc/passwd",
    "C:\\Windows\\system32\\evil.dll",
])
def test_traversal_members_rejected(monkeypatch, tmp_path, evil):
    _ws_env(monkeypatch, tmp_path)
    data = _zip_bytes({"good.txt": b"ok", evil: b"pwned"})
    res = materialize_archive("c", data, "x.zip")
    assert res.files == 1                 # only good.txt
    assert res.skipped >= 1               # traversal dropped
    root = os.path.realpath(workspace_path("c"))
    # Nothing escaped the workspace root.
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            full = os.path.realpath(os.path.join(dirpath, f))
            assert full.startswith(root + os.sep)
    assert not os.path.exists(os.path.join(os.path.dirname(root), "evil.txt"))


def test_tar_symlink_member_skipped(monkeypatch, tmp_path):
    _ws_env(monkeypatch, tmp_path)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as t:
        payload = b"real"
        ti = tarfile.TarInfo("good.txt")
        ti.size = len(payload)
        t.addfile(ti, io.BytesIO(payload))
        link = tarfile.TarInfo("escape")
        link.type = tarfile.SYMTYPE
        link.linkname = "../../../../etc/passwd"
        t.addfile(link)
    res = materialize_archive("c", buf.getvalue(), "x.tar")
    assert res.files == 1                 # symlink skipped, only good.txt
    root = workspace_path("c")
    assert os.path.isfile(os.path.join(root, "good.txt"))
    assert not os.path.islink(os.path.join(root, "escape"))


# ── extractor: archive-bomb caps (count / per-file / total expansion) ─────
def test_total_expansion_cap_truncates(monkeypatch, tmp_path):
    _ws_env(monkeypatch, tmp_path)
    monkeypatch.setattr(mz, "MAX_TOTAL_BYTES", 500_000)
    # Highly compressible payload (zeros) — small zip, large on disk.
    members = {f"f{i}.bin": b"\x00" * 200_000 for i in range(10)}
    res = materialize_archive("c", _zip_bytes(members), "bomb.zip")
    assert res.truncated
    assert res.bytes <= 500_000 + 200_000  # stopped near the cap


def test_per_file_size_cap_skips(monkeypatch, tmp_path):
    _ws_env(monkeypatch, tmp_path)
    monkeypatch.setattr(mz, "MAX_FILE_BYTES", 1000)
    data = _zip_bytes({"small.txt": b"ok", "huge.bin": b"\x00" * 50_000})
    res = materialize_archive("c", data, "x.zip")
    assert res.files == 1                 # only small.txt
    assert res.skipped >= 1
    assert not os.path.exists(os.path.join(workspace_path("c"), "huge.bin"))


def test_file_count_cap_truncates(monkeypatch, tmp_path):
    _ws_env(monkeypatch, tmp_path)
    monkeypatch.setattr(mz, "MAX_FILES", 5)
    res = materialize_archive(
        "c", _zip_bytes({f"f{i}.txt": b"x" for i in range(20)}), "x.zip")
    assert res.truncated and res.files <= 5


def test_crafted_filename_is_inert_data(monkeypatch, tmp_path):
    """A member whose NAME looks like a shell injection is written as plain
    data inside the workspace — never executed/interpreted."""
    _ws_env(monkeypatch, tmp_path)
    res = materialize_archive(
        "c", _zip_bytes({"; rm -rf ~ #.txt": b"inert"}), "x.zip")
    assert res.ok
    root = os.path.realpath(workspace_path("c"))
    # The odd-named file exists strictly inside the workspace; nothing ran.
    entries = os.listdir(root)
    assert any("rm -rf" in e for e in entries)


# ── runner: deny-list blocks catastrophic / injected commands ─────────────
@pytest.mark.parametrize("cmd", [
    "rm -rf /",
    "ls; rm -rf /",
    "build && rm -rf ~",
    ":(){ :|:& };:",
    "mkfs.ext4 /dev/sda1",
    "dd if=/dev/zero of=/dev/sda",
    "shutdown -h now",
    "sudo cat /etc/shadow",
])
def test_runner_blocks_dangerous(tmp_path, cmd):
    r = _run(cmd, cwd=str(tmp_path))
    assert r.denied and not r.ok


def test_runner_empty_command_denied(tmp_path):
    r = _run("   ", cwd=str(tmp_path))
    assert r.denied


# ── secret redaction ──────────────────────────────────────────────────────
def test_redact_provider_tokens():
    samples = [
        "AKIAIOSFODNN7EXAMPLE",
        "ghp_" + "a" * 36,
        "AIza" + "b" * 35,
        "sk-" + "c" * 40,
        "nvapi-" + "d" * 40,
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc-_DEF123",
    ]
    for s in samples:
        out = redact_secrets(f"token is {s} here")
        assert s not in out, s
        assert "[REDACTED]" in out


def test_redact_private_key_block():
    pem = ("-----BEGIN RSA PRIVATE KEY-----\n"
           "MIIEowIBAAKCAQEA0Z3VS5JJcds3xfn\n"
           "-----END RSA PRIVATE KEY-----")
    out = redact_secrets(f"key:\n{pem}\nend")
    assert "PRIVATE KEY" not in out or "[REDACTED]" in out
    assert "MIIEowIBAAKCAQEA" not in out


def test_redact_env_assignments_and_urls():
    out = redact_secrets("API_KEY=supersecretvalue123\n"
                         "DB=postgres://user:hunter2@host:5432/db")
    assert "supersecretvalue123" not in out
    assert "hunter2" not in out
    assert "[REDACTED]" in out


def test_redact_preserves_ordinary_code():
    code = "def add(a, b):\n    return a + b  # simple sum\n"
    assert redact_secrets(code) == code


def test_redact_event_scrubs_payload_fields():
    evt = {
        "type": "tool_result",
        "tool": "read",
        "result": "AWS_SECRET_ACCESS_KEY=abcd1234efgh5678ijkl",
        "args": {"path": ".env", "note": "token ghp_" + "z" * 36},
    }
    red = redact_event(evt)
    assert "abcd1234efgh5678ijkl" not in red["result"]
    assert "ghp_zzz" not in red["args"]["note"]


# ── concurrency cap ───────────────────────────────────────────────────────
def test_concurrency_semaphore_honors_config(monkeypatch):
    from app.api import routes_chat_agent as rca
    from app.core.config_loader import cfg

    monkeypatch.setattr(cfg.advanced_rag, "max_concurrent_agent_runs", 2)

    async def go():
        sem = rca._semaphore()
        await sem.acquire()
        await sem.acquire()
        assert sem.locked()              # cap of 2 reached → next would queue
        sem.release()
        sem.release()
        assert not sem.locked()

    asyncio.run(go())


def test_redaction_flag_default_on():
    from app.api import routes_chat_agent as rca
    assert rca._redact_on() is True


def test_agent_run_redacts_streamed_secret(monkeypatch):
    """End-to-end wiring: a secret in a tool_result is scrubbed before it's
    streamed/persisted by the chat agent-run endpoint."""
    from app.api import routes_chat_agent as rca

    async def _scripted(*_a, **_k):
        yield {"type": "tool_result", "tool": "read",
               "result": "AWS_SECRET_ACCESS_KEY=topsecretleak9999"}
        yield {"type": "final", "message": "done, key was ghp_" + "q" * 36}

    monkeypatch.setattr(rca, "_resolve_kind", lambda b: "edit")
    monkeypatch.setattr(rca, "_resolve_workspace", lambda c, k: ("/tmp/ws", ""))

    async def _diff(_ws):
        return ""
    monkeypatch.setattr(rca, "_diff", _diff)
    monkeypatch.setattr("storage.db.get_session_factory", lambda: None)
    import app.agent.loop as loop
    monkeypatch.setattr(loop, "run_goal", _scripted)

    async def collect():
        out = []
        resp = await rca.chat_agent_run(rca.ChatAgentRunBody(
            conversation_id="c1", task="read the env", kind="edit"))
        async for chunk in resp.body_iterator:
            out.append(chunk if isinstance(chunk, str) else chunk.decode())
        return "".join(out)

    joined = asyncio.run(collect())
    assert "topsecretleak9999" not in joined
    assert "ghp_qqq" not in joined
    assert "[REDACTED]" in joined
