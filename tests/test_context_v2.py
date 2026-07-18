"""P2-3 — context engineering v2.

Pure/offline: the hierarchical repo digest (+ caching), read compression
(signatures + relevant hunks), recency-aware ranking, the assembled
build_context_v2 preamble, and the cross-step scratchpad (brain) + the `note`/
`outline` agent tools. No LLM, no network.
"""
from __future__ import annotations

import asyncio
import os
import time

from app.agent_workspace.brain import (
    append_scratchpad,
    clear_scratchpad,
    read_scratchpad,
    scratchpad_context,
)
from app.chat.context_builder import ContextBudget, RankedFile
from app.chat.context_v2 import (
    build_context_v2,
    build_repo_digest,
    compress_source,
    rank_with_recency,
)


def _mk(tmp_path, tree: dict):
    for rel, body in tree.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return str(tmp_path)


SAMPLE = {
    "src/auth/login.py": (
        "import os\n\n"
        "def login(user, password):\n"
        "    '''authenticate a user'''\n"
        "    return check_password(user, password)\n\n"
        "class SessionManager:\n"
        "    def start(self): pass\n"
    ),
    "src/users/api.py": (
        "def list_users():\n    return db.all()\n\n"
        "def add_user(u):\n    return db.insert(u)\n"
    ),
    "src/utils/helpers.py": "def slugify(s):\n    return s.lower()\n",
    "main.py": "def main():\n    print('hi')\n",
    "README.md": "# Project\nA sample app.\n",
}


# ── hierarchical repo digest ────────────────────────────────────────────────
def test_repo_digest_summarizes_project(tmp_path):
    ws = _mk(tmp_path, SAMPLE)
    dig = build_repo_digest(ws, use_cache=False)
    assert dig.text
    assert "REPOSITORY MAP" in dig.text
    assert "Python" in dig.text
    # entrypoint detected
    assert any("main.py" in e for e in dig.entrypoints)
    # a directory and one of its symbols appears
    assert "src/auth" in dig.text
    assert dig.file_count >= 4


def test_repo_digest_caches_and_reuses(tmp_path):
    ws = _mk(tmp_path, SAMPLE)
    first = build_repo_digest(ws, use_cache=True)
    assert not first.cached
    assert os.path.isfile(os.path.join(ws, ".zapthetrick", "digest.md"))
    second = build_repo_digest(ws, use_cache=True)
    assert second.cached
    assert second.fingerprint == first.fingerprint


def test_repo_digest_rebuilds_when_files_change(tmp_path):
    ws = _mk(tmp_path, SAMPLE)
    first = build_repo_digest(ws, use_cache=True)
    time.sleep(0.01)
    # add a file → fingerprint changes → not cached
    (tmp_path / "src" / "new_mod.py").write_text("def fresh(): pass\n",
                                                 encoding="utf-8")
    third = build_repo_digest(ws, use_cache=True)
    assert third.fingerprint != first.fingerprint
    assert not third.cached


def test_repo_digest_empty_workspace(tmp_path):
    assert build_repo_digest(str(tmp_path)).text == ""


# ── read compression ─────────────────────────────────────────────────────────
def test_compress_source_small_returns_whole():
    src = "def a(): pass\ndef b(): pass\n"
    assert compress_source(src, "a") == src


def test_compress_source_extracts_signatures_and_hunks():
    lines = ["# header"]
    for i in range(120):
        lines.append(f"x{i} = {i}")
    lines.insert(60, "def important_target():")
    lines.insert(61, "    return secret_value")
    src = "\n".join(lines)
    out = compress_source(src, "fix important_target", max_lines=40)
    assert "SIGNATURES:" in out
    assert "important_target" in out
    assert "RELEVANT EXCERPTS:" in out
    # the hunk window includes the target line with its line number
    assert "important_target" in out


def test_compress_source_no_terms_falls_back_to_head():
    src = "\n".join(f"line {i}" for i in range(200))
    out = compress_source(src, "", max_lines=30)
    assert out.endswith("…")
    assert "line 0" in out


# ── recency-aware ranking ─────────────────────────────────────────────────────
def test_rank_with_recency_boosts_recent_files():
    ranked = [
        RankedFile("old.py", 5.0, "x"),
        RankedFile("new.py", 5.0, "x"),
    ]
    mtimes = {"old.py": 1000.0, "new.py": 2000.0}
    out = rank_with_recency(ranked, mtimes, weight=2.0)
    # new.py gets the full recency boost → ranks first
    assert out[0].path == "new.py"
    assert out[0].score > out[1].score
    assert "recently edited" in out[0].reason


def test_rank_with_recency_noop_without_mtimes():
    ranked = [RankedFile("a.py", 1.0, "x")]
    assert rank_with_recency(ranked, {}) == ranked


# ── assembled v2 context ─────────────────────────────────────────────────────
def test_build_context_v2_includes_digest_and_relevant_file(tmp_path):
    ws = _mk(tmp_path, SAMPLE)
    res = build_context_v2(ws, "fix the login bug")
    assert res.text
    assert "REPOSITORY MAP" in res.text          # digest prefix
    assert "RELEVANT FILES" in res.text
    assert "src/auth/login.py" in res.files
    assert res.tokens > 0


def test_build_context_v2_respects_budget(tmp_path):
    big = {f"src/mod_{i}.py": ("def f_%d():\n    return %d\n" % (i, i)) * 80
           for i in range(40)}
    ws = _mk(tmp_path, big)
    res = build_context_v2(ws, "optimize f_3 and f_7",
                           budget=ContextBudget(max_tokens=900))
    assert res.tokens <= 900
    assert res.files


def test_build_context_v2_empty_workspace(tmp_path):
    res = build_context_v2(str(tmp_path), "build an app")
    assert res.text == "" and res.files == []


def test_build_context_v2_can_disable_digest(tmp_path):
    ws = _mk(tmp_path, SAMPLE)
    res = build_context_v2(ws, "fix the login bug", include_digest=False)
    assert "REPOSITORY MAP" not in res.text
    assert "RELEVANT FILES" in res.text


# ── cross-step scratchpad ─────────────────────────────────────────────────────
def test_scratchpad_append_read_and_context(tmp_path):
    ws = str(tmp_path)
    assert append_scratchpad(ws, "auth lives in src/auth/login.py", tag="map")
    assert append_scratchpad(ws, "build uses pytest")
    raw = read_scratchpad(ws)
    assert "auth lives in" in raw and "[map]" in raw
    ctx = scratchpad_context(ws)
    assert "WORKING NOTES" in ctx
    assert "build uses pytest" in ctx


def test_scratchpad_dedupes_and_clears(tmp_path):
    ws = str(tmp_path)
    append_scratchpad(ws, "same note")
    append_scratchpad(ws, "same note")  # duplicate → ignored
    body = [ln for ln in read_scratchpad(ws).splitlines()
            if "same note" in ln]
    assert len(body) == 1
    clear_scratchpad(ws)
    assert scratchpad_context(ws) == ""


def test_scratchpad_empty_returns_blank(tmp_path):
    assert scratchpad_context(str(tmp_path)) == ""
    assert not append_scratchpad(str(tmp_path), "   ")


# ── agent tools: note + outline ───────────────────────────────────────────────
def test_note_tool_writes_scratchpad(tmp_path):
    from app.agent import tools
    ws = str(tmp_path)
    res = asyncio.run(tools.note(ws, note="found the bug in parser.py",
                                 tag="bug"))
    assert res == "noted"
    assert "found the bug" in read_scratchpad(ws)


def test_outline_tool_compresses_file(tmp_path):
    from app.agent import tools
    lines = [f"row{i} = {i}" for i in range(120)]
    lines.insert(50, "def target_fn():")
    (tmp_path / "big.py").write_text("\n".join(lines), encoding="utf-8")
    res = asyncio.run(tools.outline(str(tmp_path), path="big.py",
                                    query="target_fn"))
    assert "SIGNATURES:" in res
    assert "target_fn" in res


def test_outline_tool_missing_file(tmp_path):
    from app.agent import tools
    res = asyncio.run(tools.outline(str(tmp_path), path="nope.py"))
    assert res.startswith("ERROR")
