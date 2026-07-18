"""Phase 5 — context engineering + token budget (#1/#36).

Pure/offline: ranking + packing over a temp workspace. No LLM, no network.
"""
from __future__ import annotations

from app.chat.context_builder import (
    ContextBudget,
    build_context,
    rank_files,
)


def _mk(tmp_path, tree: dict):
    for rel, body in tree.items():
        p = tmp_path / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body, encoding="utf-8")
    return str(tmp_path)


SAMPLE = {
    "src/auth/login.py": (
        "def login(user, password):\n"
        "    '''authenticate a user'''\n"
        "    return check_password(user, password)\n"
    ),
    "src/users/api.py": (
        "def list_users():\n    return db.all()\n\n"
        "def add_user(u):\n    return db.insert(u)\n"
    ),
    "src/utils/helpers.py": "def slugify(s):\n    return s.lower()\n",
    "README.md": "# Project\nA sample app.\n",
}


# ── ranking ────────────────────────────────────────────────────────────────
def test_rank_prioritizes_task_mentioned_file(tmp_path):
    _mk(tmp_path, SAMPLE)
    files = [(k, v) for k, v in SAMPLE.items()]
    ranked = rank_files(files, "fix the login bug")
    assert ranked
    assert ranked[0].path == "src/auth/login.py"


def test_rank_api_task_surfaces_api_file(tmp_path):
    files = [(k, v) for k, v in SAMPLE.items()]
    ranked = rank_files(files, "add pagination to the users api endpoint")
    paths = [r.path for r in ranked]
    assert "src/users/api.py" in paths
    # The api file should outrank an unrelated helper.
    assert paths.index("src/users/api.py") < paths.index("src/utils/helpers.py")


def test_rank_empty_for_no_files():
    assert rank_files([], "anything") == []


# ── build_context: budget + packing ────────────────────────────────────────
def test_build_context_includes_relevant_and_reports(tmp_path):
    ws = _mk(tmp_path, SAMPLE)
    res = build_context(ws, "fix the login bug")
    assert res.text
    assert "src/auth/login.py" in res.files
    assert res.tokens > 0
    assert "RELEVANT PROJECT CONTEXT" in res.text
    # The most relevant file's content appears in the preamble.
    assert "def login" in res.text


def test_build_context_respects_token_budget(tmp_path):
    # Many sizable files + a tiny budget → must stay within budget and truncate.
    big = {f"src/mod_{i}.py": ("def f_%d():\n    return %d\n" % (i, i)) * 80
           for i in range(40)}
    ws = _mk(tmp_path, big)
    res = build_context(ws, "optimize f_3 and f_7",
                        budget=ContextBudget(max_tokens=800))
    assert res.tokens <= 800
    assert res.truncated
    assert res.files  # still included something


def test_build_context_empty_workspace(tmp_path):
    res = build_context(str(tmp_path), "build an app")
    assert res.text == ""
    assert res.files == []


def test_build_context_skips_vendored_dirs(tmp_path):
    tree = dict(SAMPLE)
    tree["node_modules/dep/index.js"] = "module.exports = 1;\n" * 50
    ws = _mk(tmp_path, tree)
    res = build_context(ws, "fix the login bug")
    assert not any("node_modules" in f for f in res.files)


# ── loop wiring: context param is accepted + injected ───────────────────────
def test_run_goal_accepts_context_param():
    import inspect

    from app.agent import loop
    assert "context" in inspect.signature(loop.run_goal).parameters
    assert "context" in inspect.signature(loop.run_agent).parameters
