"""P2-7 — git workflow on the chat path.

Pure: slugify/branch_name/commit_subject/commit_message, remote parsing, and
provider compare/MR URLs. Integration: run_git_workflow over a REAL git repo
(branch + commit), plus the no-remote and not-a-repo paths.
"""
from __future__ import annotations

import asyncio
import os
import subprocess

import pytest

from app.agent_workspace.gitflow import (
    branch_name,
    commit_message,
    commit_subject,
    compare_url,
    parse_remote,
    run_git_workflow,
    slugify,
)


# ── pure helpers ──────────────────────────────────────────────────────────
def test_slugify_and_branch_name():
    assert slugify("Fix the Login Bug!") == "fix-the-login-bug"
    assert slugify("   ") == "change"
    b = branch_name("Add pagination", prefix="dtt")
    assert b.startswith("dtt/add-pagination-")


def test_commit_subject_prefers_summary_first_line():
    s = commit_subject("do a thing", "Added /users endpoint\nmore detail")
    assert s == "Added /users endpoint"
    # falls back to the task when no summary
    assert commit_subject("Fix the bug.", "") == "Fix the bug"


def test_commit_subject_truncates():
    s = commit_subject("x" * 200, "")
    assert len(s) <= 72 and s.endswith("…")


def test_commit_message_has_subject_body_attribution():
    m = commit_message("add pagination to users", "Added pagination")
    assert m.splitlines()[0] == "Added pagination"
    assert "add pagination to users" in m
    assert "ZapTheTrick" in m


def test_parse_remote_https_and_scp():
    a = parse_remote("https://github.com/acme/widgets.git")
    assert a == {"host": "github.com", "owner": "acme", "repo": "widgets",
                 "kind": "github"}
    b = parse_remote("git@gitlab.com:group/sub/proj.git")
    assert b["host"] == "gitlab.com" and b["owner"] == "group/sub"
    assert b["repo"] == "proj" and b["kind"] == "gitlab"
    assert parse_remote("") is None
    assert parse_remote("not a url") is None


def test_compare_url_per_provider():
    gh = compare_url("https://github.com/a/b.git", "main", "dtt/x")
    assert gh == "https://github.com/a/b/compare/main...dtt/x?expand=1"
    gl = compare_url("git@gitlab.com:a/b.git", "main", "dtt/x")
    assert "merge_requests/new" in gl and "source_branch" in gl
    bb = compare_url("https://bitbucket.org/a/b.git", "main", "feat")
    assert "pull-requests/new" in bb
    assert compare_url("", "main", "x") == ""


# ── integration over a real repo ──────────────────────────────────────────
def _git_available() -> bool:
    try:
        subprocess.run(["git", "--version"], capture_output=True, check=True)
        return True
    except Exception:  # noqa: BLE001
        return False


def _init_repo(ws: str):
    def g(*a):
        subprocess.run(["git", *a], cwd=ws, check=True, capture_output=True)
    g("init")
    g("config", "user.email", "t@t.com")
    g("config", "user.name", "t")
    with open(os.path.join(ws, "a.py"), "w") as f:
        f.write("x = 1\n")
    g("add", "-A")
    g("commit", "-m", "baseline")


@pytest.mark.skipif(not _git_available(), reason="git not installed")
def test_run_git_workflow_branches_and_commits(tmp_path):
    ws = str(tmp_path)
    _init_repo(ws)
    # make an uncommitted change
    with open(os.path.join(ws, "a.py"), "w") as f:
        f.write("x = 2\ndef added():\n    return 3\n")

    res = asyncio.run(run_git_workflow(
        ws, task="fix the value and add a helper",
        summary="Updated x and added helper"))
    assert res is not None
    assert res.committed is True
    assert res.branch.startswith("zapthetrick/")
    assert res.commit  # short sha
    assert res.pushed is False           # no remote
    assert any("no remote" in n for n in res.notes)

    # the commit is real and on the new branch
    cur = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                         cwd=ws, capture_output=True, text=True)
    assert cur.stdout.strip() == res.branch


@pytest.mark.skipif(not _git_available(), reason="git not installed")
def test_run_git_workflow_nothing_to_commit(tmp_path):
    ws = str(tmp_path)
    _init_repo(ws)  # clean tree, no changes
    res = asyncio.run(run_git_workflow(ws, task="noop"))
    assert res is not None and res.committed is False
    assert any("no changes" in n for n in res.notes)


@pytest.mark.skipif(not _git_available(), reason="git not installed")
def test_run_git_workflow_with_remote_makes_compare_url(tmp_path):
    ws = str(tmp_path)
    _init_repo(ws)
    subprocess.run(["git", "remote", "add", "origin",
                    "https://github.com/acme/widgets.git"],
                   cwd=ws, check=True, capture_output=True)
    with open(os.path.join(ws, "a.py"), "w") as f:
        f.write("x = 9\n")
    # auto_push False → no network; still get a compare URL for the branch
    res = asyncio.run(run_git_workflow(ws, task="bump x"))
    assert res is not None and res.committed
    assert res.remote.endswith("widgets.git")
    assert res.pr_url.startswith("https://github.com/acme/widgets/compare/")


def test_run_git_workflow_not_a_repo(tmp_path):
    # no git init → returns None (best-effort)
    res = asyncio.run(run_git_workflow(str(tmp_path), task="x"))
    assert res is None


def test_gitresult_to_dict_shape(tmp_path):
    from app.agent_workspace.gitflow import GitResult
    d = GitResult(branch="b", base="main", committed=True, commit="abc",
                  message="Do thing\n\nbody").to_dict()
    assert d["branch"] == "b" and d["subject"] == "Do thing"
    assert d["committed"] is True
