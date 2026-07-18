"""Git workflow for the chat agent path — branch / commit / push / PR (P2-7).

Claude Code's everyday loop ends in git: a feature branch, a clean commit with a
sensible message, and (when a remote exists) a pull request. The chat workspace
is already a git repo (a `baseline` commit from materialize), so after a
successful build/edit we:

  1. create a feature branch off the current HEAD,
  2. stage + commit the change with an auto-written message (subject from the
     task / final summary, body = the task),
  3. (opt-in) push the branch to `origin` — injecting a token for HTTPS remotes,
  4. produce a **PR link**: a provider "compare/new-MR" URL (no API/token
     needed), or a real PR via the `gh`/`glab` CLI when available.

Run this AFTER the diff/semantic/test passes (they diff the *working tree* vs
the baseline; committing moves HEAD). Everything is best-effort: a missing git,
no remote, or no token degrades gracefully and never breaks the run.

Pure helpers (`slugify`, `branch_name`, `commit_subject`, `parse_remote`,
`compare_url`) are offline + unit-tested; `run_git_workflow` does the IO.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from urllib.parse import quote, urlparse


# ── pure helpers ──────────────────────────────────────────────────────────
def slugify(text: str, *, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    s = re.sub(r"-{2,}", "-", s)
    return s[:max_len].strip("-") or "change"


def branch_name(task: str, *, prefix: str = "zapthetrick") -> str:
    """A unique, readable feature-branch name."""
    return f"{prefix}/{slugify(task)}-{int(time.time()) % 100000}"


def commit_subject(task: str, summary: str = "", *, max_len: int = 72) -> str:
    """A concise imperative commit subject from the summary (else the task)."""
    src = (summary or "").strip().splitlines()
    head = (src[0] if src else "") or (task or "").strip()
    head = re.sub(r"\s+", " ", head).strip().rstrip(".")
    if len(head) > max_len:
        head = head[:max_len - 1].rstrip() + "…"
    return head or "Update project"


def commit_message(task: str, summary: str = "") -> str:
    subject = commit_subject(task, summary)
    body = (task or "").strip()
    msg = subject
    if body and body.lower() not in subject.lower():
        msg += f"\n\n{body[:500]}"
    msg += "\n\n🤖 Generated with ZapTheTrick"
    return msg


def parse_remote(url: str) -> dict | None:
    """Parse a git remote URL → {host, owner, repo, kind}. kind∈{github,gitlab,
    bitbucket,other}. Handles https and scp-style git@host:owner/repo.git."""
    u = (url or "").strip()
    if not u:
        return None
    host = owner = repo = ""
    if u.startswith("git@") or (("@" in u) and "://" not in u):
        # scp-style: git@github.com:owner/repo.git
        m = re.match(r"[^@]+@([^:]+):(.+)", u)
        if not m:
            return None
        host, path = m.group(1), m.group(2)
    else:
        p = urlparse(u)
        host = p.hostname or ""
        path = (p.path or "").lstrip("/")
    path = re.sub(r"\.git$", "", path).strip("/")
    parts = path.split("/")
    if host and len(parts) >= 2:
        owner = "/".join(parts[:-1])
        repo = parts[-1]
    if not (host and owner and repo):
        return None
    low = host.lower()
    kind = ("github" if "github" in low else
            "gitlab" if "gitlab" in low else
            "bitbucket" if "bitbucket" in low else "other")
    return {"host": host, "owner": owner, "repo": repo, "kind": kind}


def compare_url(remote: str, base: str, branch: str) -> str:
    """A 'open a PR/MR' web URL for the pushed branch (no API needed)."""
    info = parse_remote(remote)
    if not info:
        return ""
    host, owner, repo, kind = (info["host"], info["owner"], info["repo"],
                               info["kind"])
    b, h = quote(base), quote(branch)
    if kind == "github":
        return f"https://{host}/{owner}/{repo}/compare/{b}...{h}?expand=1"
    if kind == "gitlab":
        return (f"https://{host}/{owner}/{repo}/-/merge_requests/new"
                f"?merge_request%5Bsource_branch%5D={h}"
                f"&merge_request%5Btarget_branch%5D={b}")
    if kind == "bitbucket":
        return (f"https://{host}/{owner}/{repo}/pull-requests/new"
                f"?source={h}&dest={b}")
    return ""


def _authed_remote(url: str, token: str) -> str:
    """Inject a token into an HTTPS remote for a non-interactive push.
    Returns the URL unchanged for non-HTTPS (SSH) remotes."""
    if not token:
        return url
    p = urlparse(url)
    if p.scheme not in ("http", "https") or not p.hostname:
        return url
    host = p.hostname + (f":{p.port}" if p.port else "")
    return f"{p.scheme}://x-access-token:{quote(token)}@{host}{p.path}"


@dataclass
class GitResult:
    enabled: bool = True
    branch: str = ""
    base: str = ""
    committed: bool = False
    commit: str = ""
    pushed: bool = False
    remote: str = ""
    pr_url: str = ""
    message: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        subject = self.message.splitlines()[0] if self.message.strip() else ""
        return {
            "branch": self.branch, "base": self.base,
            "committed": self.committed, "commit": self.commit,
            "pushed": self.pushed, "remote": self.remote,
            "pr_url": self.pr_url, "subject": subject,
            "notes": self.notes,
        }


async def _git(workspace: str, cmd: str, *, timeout: int = 120):
    from .runner import run_in_workspace
    return await run_in_workspace(cmd, cwd=workspace, timeout=timeout)


async def run_git_workflow(
    workspace: str,
    *,
    task: str,
    summary: str = "",
    branch_prefix: str = "zapthetrick",
    auto_push: bool = False,
    token: str = "",
    open_pr: bool = False,
) -> GitResult | None:
    """Branch + commit (always), optionally push + produce a PR link.

    Returns None if the workspace isn't a git repo / git is unavailable / there
    was nothing to commit. Never raises."""
    res = GitResult()
    try:
        # Is this a git repo with a HEAD to branch from?
        head = await _git(workspace, "git rev-parse --verify HEAD")
        if head.denied or head.exit_code != 0:
            return None
        base = await _git(workspace, "git rev-parse --abbrev-ref HEAD")
        res.base = (base.stdout or "").strip() or "main"

        res.message = commit_message(task, summary)
        res.branch = branch_name(task, prefix=branch_prefix)

        await _git(workspace, "git config user.email agent@zapthetrick")
        await _git(workspace, "git config user.name ZapTheTrick")
        co = await _git(workspace, f'git checkout -b "{res.branch}"')
        if co.exit_code != 0:
            res.notes.append("could not create branch")
            return res
        await _git(workspace, "git add -A")
        # Nothing staged → no change to commit (don't make an empty commit).
        diff = await _git(workspace, "git diff --cached --quiet")
        if diff.exit_code == 0:
            res.notes.append("no changes to commit")
            return res
        # Write the message via a file to keep quoting/newlines intact.
        import os
        msg_path = os.path.join(workspace, ".zapthetrick", "commit_msg.txt")
        os.makedirs(os.path.dirname(msg_path), exist_ok=True)
        with open(msg_path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(res.message)
        cm = await _git(workspace, 'git commit -F ".zapthetrick/commit_msg.txt"')
        try:
            os.remove(msg_path)
        except OSError:
            pass
        if cm.exit_code != 0:
            res.notes.append("commit failed")
            return res
        res.committed = True
        sha = await _git(workspace, "git rev-parse --short HEAD")
        res.commit = (sha.stdout or "").strip()

        # Remote? (the materialized-upload flow usually has none.)
        rem = await _git(workspace, "git remote get-url origin")
        if rem.exit_code == 0 and (rem.stdout or "").strip():
            res.remote = rem.stdout.strip()
        if not res.remote:
            res.notes.append("no remote configured — branch is local "
                             "(in the downloadable project)")
            return res

        if auto_push:
            push_to = _authed_remote(res.remote, token) if token else "origin"
            push = await _git(
                workspace,
                f'git push -u "{push_to}" "{res.branch}"', timeout=300)
            if push.exit_code == 0:
                res.pushed = True
            else:
                res.notes.append("push failed (check remote credentials/token)")

        # PR link: a real PR via CLI if asked + available, else a compare URL.
        if res.pushed or not auto_push:
            res.pr_url = compare_url(res.remote, res.base, res.branch)
        if open_pr and res.pushed:
            pr = await _open_pr_via_cli(workspace, res)
            if pr:
                res.pr_url = pr
        return res
    except Exception:  # noqa: BLE001 — git workflow must never break the run
        return res if res.branch else None


async def _open_pr_via_cli(workspace: str, res: GitResult) -> str:
    """Best-effort real PR via the gh/glab CLI (returns the PR URL or '')."""
    info = parse_remote(res.remote)
    if not info:
        return ""
    subject = commit_subject(res.message)
    if info["kind"] == "github":
        r = await _git(
            workspace,
            f'gh pr create --base "{res.base}" --head "{res.branch}" '
            f'--title "{subject}" --fill', timeout=120)
        if r.exit_code == 0:
            m = re.search(r"https?://\S+", r.stdout or "")
            return m.group(0) if m else ""
    elif info["kind"] == "gitlab":
        r = await _git(
            workspace,
            f'glab mr create --source-branch "{res.branch}" '
            f'--target-branch "{res.base}" --title "{subject}" --fill --yes',
            timeout=120)
        if r.exit_code == 0:
            m = re.search(r"https?://\S+", r.stdout or "")
            return m.group(0) if m else ""
    return ""


__all__ = [
    "GitResult", "slugify", "branch_name", "commit_subject", "commit_message",
    "parse_remote", "compare_url", "run_git_workflow",
]
