"""Project brain + decision ledger + learning-lite (Phase 11, #18/#59/#14).

A per-workspace, file-based memory so the chat agent has CONTINUITY across
turns on the same project (Flow C reuses the workspace):

  <workspace>/.zapthetrick/brain.md     — human-readable project memory:
        ## Project Facts, ## Decisions (the ledger), ## Conventions
  <workspace>/.zapthetrick/learning.json — tiny stats: which model/approach
        succeeded, so future runs can be biased toward it.

No DB, no migration, no LLM — pure files, so it ships inside the downloadable
project zip (the user keeps the decision log) and is trivially testable. The
agent reads `brain_context()` at the start of each run and records decisions via
the `record_decision` tool; the run loop appends an auto-summary + remembers the
model outcome at the end.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

_DIR = ".zapthetrick"
_BRAIN = "brain.md"
_LEARN = "learning.json"
_SCRATCH = "scratchpad.md"

_TEMPLATE = """\
# Project Brain

_Auto-maintained memory for this project. The agent reads this at the start of
every run and records key decisions here, so follow-up requests keep context._

## Project Facts
- (none recorded yet)

## Decisions
<!-- The decision ledger — newest entries appended below. -->

## Conventions
- (none recorded yet)
"""


def _dir(workspace: str) -> str:
    d = os.path.join(os.path.realpath(workspace), _DIR)
    os.makedirs(d, exist_ok=True)
    return d


def brain_path(workspace: str) -> str:
    return os.path.join(os.path.realpath(workspace), _DIR, _BRAIN)


def _learn_path(workspace: str) -> str:
    return os.path.join(os.path.realpath(workspace), _DIR, _LEARN)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


# --------------------------------------------------------------------------
# brain.md
# --------------------------------------------------------------------------
def read_brain(workspace: str) -> str:
    try:
        with open(brain_path(workspace), encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def ensure_brain(workspace: str, *, project: str | None = None) -> str:
    """Create the brain file from a template if missing. Returns its path."""
    p = brain_path(workspace)
    if not os.path.isfile(p):
        _dir(workspace)
        body = _TEMPLATE
        if project:
            body = body.replace("- (none recorded yet)\n\n## Decisions",
                                 f"- Project: {project}\n\n## Decisions", 1)
        try:
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                f.write(body)
        except OSError:
            pass
    return p


def record_decision(workspace: str, title: str, decision: str,
                    rationale: str = "") -> bool:
    """Append one entry to the `## Decisions` ledger. Returns True on success."""
    title = (title or "").strip()
    decision = (decision or "").strip()
    if not title and not decision:
        return False
    ensure_brain(workspace)
    p = brain_path(workspace)
    try:
        with open(p, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return False
    entry = f"\n- **{title or 'Decision'}** ({_now()}) — {decision}"
    if rationale.strip():
        entry += f"\n  - _Rationale:_ {rationale.strip()}"
    marker = "## Decisions"
    if marker in text:
        idx = text.index(marker) + len(marker)
        # Insert right after the marker's comment line (or the marker itself).
        nl = text.find("\n", idx)
        insert_at = nl + 1 if nl != -1 else len(text)
        # skip the template comment line if present
        if text[insert_at:].lstrip().startswith("<!--"):
            cend = text.find("\n", insert_at)
            insert_at = cend + 1 if cend != -1 else insert_at
        text = text[:insert_at] + entry + "\n" + text[insert_at:]
    else:
        text += f"\n\n## Decisions{entry}\n"
    try:
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        return True
    except OSError:
        return False


def brain_context(workspace: str, *, max_chars: int = 2000) -> str:
    """A compact 'PROJECT MEMORY' preamble for the agent loop, or '' if the
    brain is empty / still the pristine template. Includes the learning hint
    (preferred model)."""
    raw = read_brain(workspace).strip()
    parts: list[str] = []
    # Only inject when there's REAL content: a recorded decision, or a project
    # fact beyond the template placeholder.
    has_facts = bool(raw) and "- (none recorded yet)" not in raw.split(
        "## Decisions", 1)[0]
    if raw and (_has_decisions(raw) or has_facts):
        parts.append(
            "PROJECT MEMORY (from earlier work on this project — use it for "
            "continuity; don't contradict prior decisions without reason):\n"
            + raw[:max_chars])
    hint = learning_hint(workspace)
    if hint:
        parts.append(hint)
    return "\n\n".join(parts).strip()


def _has_decisions(text: str) -> bool:
    if "## Decisions" not in text:
        return False
    tail = text.split("## Decisions", 1)[1]
    # Only the Decisions section — stop at the next "## " heading (Conventions).
    for ln in tail.splitlines():
        s = ln.strip()
        if s.startswith("## "):
            break
        if s.startswith("- "):
            return True
    return False


# --------------------------------------------------------------------------
# learning.json — remember which model succeeded (learning-lite)
# --------------------------------------------------------------------------
def _load_learn(workspace: str) -> dict:
    try:
        with open(_learn_path(workspace), encoding="utf-8") as f:
            obj = json.load(f)
            return obj if isinstance(obj, dict) else {}
    except (OSError, ValueError):
        return {}


def remember_run(workspace: str, *, model: str | None, kind: str = "edit",
                 success: bool = False) -> None:
    """Record a run's (model, outcome) so we can prefer what works."""
    if not model:
        return
    data = _load_learn(workspace)
    models = data.setdefault("models", {})
    m = models.setdefault(model, {"ok": 0, "total": 0})
    m["total"] += 1
    if success:
        m["ok"] += 1
    data["updated"] = _now()
    data.setdefault("kind", kind)
    try:
        _dir(workspace)
        with open(_learn_path(workspace), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def preferred_model(workspace: str) -> str | None:
    """The model with the best success record on this project (>=1 success)."""
    models = _load_learn(workspace).get("models", {})
    best, best_key = None, None
    for name, s in models.items():
        ok = int(s.get("ok", 0))
        if ok <= 0:
            continue
        rate = ok / max(1, int(s.get("total", 1)))
        score = (ok, rate)
        if best is None or score > best:
            best, best_key = score, name
    return best_key


def learning_hint(workspace: str) -> str:
    m = preferred_model(workspace)
    return (f"Note: on this project, runs have succeeded most reliably with "
            f"model `{m}`.") if m else ""


# --------------------------------------------------------------------------
# scratchpad.md — cross-step working notes (P2-3)
# --------------------------------------------------------------------------
# A short-lived "what I've learned this task" log so a long-horizon run (and
# each fresh round of the goal loop, which starts a clean agent) doesn't
# re-explore the same files. The agent appends findings via the `note` tool;
# the goal loop injects `scratchpad_context` into every round. It's per-task,
# so the runner clears it at the start of a fresh build/edit.
def _scratch_path(workspace: str) -> str:
    return os.path.join(os.path.realpath(workspace), _DIR, _SCRATCH)


_MAX_SCRATCH_NOTES = 60


def append_scratchpad(workspace: str, note: str, *, tag: str = "") -> bool:
    """Append one working note (deduped against the last few). Returns success."""
    note = (note or "").strip()
    if not note:
        return False
    _dir(workspace)
    p = _scratch_path(workspace)
    line = f"- {('[' + tag.strip() + '] ') if tag.strip() else ''}{note}"
    try:
        existing = ""
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                existing = f.read()
        # de-dupe: skip if this exact note is already in the last 10 lines
        recent = existing.strip().splitlines()[-10:]
        if line in recent:
            return True
        lines = [ln for ln in existing.splitlines() if ln.strip()]
        lines.append(line)
        if len(lines) > _MAX_SCRATCH_NOTES:
            lines = lines[-_MAX_SCRATCH_NOTES:]
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write("# Working Notes\n" if not lines[0].startswith("#") else "")
            f.write("\n".join(lines) + "\n")
        return True
    except OSError:
        return False


def read_scratchpad(workspace: str, *, max_chars: int = 1500) -> str:
    try:
        with open(_scratch_path(workspace), encoding="utf-8") as f:
            text = f.read().strip()
    except OSError:
        return ""
    # keep the most RECENT notes within the char budget
    if len(text) > max_chars:
        text = "…\n" + text[-max_chars:]
    return text


def scratchpad_context(workspace: str, *, max_chars: int = 1500) -> str:
    """A 'WORKING NOTES' preamble for the loop, or '' when empty."""
    raw = read_scratchpad(workspace, max_chars=max_chars)
    body = "\n".join(ln for ln in raw.splitlines()
                     if ln.strip() and not ln.lstrip().startswith("#"))
    if not body.strip():
        return ""
    return ("WORKING NOTES (facts you established earlier this task — rely on "
            "them; don't re-read files you've already summarized here):\n" + body)


def clear_scratchpad(workspace: str) -> None:
    try:
        os.remove(_scratch_path(workspace))
    except OSError:
        pass


__all__ = [
    "brain_path", "read_brain", "ensure_brain", "record_decision",
    "brain_context", "remember_run", "preferred_model", "learning_hint",
    "append_scratchpad", "read_scratchpad", "scratchpad_context",
    "clear_scratchpad",
]
