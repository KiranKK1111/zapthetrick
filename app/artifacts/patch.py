"""Artifact patching (workspace-and-artifacts R6).

`apply_patch(current, instruction) -> (next_content, applied)` makes a targeted,
deterministic edit to the current artifact content so "add a section" / "append
X" / "replace A with B" update the document IN PLACE rather than regenerating it
(R6.1). When the instruction can't be applied deterministically, it returns
``applied=False`` so the caller regenerates and records that (R6.2, Property 6).
Pure; never raises.
"""
from __future__ import annotations

import re

# "replace <a> with <b>" / "change <a> to <b>".
_REPLACE_RE = re.compile(
    r"\b(?:replace|change)\s+['\"]?(.+?)['\"]?\s+(?:with|to)\s+['\"]?(.+?)['\"]?\s*$",
    re.I)
# "add|append|include <text>" / "add a section about <text>".
_APPEND_RE = re.compile(
    r"\b(?:add|append|include|also add)\b(?:\s+a\s+section(?:\s+(?:about|on|for))?)?"
    r"[:\s]+(.+)$", re.I)
# "remove|delete <text>".
_REMOVE_RE = re.compile(r"\b(?:remove|delete|drop)\b[:\s]+(.+)$", re.I)


def apply_patch(current: str, instruction: str) -> tuple[str, bool]:
    """Return ``(next_content, applied)``. Never raises."""
    try:
        return _apply(current, instruction)
    except Exception:  # noqa: BLE001
        return current, False


def _apply(current: str, instruction: str) -> tuple[str, bool]:
    cur = current or ""
    instr = " ".join((instruction or "").split())
    if not instr:
        return cur, False

    # 1) Replace / change A → B (literal, first occurrence).
    m = _REPLACE_RE.search(instr)
    if m:
        a, b = m.group(1).strip(), m.group(2).strip()
        if a and a in cur:
            return cur.replace(a, b, 1), True
        return cur, False        # target not found → regenerate (R6.2)

    # 2) Remove / delete a line or phrase.
    m = _REMOVE_RE.search(instr)
    if m:
        target = m.group(1).strip().rstrip(".")
        if target and target in cur:
            # Drop whole lines that contain the target, else the phrase.
            lines = cur.splitlines()
            kept = [ln for ln in lines if target.lower() not in ln.lower()]
            if len(kept) != len(lines):
                return "\n".join(kept), True
            return cur.replace(target, "", 1), True
        return cur, False

    # 3) Add / append a section or text (the common "add X to it").
    m = _APPEND_RE.search(instr)
    if m:
        addition = m.group(1).strip()
        if addition:
            sep = "" if cur.endswith("\n\n") or not cur else "\n\n"
            # If the addition reads like a section title, render it as a heading.
            if len(addition.split()) <= 8 and not addition.endswith((".", ":")):
                block = f"## {addition[0].upper()}{addition[1:]}\n\n_(section added)_"
            else:
                block = addition
            return f"{cur}{sep}{block}", True

    # 4) Unrecognized edit shape → caller regenerates (R6.2).
    return cur, False


__all__ = ["apply_patch"]
