"""Patch-based follow-up edits for chat code (vNext §3.5).

"Now add error handling" should not regenerate the whole answer: the model emits
targeted **`str_replace` patches** (schema-enforced §8.7) against the previous
code block, we apply them exactly, and only the changed file is re-verified with
the compile cache warm. A patch that fails to apply falls back to full
regeneration (exactly §8.8). Iteration drops from ~10 s to ~2-3 s and code the
user didn't ask to change can no longer silently mutate between rounds.

The "is this an edit to the code we just produced?" decision is INTENT, so it is
SEMANTIC — `app.semantics.gates` (`code_edit_request`) is the authority; a tiny
cue list is only the cold-start fallback (matching the app's semantic-gates
architecture). The patch APPLICATION is pure string surgery; the fence parse is
parsing, not intent. `chat → semantics` / `chat → core` edges already exist.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_FENCE = re.compile(r"```([\w+#.\-]*)[ \t]*\n(.*?)```", re.DOTALL)

# Schema for the model's patch payload (validated via app.core.structured §8.7).
CODE_PATCH_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "patches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
                "required": ["old_str", "new_str"],
            },
        },
    },
    "required": ["patches"],
}

# Cold-start fallback ONLY (used when the embedder is unavailable): a short
# imperative change against existing code. The semantic gate is the authority.
_FALLBACK_EDIT = re.compile(
    r"^\s*(?:now\s+|also\s+|and\s+|please\s+)*"
    r"(add|make|fix|change|refactor|rename|use|handle|wrap|optimi[sz]e|remove|"
    r"replace|update|convert|extract|inline|simplify|rewrite)\b", re.I)


@dataclass
class PatchResult:
    code: str
    applied: bool
    applied_count: int = 0
    failures: list[str] = field(default_factory=list)


def apply_str_replace(code: str, patches: list[dict]) -> PatchResult:
    """Apply each ``{old_str → new_str}`` to ``code`` (first occurrence). ALL
    patches must find their target; if any ``old_str`` is missing the edit is
    ambiguous/stale → ``applied=False`` with the ORIGINAL code (caller falls back
    to a full regeneration). Never raises."""
    cur = code or ""
    if not patches:
        return PatchResult(cur, False, 0, ["no patches"])
    out = cur
    count = 0
    fails: list[str] = []
    for p in patches:
        if not isinstance(p, dict):
            fails.append("malformed patch")
            continue
        old = str(p.get("old_str") or "")
        new = str(p.get("new_str") or "")
        if not old:
            fails.append("empty old_str")
            continue
        if old not in out:
            fails.append(f"target not found: {old[:48]!r}")
            return PatchResult(code or "", False, count, fails)  # → full regen
        out = out.replace(old, new, 1)
        count += 1
    return PatchResult(out, count > 0 and not fails, count, fails)


def extract_last_code_block(text: str) -> tuple[str, str] | None:
    """The LAST substantial (≥ 2 non-blank lines) fenced code block → ``(code,
    label)`` — the block a follow-up edit targets. None when there isn't one."""
    found: tuple[str, str] | None = None
    for m in _FENCE.finditer(text or ""):
        label = (m.group(1) or "").strip()
        body = (m.group(2) or "").rstrip("\n")
        if len([ln for ln in body.splitlines() if ln.strip()]) >= 2:
            found = (body, label)
    return found


def is_edit_request(text: str, *, has_prior_code: bool) -> bool:
    """Whether this turn edits the code just produced (→ patch, not regenerate).
    SEMANTIC-first: the `code_edit_request` gate is the authority; the cue regex
    is only the cold-start fallback. Requires prior code to edit."""
    if not has_prior_code:
        return False
    t = (text or "").strip()
    if not t:
        return False
    try:
        from app.semantics import gates
        verdict = gates.matches("code_edit_request", t)
        if verdict is not None:
            return bool(verdict)          # embedder answered → trust it
    except Exception:  # noqa: BLE001
        pass
    # Cold-start fallback only.
    return bool(_FALLBACK_EDIT.match(t)) and len(t.split()) <= 16


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.code_solver, "patch_edits", False))
    except Exception:  # noqa: BLE001
        return False


__all__ = [
    "CODE_PATCH_SCHEMA", "PatchResult", "apply_str_replace",
    "extract_last_code_block", "is_edit_request", "enabled",
]
