"""Diff-edits engine (vNext §8.8, Stage 8 Component A).

Editing a generated artifact should not regenerate it whole — the model emits
targeted `str_replace` patches (Anthropic text-editor semantics), which apply
ATOMICALLY as a new VERSION over `app/documents/store.py`, and the FE streams the
resulting diff. A patch that doesn't match cleanly is REJECTED without mutating
the artifact — an edit is all-or-nothing, never a half-applied document.

This module owns the deterministic core:
  * `apply_patches(content, patches)` — atomic, order-sensitive apply with strict
    match rules (unique unless `replace_all`); any failure → the ORIGINAL content
    unchanged + a reason;
  * `build_patches(instruction, content, structured_fn)` — turn a natural-language
    edit into schema-enforced patches via an INJECTED structured call (unit-tested
    with no LLM, same seam as `chat.interpret` / `memory.compaction`);
  * `diff_summary(before, after)` — a unified-diff summary for the streamed FE view.

The store glue (`apply_and_version`) is thin + fail-open. Flag-gated
(`documents.diff_edits`, default OFF → today's regenerate-whole behaviour).
"""
from __future__ import annotations

import difflib
from dataclasses import dataclass, field


def enabled() -> bool:
    try:
        from app.core.config_loader import cfg
        return bool(getattr(cfg.documents, "diff_edits", False))
    except Exception:  # noqa: BLE001
        return False


# Schema (§8.7) for the model's patch emission.
PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "patches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["old_str", "new_str"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["patches"],
    "additionalProperties": False,
}


@dataclass
class Patch:
    old_str: str
    new_str: str
    replace_all: bool = False


@dataclass
class PatchResult:
    ok: bool
    content: str                       # new content if ok, else the ORIGINAL
    applied: int = 0                   # patches applied
    changed: bool = False              # content actually differs
    rejected_index: int = -1           # which patch failed (−1 = none)
    reason: str = ""

    def to_dict(self) -> dict:
        return {"ok": self.ok, "applied": self.applied, "changed": self.changed,
                "rejected_index": self.rejected_index, "reason": self.reason}


def _coerce_patches(patches) -> "list[Patch]":
    out: list[Patch] = []
    for p in patches or ():
        if isinstance(p, Patch):
            out.append(p)
        elif isinstance(p, dict):
            out.append(Patch(old_str=str(p.get("old_str", "")),
                             new_str=str(p.get("new_str", "")),
                             replace_all=bool(p.get("replace_all", False))))
    return out


def apply_patches(content: str, patches) -> PatchResult:
    """Apply `str_replace` patches ATOMICALLY. Each `old_str` must be present and
    UNIQUE (unless `replace_all`); patches apply in order over the evolving text.
    Any failure aborts the whole edit → the ORIGINAL content, ok=False, and the
    failing index + reason. Never raises."""
    try:
        original = content or ""
        work = original
        plist = _coerce_patches(patches)
        if not plist:
            return PatchResult(False, original, 0, False, -1, "no patches")
        applied = 0
        for i, p in enumerate(plist):
            if p.old_str == "":
                return PatchResult(False, original, 0, False, i,
                                   "empty old_str cannot match")
            count = work.count(p.old_str)
            if count == 0:
                return PatchResult(False, original, 0, False, i,
                                   "old_str not found")
            if count > 1 and not p.replace_all:
                return PatchResult(False, original, 0, False, i,
                                   f"old_str is ambiguous ({count} matches) — "
                                   "set replace_all or add context")
            if p.replace_all:
                work = work.replace(p.old_str, p.new_str)
            else:
                work = work.replace(p.old_str, p.new_str, 1)
            applied += 1
        return PatchResult(True, work, applied, work != original, -1, "")
    except Exception as exc:  # noqa: BLE001 — never half-apply on an error
        return PatchResult(False, content or "", 0, False, -1, f"error: {exc}")


async def build_patches(instruction: str, content: str, *,
                        structured_fn=None) -> "list[Patch]":
    """Turn a natural-language edit `instruction` into `str_replace` patches via a
    schema-enforced structured call (INJECTED `structured_fn`, default the
    `core.structured` facade — same seam as `interpret.build_brief`). Fail-open:
    disabled OR any error → [] (the caller keeps the artifact unchanged)."""
    if not enabled() or not (instruction or "").strip() or not (content or "").strip():
        return []
    try:
        fn = structured_fn
        if fn is None:
            from app.core.structured import structured as fn  # type: ignore
        msgs = [
            {"role": "system", "content": _PATCH_INSTRUCTION},
            {"role": "user", "content":
                f"DOCUMENT:\n{(content or '')[:16000]}\n\nEDIT:\n{instruction}"},
        ]
        res = await fn(PATCH_SCHEMA, msgs, tier="standard", name="diff_edit")
        obj = getattr(res, "obj", None)
        if not isinstance(obj, dict):
            return []
        return _coerce_patches(obj.get("patches"))
    except Exception:  # noqa: BLE001
        return []


_PATCH_INSTRUCTION = (
    "You edit a document with minimal, targeted string replacements. Emit a list "
    "of patches, each an EXACT `old_str` copied verbatim from the document and its "
    "`new_str` replacement. Make `old_str` long enough to be UNIQUE in the "
    "document (include surrounding context); set `replace_all` only to change "
    "every occurrence. Change only what the edit asks; never rewrite the whole "
    "document. Preserve formatting and indentation exactly.")


def diff_summary(before: str, after: str) -> dict:
    """A compact unified-diff summary for the FE streamed diff view: added/removed
    line counts + the unified-diff hunks. Never raises."""
    try:
        b = (before or "").splitlines()
        a = (after or "").splitlines()
        diff = list(difflib.unified_diff(b, a, lineterm="", n=2))
        added = sum(1 for ln in diff if ln.startswith("+") and not ln.startswith("+++"))
        removed = sum(1 for ln in diff if ln.startswith("-") and not ln.startswith("---"))
        return {"added": added, "removed": removed,
                "changed": added > 0 or removed > 0, "hunks": diff}
    except Exception:  # noqa: BLE001
        return {"added": 0, "removed": 0, "changed": False, "hunks": []}


async def apply_and_version(session_id, doc_key, instruction: str, *,
                            structured_fn=None, title: str = "", fmt: str = "",
                            patches=None):
    """Load the latest content of `doc_key`, apply the edit as patches, and save
    the result as the NEXT version via `store.save_version`. Returns
    `(PatchResult, new_doc_key)`. Fully fail-open: disabled / no store / no match
    → a not-ok result and the artifact untouched (no new version written)."""
    from app.documents import store as _store
    if not enabled():
        return PatchResult(False, "", 0, False, -1, "disabled"), doc_key
    try:
        from storage.db import get_session_factory
        from storage.models import GeneratedDocument
        from sqlalchemy import func, select
        factory = get_session_factory()
        key = _store._as_uuid(doc_key)
        if factory is None or key is None:
            return PatchResult(False, "", 0, False, -1, "no store"), doc_key
        async with factory() as s:
            latest = await _store.latest_version(s, key)
            if latest is None:
                return PatchResult(False, "", 0, False, -1, "no such doc"), doc_key
            content = latest.content_md or ""
            plist = patches if patches is not None else await build_patches(
                instruction, content, structured_fn=structured_fn)
            result = apply_patches(content, plist)
            if not result.ok or not result.changed:
                return result, doc_key      # nothing written on reject/no-op
            row = await _store.save_version(
                s, latest.session_id, result.content,
                title=title or (latest.title or ""),
                fmt=fmt or (latest.doc_format or "pdf"), doc_key=key)
            await s.commit()
            return result, row.doc_key
    except Exception as exc:  # noqa: BLE001 — an edit must never break a turn
        return PatchResult(False, "", 0, False, -1, f"error: {exc}"), doc_key


__all__ = ["enabled", "PATCH_SCHEMA", "Patch", "PatchResult", "apply_patches",
           "build_patches", "diff_summary", "apply_and_version"]
