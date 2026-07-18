"""Universal doc & code transformation bridge (roadmap Phase 4 #20/#21).

Wires the already-built unified transform flow (`app/documents/transform.py`:
parse → transform → format → validate) into the artifact pipeline from THIS
side, so a produced or uploaded document/code artifact runs the one orchestrated
transform and lands as a validated version in the `ArtifactStore`.

Also exposes the resume-templating renderer (`documents/templates.render_resume`,
#21) as a transformer so a resume artifact is rendered into a clean, ATS-safe
template as part of the same flow.

Everything is fail-open: if the documents layer isn't importable (it lives in
another area), the bridge degrades to an identity transform and the raw content
is stored unchanged — never an exception.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable


async def run_transform(content: str, *, filename: str = "",
                        transformer: Callable[[str], Awaitable[str]] | None = None,
                        do_validate: bool = True):
    """Run the unified transform flow over in-memory content (#20). Returns a
    `TransformResult` or None if the documents layer is unavailable."""
    try:
        from app.documents.transform import transform_content
    except Exception:  # noqa: BLE001
        return None
    try:
        return await transform_content(
            content, filename=filename, transformer=transformer,
            do_validate=do_validate)
    except Exception:  # noqa: BLE001
        return None


def resume_transformer(template: str = "classic"
                       ) -> Callable[[str], Awaitable[str]]:
    """A transformer that renders markdown resume content into a clean template
    via `documents/templates.render_resume` (#21). Fail-open — identity if the
    templating layer is unavailable."""
    async def _tx(content: str) -> str:
        try:
            from app.documents.templates import (
                apply_template,
                sections_from_markdown,
                render_resume,
            )
            # Prefer the structured render; fall back to apply_template.
            sections = sections_from_markdown(content)
            if sections:
                return render_resume(sections, template)
            return apply_template(content, template)
        except Exception:  # noqa: BLE001
            return content
    return _tx


async def transform_and_store(store, workspace_id: str, content: str, *,
                              title: str = "Untitled", filename: str = "",
                              kind: str | None = None,
                              transformer: Callable[[str], Awaitable[str]] | None = None):
    """Run the unified transform, then persist the validated result as an
    artifact version in `store` (#20). Returns (artifact, TransformResult) or
    (None, None) on failure. Fail-open."""
    res = await run_transform(content, filename=filename,
                              transformer=transformer)
    if res is None:
        # Degrade: still store the raw content so nothing is lost.
        try:
            art = await store.create(workspace_id, kind or "document", title,
                                     content, "md")
            return art, None
        except Exception:  # noqa: BLE001
            return None, None
    try:
        art = await store.create(
            workspace_id, kind or res.kind, title, res.content,
            res.ext or "md")
        return art, res
    except Exception:  # noqa: BLE001
        return None, res


__all__ = ["run_transform", "resume_transformer", "transform_and_store"]
