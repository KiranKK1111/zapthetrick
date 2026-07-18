"""Current-artifact bridge to the follow-up engine (workspace-and-artifacts R7).

Registers an Artifact as the conversation's Current_Artifact in the
`followup-context-engine` ConversationState so a vague follow-up ("add a section
to it", "change the diagram") binds to that Artifact for an in-place edit (R7.1/
R7.2). With no Current_Artifact the turn is a normal answer (R7.3). Fail-open.
"""
from __future__ import annotations

# Generic words a follow-up may use to refer to the open artifact.
_KIND_WORDS = {
    "document": "document", "markdown": "document", "code": "code file",
    "diagram": "diagram", "sql": "schema", "html": "page",
}


def register_current_artifact(state, artifact) -> None:
    """Expose `artifact` as the conversation's Current_Artifact entity. Never
    raises (R7.3 / Property 7/9)."""
    try:
        if state is None or artifact is None:
            return
        kind_word = _KIND_WORDS.get(getattr(artifact, "kind", ""), "")
        state.set_current_artifact(
            getattr(artifact, "id", ""),
            title=getattr(artifact, "title", "") or None,
            kind=kind_word or None,
        )
    except Exception:  # noqa: BLE001
        pass


def resolve_target_artifact(state, resolution) -> str | None:
    """If a follow-up resolved a reference that matches the Current_Artifact
    (by title or kind word), return that artifact id to target for an edit; else
    None (→ normal answer, R7.2/R7.3). Fail-open."""
    try:
        cur = state.current_artifact() if state is not None else None
        if not cur:
            return None
        ants = [a.lower() for a in getattr(resolution, "antecedents", [])]
        title = (cur.get("title") or "").lower()
        kind = (cur.get("kind") or "").lower()
        if not ants:
            # A pronoun follow-up with a current artifact still targets it.
            refs = getattr(resolution, "refs", [])
            return cur.get("id") if refs else None
        if any(title and title in a or kind and kind in a or a in (title, kind)
               for a in ants):
            return cur.get("id")
        return None
    except Exception:  # noqa: BLE001
        return None


__all__ = ["register_current_artifact", "resolve_target_artifact"]
