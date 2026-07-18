"""Current-artifact bridge (workspace-and-artifacts R7, task 7.2).

Pins Property 7: the Current_Artifact is a resolvable entity; a follow-up
referencing it targets that artifact; with no current artifact → passthrough.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.followup.state import ConversationState
from app.followup.reference import resolve
from app.artifacts.bridge import register_current_artifact, resolve_target_artifact


@dataclass
class _Art:
    id: str
    kind: str
    title: str


def test_artifact_registered_as_resolvable_entity():
    s = ConversationState({}, "c1")
    register_current_artifact(s, _Art("art1", "diagram", "Auth Flow"))
    cur = s.current_artifact()
    assert cur and cur["id"] == "art1"
    # The kind word + title are entities the resolver can bind to.
    assert any("diagram" in e.lower() for e in s.entities())


def test_followup_targets_current_artifact_by_kind():
    s = ConversationState({}, "c1")
    register_current_artifact(s, _Art("art1", "diagram", "Auth Flow"))
    res = resolve("update the diagram with a logout step", s)
    target = resolve_target_artifact(s, res)
    assert target == "art1"


def test_pronoun_followup_targets_current_artifact():
    s = ConversationState({}, "c1")
    register_current_artifact(s, _Art("art2", "document", "Spec"))
    res = resolve("add a section to it", s)
    assert resolve_target_artifact(s, res) == "art2"


def test_no_current_artifact_is_passthrough():
    s = ConversationState({}, "c1")
    res = resolve("add a section to it", s)
    assert resolve_target_artifact(s, res) is None


def test_unrelated_reference_does_not_target_artifact():
    s = ConversationState({}, "c1")
    register_current_artifact(s, _Art("art1", "diagram", "Auth Flow"))
    # Resolves to a different entity, not the artifact.
    s.add_entity("PostgreSQL")
    res = resolve("tune the PostgreSQL settings", s)
    assert resolve_target_artifact(s, res) != "art1"
