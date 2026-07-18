"""Artifact runtime (workspace-and-artifacts spec).

An Artifact is a first-class, addressable, versioned, editable output (document,
markdown, diagram, code file, SQL, HTML) that the user works *with* across turns.
This package reuses the existing document generators + blob store rather than a
new renderer, and exposes the Current_Artifact to the follow-up engine.

All entry points are additive + fail-open: with `cfg.workspace.artifacts_enabled`
off (or on any error) no artifact is created and behavior is byte-for-byte today's.
"""
from .discipline import should_create_artifact, ARTIFACT_KINDS
from .patch import apply_patch
from .store import ArtifactStore, Artifact, ArtifactVersion, artifact_store

__all__ = [
    "should_create_artifact", "ARTIFACT_KINDS", "apply_patch",
    "ArtifactStore", "Artifact", "ArtifactVersion", "artifact_store",
]
