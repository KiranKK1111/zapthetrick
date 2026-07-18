"""Artifact store + versioning (workspace-and-artifacts R5).

`ArtifactStore` keeps artifact metadata + an immutable, monotonically-numbered
version chain; version BYTES persist via the injected blob store (the existing
`storage/blobs`), metadata in-process (bounded). Prior versions are restorable
and retained versions are bounded per artifact, evicting oldest-first (R5.3,
Property 5).

The blob backend is injected so the store is testable with no DB/blob config:
with no backend it uses an in-process byte map (fail-open). The process-wide
`artifact_store()` singleton wires the real blob store when available.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class ArtifactVersion:
    version: int
    blob_ref: str
    fmt: str
    created: float = 0.0


@dataclass
class Artifact:
    id: str
    workspace_id: str
    kind: str
    title: str
    versions: list[ArtifactVersion] = field(default_factory=list)
    _seq: int = 0          # monotonic version counter (never reused)
    # #10 dependency graph: ids this artifact is DERIVED FROM. When an upstream
    # dependency changes, this artifact is marked `stale` (needs regeneration).
    depends_on: set = field(default_factory=set)
    stale: bool = False
    stale_reason: str = ""

    @property
    def current(self) -> ArtifactVersion | None:
        return self.versions[-1] if self.versions else None

    @property
    def current_version(self) -> int:
        return self.versions[-1].version if self.versions else 0


def _max_versions() -> int:
    try:
        from app.core.config_loader import cfg
        return max(1, int(getattr(cfg.workspace, "max_artifact_versions", 20)))
    except Exception:  # noqa: BLE001
        return 20


def _as_bytes(content) -> bytes:
    if isinstance(content, bytes):
        return content
    return str(content or "").encode("utf-8")


class ArtifactStore:
    def __init__(self, blob_put=None, blob_get=None, max_versions: int | None = None):
        self._artifacts: dict[str, Artifact] = {}
        self._mem: dict[str, bytes] = {}        # fallback byte store (no blob config)
        self._blob_put = blob_put
        self._blob_get = blob_get
        self._max = max_versions

    def _cap(self) -> int:
        return self._max if self._max is not None else _max_versions()

    async def _put(self, ref: str, data: bytes) -> str:
        if self._blob_put is not None:
            return await self._blob_put(ref, data)
        self._mem[ref] = data
        return ref

    async def _get(self, ref: str) -> bytes:
        if self._blob_get is not None:
            return await self._blob_get(ref)
        return self._mem.get(ref, b"")

    # ---- writes ----------------------------------------------------------
    async def create(self, workspace_id: str, kind: str, title: str,
                     content, fmt: str) -> Artifact:
        art = Artifact(id=uuid.uuid4().hex, workspace_id=workspace_id or "default",
                       kind=kind, title=title or "Untitled")
        self._artifacts[art.id] = art
        await self._append(art, content, fmt)
        return art

    async def append_version(self, artifact_id: str, content, fmt: str | None = None
                             ) -> ArtifactVersion | None:
        art = self._artifacts.get(artifact_id)
        if art is None:
            return None
        return await self._append(art, content, fmt or (
            art.current.fmt if art.current else "md"))

    async def _append(self, art: Artifact, content, fmt: str) -> ArtifactVersion:
        art._seq += 1
        ref = f"artifacts/{art.id}/v{art._seq}"
        stored = await self._put(ref, _as_bytes(content))
        ver = ArtifactVersion(version=art._seq, blob_ref=stored, fmt=fmt,
                              created=time.time())
        art.versions.append(ver)
        # Retention bound: evict oldest versions past the cap (R5.3).
        cap = self._cap()
        while len(art.versions) > cap:
            old = art.versions.pop(0)
            if self._blob_put is None:
                self._mem.pop(old.blob_ref, None)
        # #10 — this artifact just changed: mark every DEPENDENT as stale, and
        # clear this one's own stale flag (it's now fresh).
        art.stale = False
        art.stale_reason = ""
        self._mark_dependents_stale(art.id)
        return ver

    # ---- #10 dependency graph + stale-marking --------------------------------
    def add_dependency(self, artifact_id: str, depends_on: str) -> bool:
        """Record that `artifact_id` is derived from `depends_on` (#10)."""
        art = self._artifacts.get(artifact_id)
        if art is None or not depends_on or depends_on == artifact_id:
            return False
        art.depends_on.add(depends_on)
        return True

    def _mark_dependents_stale(self, changed_id: str) -> list[str]:
        """Mark (transitively) every artifact that depends on `changed_id`
        stale. Returns the ids newly marked."""
        marked: list[str] = []
        frontier = [changed_id]
        seen = {changed_id}
        while frontier:
            cur = frontier.pop()
            for aid, art in self._artifacts.items():
                if aid in seen:
                    continue
                if cur in art.depends_on and not art.stale:
                    art.stale = True
                    art.stale_reason = f"dependency {cur[:8]} changed"
                    marked.append(aid)
                    seen.add(aid)
                    frontier.append(aid)
        return marked

    def is_stale(self, artifact_id: str) -> bool:
        art = self._artifacts.get(artifact_id)
        return bool(art and art.stale)

    def stale_dependents(self, artifact_id: str) -> list[str]:
        """Ids currently stale because they (transitively) depend on this one."""
        out: list[str] = []
        for aid, art in self._artifacts.items():
            if aid != artifact_id and art.stale and artifact_id in art.depends_on:
                out.append(aid)
        return out

    # ---- #11 incremental patch (wires artifacts/patch.apply_patch) -----------
    async def patch_version(self, artifact_id: str, instruction: str
                            ) -> tuple[ArtifactVersion | None, bool]:
        """Incrementally edit the current version IN PLACE from an NL
        instruction (#11): apply the targeted patch and, only if it applied,
        append the patched content as a new version. Returns
        `(version|None, applied)`; `applied=False` signals the caller to
        regenerate instead. Fail-open."""
        art = self._artifacts.get(artifact_id)
        if art is None or art.current is None:
            return None, False
        try:
            from app.artifacts.patch import apply_patch
            cur_bytes = await self._get(art.current.blob_ref)
            cur_text = cur_bytes.decode("utf-8", errors="replace")
            new_text, applied = apply_patch(cur_text, instruction)
            if not applied:
                return None, False
            ver = await self._append(art, new_text, art.current.fmt)
            return ver, True
        except Exception:  # noqa: BLE001
            return None, False

    # ---- #17 universal undo --------------------------------------------------
    async def undo(self, artifact_id: str) -> ArtifactVersion | None:
        """Undo the most recent change by restoring the PREVIOUS version as a
        new version (#17) — a non-destructive, universally-applicable undo that
        preserves the immutable chain. No prior version → None."""
        art = self._artifacts.get(artifact_id)
        if art is None or len(art.versions) < 2:
            return None
        prev = art.versions[-2]
        data = await self._get(prev.blob_ref)
        return await self._append(art, data, prev.fmt)

    async def restore(self, artifact_id: str, version: int) -> ArtifactVersion | None:
        """Restore an earlier version by appending its bytes as a NEW version,
        preserving the immutable chain + identity (R5.2/R6.3)."""
        art = self._artifacts.get(artifact_id)
        if art is None:
            return None
        target = next((v for v in art.versions if v.version == version), None)
        if target is None:
            return None
        data = await self._get(target.blob_ref)
        return await self._append(art, data, target.fmt)

    # ---- reads -----------------------------------------------------------
    def get(self, artifact_id: str) -> Artifact | None:
        return self._artifacts.get(artifact_id)

    def versions(self, artifact_id: str) -> list[ArtifactVersion]:
        art = self._artifacts.get(artifact_id)
        return list(art.versions) if art else []

    async def content(self, artifact_id: str, version: int | None = None) -> bytes:
        art = self._artifacts.get(artifact_id)
        if art is None or not art.versions:
            return b""
        ver = (art.current if version is None
               else next((v for v in art.versions if v.version == version), None))
        if ver is None:
            return b""
        return await self._get(ver.blob_ref)

    def reset(self) -> None:
        self._artifacts.clear()
        self._mem.clear()


# Process-wide singleton (wires the real blob store lazily; falls back to the
# in-process byte map when blobs aren't configured).
_STORE: ArtifactStore | None = None


def artifact_store() -> ArtifactStore:
    global _STORE
    if _STORE is None:
        put = get = None
        try:
            from storage.blobs.factory import get_blobs
            blobs = get_blobs()

            async def _put(ref, data):
                return await blobs.put(ref, data)

            async def _get(ref):
                return await blobs.get(ref)
            put, get = _put, _get
        except Exception:  # noqa: BLE001 — no blob config → in-process bytes
            put = get = None
        _STORE = ArtifactStore(blob_put=put, blob_get=get)
    return _STORE


__all__ = ["ArtifactStore", "Artifact", "ArtifactVersion", "artifact_store"]
