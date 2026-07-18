"""Version manifest + update-check foundation (roadmap Phase 1 / gap-fill #24).

The **decision logic** for "is an update available / required?" — pure and
offline-testable. `APP_VERSION` is the single source of truth for the running
version (main.py's `/` endpoint reads it here).

Out of scope by design (needs the release pipeline / a server, and honors the
"no auto-rebuild" rule): actually FETCHING the manifest over the network,
downloading a binary, and staged rollout. Those wrap this pure core; this is the
part that can be built and tested now.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass

# Single source of truth for the running app version (kept in sync with the
# Flutter pubspec `version:`). main.py imports this instead of hardcoding.
APP_VERSION = "0.2.0"


def parse_version(v: str) -> tuple[int, int, int] | None:
    """Parse 'MAJOR.MINOR.PATCH' (extra build/pre-release suffixes ignored).
    Returns None if unparseable."""
    if not v or not isinstance(v, str):
        return None
    core = v.strip().lstrip("vV").split("+")[0].split("-")[0]
    parts = core.split(".")
    if not 1 <= len(parts) <= 3:
        return None
    nums = []
    for p in parts:
        if not p.isdigit():
            return None
        nums.append(int(p))
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums[:3])  # type: ignore[return-value]


def compare_versions(a: str, b: str) -> int | None:
    """-1 if a<b, 0 if equal, 1 if a>b; None if either is unparseable."""
    pa, pb = parse_version(a), parse_version(b)
    if pa is None or pb is None:
        return None
    return (pa > pb) - (pa < pb)


class UpdateStatus(enum.Enum):
    UP_TO_DATE = "up_to_date"
    UPDATE_AVAILABLE = "update_available"
    UPDATE_REQUIRED = "update_required"   # below minimum_supported — must update
    UNKNOWN = "unknown"                   # unparseable version(s)


@dataclass(frozen=True)
class ReleaseManifest:
    latest: str
    minimum_supported: str = "0.0.0"
    channel: str = "stable"
    notes: str = ""
    url: str = ""

    @staticmethod
    def from_dict(d: dict) -> "ReleaseManifest":
        return ReleaseManifest(
            latest=str(d.get("latest", "")),
            minimum_supported=str(d.get("minimum_supported", "0.0.0")),
            channel=str(d.get("channel", "stable")),
            notes=str(d.get("notes", "")),
            url=str(d.get("url", "")),
        )


@dataclass(frozen=True)
class UpdateResult:
    status: UpdateStatus
    current: str
    latest: str
    notes: str = ""
    url: str = ""

    @property
    def update_available(self) -> bool:
        return self.status in (UpdateStatus.UPDATE_AVAILABLE,
                               UpdateStatus.UPDATE_REQUIRED)


def check_for_update(current: str, manifest: ReleaseManifest) -> UpdateResult:
    """Compare the running version against a release manifest. Pure — the caller
    supplies the manifest (fetched however it likes; None-safe here)."""
    cur_min = compare_versions(current, manifest.minimum_supported)
    cur_latest = compare_versions(current, manifest.latest)
    if cur_min is None or cur_latest is None:
        status = UpdateStatus.UNKNOWN
    elif cur_min < 0:
        status = UpdateStatus.UPDATE_REQUIRED
    elif cur_latest < 0:
        status = UpdateStatus.UPDATE_AVAILABLE
    else:
        status = UpdateStatus.UP_TO_DATE
    return UpdateResult(status=status, current=current, latest=manifest.latest,
                        notes=manifest.notes, url=manifest.url)


__all__ = [
    "APP_VERSION", "parse_version", "compare_versions",
    "UpdateStatus", "ReleaseManifest", "UpdateResult", "check_for_update",
]
