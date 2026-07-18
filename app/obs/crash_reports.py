"""Crash-report intake (roadmap Phase 1 #24 — release channel).

A bounded in-process ring of client crash reports (message, stack, version,
platform) so a bad rollout surfaces fast and can inform a rollback. Lock-guarded,
fail-open. Persisting to durable storage is an ops choice layered on top.
"""
from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass


@dataclass
class CrashReport:
    message: str
    stack: str = ""
    version: str = ""
    platform: str = ""
    ts: float = 0.0


class CrashLog:
    def __init__(self, max_reports: int = 200) -> None:
        self._reports: list[CrashReport] = []
        self._max = max_reports
        self._lock = threading.Lock()

    def record(self, *, message: str, stack: str = "", version: str = "",
               platform: str = "") -> None:
        if not (message or "").strip():
            return
        with self._lock:
            self._reports.append(CrashReport(
                message=message[:500], stack=stack[:4000],
                version=version[:40], platform=platform[:40], ts=time.time()))
            if len(self._reports) > self._max:
                self._reports = self._reports[-self._max:]

    def recent(self, n: int = 50) -> list[dict]:
        with self._lock:
            return [asdict(r) for r in self._reports[-n:][::-1]]

    def summary(self) -> dict:
        with self._lock:
            by_version = Counter(r.version or "?" for r in self._reports)
            return {
                "total": len(self._reports),
                "by_version": dict(by_version),
            }

    def clear(self) -> None:
        with self._lock:
            self._reports.clear()


_log = CrashLog()


def crash_log() -> CrashLog:
    return _log


__all__ = ["CrashReport", "CrashLog", "crash_log"]
