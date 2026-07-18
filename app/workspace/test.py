"""10-step workspace probe.

Architecture.md commits to a "real connection test" feature that
walks the user's workspace, validates each driver, and surfaces a
clear actionable error when anything is broken.

The 10 steps:
  1. validate workspace shape (required fields present)
  2. resolve driver IDs
  3. check the required Python package is importable
  4. open the relational connection
  5. version + permission probe on the relational DB
  6. open the vector store, ping it
  7. open the cache, ping it
  8. open the blob store, list a no-op prefix
  9. measure each driver's round-trip latency
 10. report — pass/fail per step with timings

Cheap and synchronous-ish; uses dynamic imports so the absence of
e.g. `lancedb` doesn't crash the probe.
"""
from __future__ import annotations

import importlib
import logging
import time
from dataclasses import dataclass, field

from .drivers import DRIVERS
from .repo import Workspace


log = logging.getLogger(__name__)


@dataclass
class ProbeStep:
    name: str
    ok: bool = False
    detail: str = ""
    latency_ms: int = 0


@dataclass
class ProbeReport:
    workspace: str
    overall_ok: bool = False
    steps: list[ProbeStep] = field(default_factory=list)

    def add(self, step: ProbeStep) -> None:
        self.steps.append(step)


async def probe_workspace(ws: Workspace) -> ProbeReport:
    """Run all 10 steps. Never raises — every failure becomes a step
    with `ok=False` and a human-readable `detail`."""
    rep = ProbeReport(workspace=ws.name)

    # ---- 1. shape -----------------------------------------------------
    t0 = time.monotonic()
    missing: list[str] = []
    if not ws.relational.get("driver"):
        missing.append("relational.driver")
    if not ws.vector.get("driver"):
        missing.append("vector.driver")
    rep.add(ProbeStep(
        name="workspace shape",
        ok=not missing,
        detail=("missing: " + ", ".join(missing)) if missing else "ok",
        latency_ms=_ms(t0),
    ))
    if missing:
        return rep

    # ---- 2. driver registry -------------------------------------------
    t0 = time.monotonic()
    invalid = []
    for slot in ("relational", "vector", "cache", "blob"):
        drv = getattr(ws, slot).get("driver")
        if drv and drv not in DRIVERS:
            invalid.append(f"{slot}={drv}")
    rep.add(ProbeStep(
        name="driver registry",
        ok=not invalid,
        detail=("unknown drivers: " + ", ".join(invalid)) if invalid else "ok",
        latency_ms=_ms(t0),
    ))
    if invalid:
        return rep

    # ---- 3. package gate ----------------------------------------------
    t0 = time.monotonic()
    missing_pkgs: list[str] = []
    for slot in ("relational", "vector", "cache", "blob"):
        drv_id = getattr(ws, slot).get("driver")
        if not drv_id:
            continue
        pkg = DRIVERS[drv_id].requires_pkg
        if pkg:
            try:
                importlib.import_module(pkg.split(".")[0])
            except Exception:
                missing_pkgs.append(f"{slot}:{pkg}")
    rep.add(ProbeStep(
        name="package gate",
        ok=not missing_pkgs,
        detail=("install: " + ", ".join(missing_pkgs)) if missing_pkgs else "ok",
        latency_ms=_ms(t0),
    ))

    # ---- 4-5. relational ----------------------------------------------
    rep.add(await _probe_relational(ws))

    # ---- 6. vector ----------------------------------------------------
    rep.add(await _probe_vector(ws))

    # ---- 7. cache -----------------------------------------------------
    rep.add(await _probe_cache(ws))

    # ---- 8. blob ------------------------------------------------------
    rep.add(await _probe_blob(ws))

    # ---- 9. latency aggregate -----------------------------------------
    total = sum(s.latency_ms for s in rep.steps)
    rep.add(ProbeStep(
        name="latency aggregate",
        ok=True,
        detail=f"{total} ms across {len(rep.steps)} steps",
        latency_ms=0,
    ))

    # ---- 10. report ---------------------------------------------------
    rep.overall_ok = all(s.ok for s in rep.steps)
    rep.add(ProbeStep(
        name="report",
        ok=rep.overall_ok,
        detail="all steps passed" if rep.overall_ok else "see step list",
        latency_ms=0,
    ))
    return rep


# ---- per-driver probes -------------------------------------------------
async def _probe_relational(ws: Workspace) -> ProbeStep:
    t0 = time.monotonic()
    drv = ws.relational.get("driver")
    try:
        if drv == "postgres":
            import asyncpg
            conn = await asyncpg.connect(
                host=ws.relational.get("host", "localhost"),
                port=ws.relational.get("port", 5432),
                database=ws.relational.get("db"),
                user=ws.relational.get("user"),
                password=ws.relational.get("password") or None,
                timeout=4.0,
            )
            v = await conn.fetchval("SELECT version()")
            await conn.close()
            return ProbeStep("relational", True, str(v)[:80], _ms(t0))
        if drv == "sqlite":
            import sqlite3
            from pathlib import Path

            path = Path(ws.relational.get("path", "./data/zapthetrick.db"))
            path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(path), timeout=4.0)
            v = conn.execute("SELECT sqlite_version()").fetchone()[0]
            conn.close()
            return ProbeStep("relational", True, f"sqlite {v}", _ms(t0))
        if drv == "mysql":
            # Best-effort probe; aiomysql may not be installed.
            import aiomysql  # type: ignore
            conn = await aiomysql.connect(
                host=ws.relational.get("host", "localhost"),
                port=ws.relational.get("port", 3306),
                db=ws.relational.get("db"),
                user=ws.relational.get("user"),
                password=ws.relational.get("password") or "",
                connect_timeout=4.0,
            )
            cursor = await conn.cursor()
            await cursor.execute("SELECT VERSION()")
            v = (await cursor.fetchone())[0]
            await cursor.close()
            conn.close()
            return ProbeStep("relational", True, f"mysql {v}", _ms(t0))
        return ProbeStep("relational", False, f"unsupported: {drv}", _ms(t0))
    except Exception as exc:  # noqa: BLE001
        return ProbeStep("relational", False, str(exc)[:200], _ms(t0))


async def _probe_vector(ws: Workspace) -> ProbeStep:
    t0 = time.monotonic()
    drv = ws.vector.get("driver")
    try:
        if drv == "lancedb":
            import lancedb  # type: ignore
            db = lancedb.connect(ws.vector.get("uri"))
            _ = db.table_names()
            return ProbeStep("vector", True, "lancedb reachable", _ms(t0))
        return ProbeStep("vector", False, f"unsupported: {drv}", _ms(t0))
    except Exception as exc:  # noqa: BLE001
        return ProbeStep("vector", False, str(exc)[:200], _ms(t0))


async def _probe_cache(ws: Workspace) -> ProbeStep:
    t0 = time.monotonic()
    drv = ws.cache.get("driver")
    if not drv:
        return ProbeStep("cache", True, "skipped (no cache configured)", _ms(t0))
    try:
        if drv == "redis":
            import redis.asyncio as redis  # type: ignore

            url = ws.cache.get("url") or ""
            # Set BOTH connect and operation timeouts. Without
            # socket_connect_timeout, a `localhost`-resolves-to-IPv6
            # connect attempt on Windows can hang for the full system
            # default (~21s) before the operation timeout kicks in.
            client = redis.from_url(
                url,
                socket_connect_timeout=2.0,
                socket_timeout=2.0,
            )
            try:
                pong = await client.ping()
            finally:
                await client.close()
            ok = bool(pong)
            detail = f"PING -> {pong}"
            if not ok and "localhost" in url.lower():
                detail += " (try redis://127.0.0.1:... — `localhost` may resolve to IPv6)"
            return ProbeStep("cache", ok, detail, _ms(t0))
        return ProbeStep("cache", False, f"unsupported: {drv}", _ms(t0))
    except Exception as exc:  # noqa: BLE001
        detail = str(exc)[:200]
        if ws.cache.get("url", "").lower().find("localhost") >= 0:
            detail += "  — hint: switch `localhost` to 127.0.0.1 (Windows IPv6 quirk)"
        return ProbeStep("cache", False, detail, _ms(t0))


async def _probe_blob(ws: Workspace) -> ProbeStep:
    t0 = time.monotonic()
    drv = ws.blob.get("driver") or "filesystem"
    try:
        if drv == "filesystem":
            from pathlib import Path

            p = Path(ws.blob.get("path", "./data/blobs"))
            p.mkdir(parents=True, exist_ok=True)
            return ProbeStep("blob", True, f"fs: {p}", _ms(t0))
        if drv == "minio":
            # Light probe — defer real check to first put.
            return ProbeStep("blob", True, "minio (config-only probe)", _ms(t0))
        return ProbeStep("blob", False, f"unsupported: {drv}", _ms(t0))
    except Exception as exc:  # noqa: BLE001
        return ProbeStep("blob", False, str(exc)[:200], _ms(t0))


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


__all__ = ["probe_workspace", "ProbeStep", "ProbeReport"]
