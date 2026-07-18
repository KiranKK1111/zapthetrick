"""First-run environment setup ("doctor") endpoints.

Drives the animated onboarding screen. Each step is a re-checkable *gate* the
UI can poll or "Proceed" through:

    Docker present?  ->  bring up docker-compose (pgvector image + cache)
      ->  PostgreSQL >= 17 reachable?  ->  `vector` extension  ->  migrations  ->  ready

The pgvector path is deliberately Docker-first: the official `pgvector/pgvector`
image ships the extension prebuilt, so "install pgvector" collapses to a single
`CREATE EXTENSION vector` (which the migrations also do) — no native build.

All endpoints are read-mostly and safe to call repeatedly (idempotent).
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import sys

from fastapi import APIRouter

from app.core.config_loader import get_config, update_config

router = APIRouter(tags=["setup"])

_MIN_PG_MAJOR = 17
# Fixed compose project name so the stack is stable regardless of where the
# compose file lives (otherwise the project name = file's dir → a frozen build
# running from a temp _MEIPASS dir would spin up a NEW stack every launch).
_PROJECT = "zapthetrick"

# The connection the docker-compose `postgres` service exposes (host port 5433,
# to avoid clashing with a local Postgres on 5432). `compose-up` points the
# app's config here so the rest of the flow talks to the bundled database.
_DOCKER_PG = {
    # 127.0.0.1 (not "localhost") to avoid the IPv6 ::1 trap on Windows that
    # makes the startup TCP probe fail even though asyncpg connects fine.
    "host": "127.0.0.1",
    "port": 5433,
    "user": os.environ.get("POSTGRES_USER", "postgres"),
    "password": os.environ.get("POSTGRES_PASSWORD", "zapthetrick"),
    # Use the default `postgres` database; the app's data lives in the
    # `zapthetrick` SCHEMA inside it (created by ensure_schema_exists/migrations).
    "db": os.environ.get("POSTGRES_DB", "postgres"),
    "schema_name": "zapthetrick",
    # The pgvector image doesn't ship Apache AGE — don't try to enable it.
    "enable_age": False,
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def point_app_at_docker_db() -> None:
    """Repoint BOTH config layers at the bundled DB.

    config.yaml alone isn't enough: the storage layer projects the *active
    workspace* (~/.zapthetrick/workspaces.json) over config.yaml, so if we only
    update config.yaml the DB connection still uses the old workspace
    (localhost:5432). Update both so migrations + the app talk to 5433.
    """
    try:
        update_config({"database": {"postgres": _DOCKER_PG}})
    except Exception:  # noqa: BLE001
        pass
    try:
        from app.workspace.repo import Workspace, default_workspace_repo

        rel = {
            "driver": "postgres",
            "host": _DOCKER_PG["host"],
            "port": _DOCKER_PG["port"],
            "db": _DOCKER_PG["db"],
            "schema_name": _DOCKER_PG["schema_name"],
            "user": _DOCKER_PG["user"],
            "password": _DOCKER_PG["password"],
        }
        repo = default_workspace_repo()
        ws = repo.active()
        if ws is None:
            ws = Workspace(name="default", relational=rel)
        else:
            ws.relational = {**(ws.relational or {}), **rel}
        repo.upsert(ws)
        repo.set_active(ws.name)
    except Exception:  # noqa: BLE001 — workspace module optional
        pass


def _step(step_id: str, title: str, status: str, detail: str = "", hint: str = "") -> dict:
    """One gate's state. status ∈ {ok, pending, fail}."""
    return {"id": step_id, "title": title, "status": status, "detail": detail, "hint": hint}


def _compose_file() -> str | None:
    """Locate docker-compose.yml across source, installed, and frozen layouts."""
    here = os.path.dirname(os.path.abspath(__file__))            # app/api
    backend_root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    # Installed layout: exe at {app}\backend\ZapTheTrickBackend.exe → look in
    # {app}\backend and {app}. (sys.executable is the real exe when frozen.)
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    candidates = [
        os.path.join(os.path.dirname(backend_root), "docker-compose.yml"),  # repo root (source)
        os.path.join(backend_root, "docker-compose.yml"),
        os.path.join(exe_dir, "docker-compose.yml"),                        # {app}\backend
        os.path.join(os.path.dirname(exe_dir), "docker-compose.yml"),       # {app}
    ]
    meipass = getattr(sys, "_MEIPASS", None)                                # bundled fallback
    if meipass:
        candidates.append(os.path.join(meipass, "docker-compose.yml"))
    env = os.environ.get("ZAPTHETRICK_COMPOSE_FILE")
    if env:
        candidates.insert(0, env)
    return next((c for c in candidates if os.path.exists(c)), None)


# Windows: run subprocesses with no console window (a frozen GUI app has no
# console, so a spawned child flashing one is both ugly and a source of hangs).
_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


async def _sh(*args: str, timeout: float = 60.0) -> tuple[int, str, str]:
    """Run a command, return (code, stdout, stderr). Never raises.

    Uses blocking `subprocess.run` on a worker thread instead of
    `asyncio.create_subprocess_exec`. The asyncio subprocess transport relies on
    the Proactor event loop + inheritable stdio handles, which routinely hang or
    raise NotImplementedError inside a PyInstaller-frozen Windows app — the dev
    server worked but the bundled exe stalled on every `docker` call. The
    thread + plain subprocess path is reliable in both.
    """

    def _run() -> tuple[int, str, str]:
        try:
            proc = subprocess.run(  # noqa: S603 — fixed argv, no shell
                list(args),
                capture_output=True,
                text=True,
                timeout=timeout,
                creationflags=_NO_WINDOW,
                # Give the child a real (empty) stdin. A PyInstaller-frozen
                # GUI/console app launched with redirected handles can have an
                # invalid stdin; without this the child inherits a bad handle
                # and `subprocess.run` wedges on Windows *past* its own timeout.
                stdin=subprocess.DEVNULL,
            )
            return (
                proc.returncode,
                (proc.stdout or "").strip(),
                (proc.stderr or "").strip(),
            )
        except FileNotFoundError:
            return 127, "", "executable not found"
        except subprocess.TimeoutExpired:
            return 124, "", "timed out"
        except Exception as exc:  # noqa: BLE001
            return 1, "", str(exc)

    return await asyncio.to_thread(_run)


async def _pg_connect():
    """Raw asyncpg connection using the current config's Postgres params."""
    import asyncpg

    pg = get_config().database.postgres
    return await asyncpg.connect(
        host=pg.host,
        port=pg.port,
        user=pg.user,
        password=pg.password,
        database=pg.db,
        timeout=8.0,
    )


# --------------------------------------------------------------------------- #
# individual checks
# --------------------------------------------------------------------------- #
async def _check_docker() -> dict:
    code, out, _ = await _sh("docker", "--version", timeout=15)
    if code != 0:
        return _step(
            "docker", "Docker", "fail",
            detail="Not found",
            hint="Install Docker Desktop and start it, then Re-check.",
        )
    info_code, _, _ = await _sh("docker", "info", timeout=20)
    if info_code != 0:
        return _step(
            "docker", "Docker", "fail", detail=out,
            hint="Docker is installed but not running — start Docker Desktop, then Re-check.",
        )
    return _step("docker", "Docker", "ok", detail=out)


async def _check_compose() -> dict:
    cf = _compose_file()
    if not cf:
        return _step("compose", "Database services", "fail",
                     detail="docker-compose.yml not found",
                     hint="Reinstall the app, or set ZAPTHETRICK_COMPOSE_FILE.")
    code, out, _ = await _sh("docker", "compose", "-p", _PROJECT, "-f", cf, "ps",
                             "--status", "running", "--format", "{{.Service}}",
                             timeout=30)
    running = out.splitlines() if code == 0 else []
    if "postgres" in running:
        # Idempotently point the app at the bundled DB (covers the case where
        # the services were already running before onboarding).
        try:
            if get_config().database.postgres.port != _DOCKER_PG["port"]:
                point_app_at_docker_db()
        except Exception:  # noqa: BLE001
            pass
        return _step("compose", "Database services", "ok",
                     detail="postgres + cache running")
    # Progress honesty: while auto-provision is mid-flight, say WHAT is
    # happening (loading bundled images / pulling from the registry) instead
    # of a frozen-looking "Not started".
    phase = PROVISION_STATE.get("phase")
    if phase == "loading-images":
        return _step("compose", "Database services", "pending",
                     detail="Loading bundled database images…")
    if phase == "pulling-images":
        return _step("compose", "Database services", "pending",
                     detail="Downloading database images… (one-time, ~500 MB)")
    if phase == "compose":
        return _step("compose", "Database services", "pending",
                     detail="Starting containers…")
    return _step("compose", "Database services", "pending",
                 detail="Not started",
                 hint="Click Start to launch the bundled database.")


async def _check_cache() -> dict:
    """Dragonfly (the Redis-compatible cache) — its own visible line so the
    user sees the WHOLE bundled stack, not just Postgres. A dead cache never
    blocks readiness (the app degrades gracefully without it), so this step
    reports ok/pending, never fail."""
    import socket

    def _probe() -> bool:
        try:
            with socket.create_connection(("127.0.0.1", 6379), timeout=2.0):
                return True
        except Exception:  # noqa: BLE001
            return False

    up = await asyncio.to_thread(_probe)
    if up:
        return _step("cache", "Cache (Dragonfly)", "ok", detail="Running")
    return _step("cache", "Cache (Dragonfly)", "pending",
                 detail="Waiting for the cache container")


# ── Bundled Docker images (offline-first full installer) ──────────────────
_REQUIRED_IMAGES = (
    "pgvector/pgvector:pg17",
    "docker.dragonflydb.io/dragonflydb/dragonfly:latest",
)


def _bundled_images_dir() -> str | None:
    """The installer's docker-images folder (full/offline variant), if present.
    Installed layout: exe at {app}\\backend\\… → tars at {app}\\docker-images."""
    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    here = os.path.dirname(os.path.abspath(__file__))
    backend_root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    for cand in (
        os.path.join(os.path.dirname(exe_dir), "docker-images"),   # {app}
        os.path.join(exe_dir, "docker-images"),                     # {app}\backend
        os.path.join(os.path.dirname(backend_root), "installer", "docker-images"),  # dev
    ):
        if os.path.isdir(cand):
            return cand
    return None


async def _ensure_images_present() -> None:
    """Before compose-up: make the required images available. Prefer the
    installer's bundled tars (`docker load` — seconds, zero network); fall back
    to flagging the registry pull so the UI can say so. Best-effort."""
    missing = []
    for img in _REQUIRED_IMAGES:
        code, _, _ = await _sh("docker", "image", "inspect", img, timeout=15)
        if code != 0:
            missing.append(img)
    if not missing:
        return
    tars_dir = _bundled_images_dir()
    if tars_dir:
        tars = sorted(
            os.path.join(tars_dir, f) for f in os.listdir(tars_dir)
            if f.lower().endswith(".tar")
        )
        if tars:
            PROVISION_STATE["phase"] = "loading-images"
            for t in tars:
                await _sh("docker", "load", "-i", t, timeout=600)
            # Re-check; anything STILL missing falls through to the pull.
            missing = []
            for img in _REQUIRED_IMAGES:
                code, _, _ = await _sh("docker", "image", "inspect", img,
                                       timeout=15)
                if code != 0:
                    missing.append(img)
    if missing:
        # `docker compose up` will pull these — surface it as a visible phase
        # so the first-run screen shows "Downloading database images…".
        PROVISION_STATE["phase"] = "pulling-images"


async def _check_db() -> dict:
    try:
        conn = await _pg_connect()
    except Exception as exc:  # noqa: BLE001
        return _step("db", "PostgreSQL 17+", "pending",
                     detail="Waiting for the database…",
                     hint=str(exc).splitlines()[0] if str(exc) else "")
    try:
        ver = await conn.fetchval("SHOW server_version")
    finally:
        await conn.close()
    major = int(re.match(r"(\d+)", ver or "0").group(1))
    if major < _MIN_PG_MAJOR:
        return _step("db", "PostgreSQL 17+", "fail", detail=f"Found {ver}",
                     hint=f"PostgreSQL {_MIN_PG_MAJOR}+ is required.")
    return _step("db", "PostgreSQL 17+", "ok", detail=f"v{ver}")


async def _check_pgvector() -> dict:
    try:
        conn = await _pg_connect()
    except Exception:  # noqa: BLE001
        return _step("pgvector", "pgvector extension", "pending",
                     detail="Waiting for the database…")
    try:
        present = await conn.fetchval(
            "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
        )
    finally:
        await conn.close()
    if present:
        return _step("pgvector", "pgvector extension", "ok", detail="Installed")
    return _step("pgvector", "pgvector extension", "pending",
                 detail="Not installed",
                 hint="Click Proceed to create it (CREATE EXTENSION vector).")


async def _check_migrations() -> dict:
    pg = get_config().database.postgres
    schema = pg.schema_name or "public"
    try:
        conn = await _pg_connect()
    except Exception:  # noqa: BLE001
        return _step("migrations", "Database schema", "pending",
                     detail="Waiting for the database…")
    try:
        exists = await conn.fetchval(
            "SELECT 1 FROM information_schema.tables "
            "WHERE table_schema = $1 AND table_name = 'sessions'",
            schema,
        )
    finally:
        await conn.close()
    if exists:
        return _step("migrations", "Database schema", "ok", detail="Up to date")
    return _step("migrations", "Database schema", "pending",
                 detail="Not applied",
                 hint="Click Proceed to run database migrations.")


# --------------------------------------------------------------------------- #
# actions
# --------------------------------------------------------------------------- #
async def _compose_up() -> dict:
    cf = _compose_file()
    if not cf:
        return _step("compose", "Database services", "fail",
                     detail="docker-compose.yml not found")
    # Offline-first: load the installer's bundled image tars when the images
    # aren't in Docker yet (or flag the registry pull for the progress UI).
    try:
        await _ensure_images_present()
    except Exception:  # noqa: BLE001 — compose-up pulls as the fallback anyway
        pass
    if PROVISION_STATE.get("phase") not in ("pulling-images",):
        PROVISION_STATE["phase"] = "compose"
    code, out, err = await _sh("docker", "compose", "-p", _PROJECT, "-f", cf,
                               "up", "-d", timeout=420)
    if code != 0:
        return _step("compose", "Database services", "fail",
                     detail=(err or out)[:500],
                     hint="`docker compose up` failed — make sure Docker is running.")
    # Point the app (config.yaml + active workspace) at the bundled database so
    # the next gates and migrations talk to it.
    point_app_at_docker_db()
    # Wait for Postgres to accept connections. On a COLD first run the pgvector
    # image was just pulled and the container runs initdb (first-time DB init),
    # which can take well over 30s — the old 30s wait expired before Postgres
    # was ready, so the first auto-provision failed and only a manual Retry
    # (by which point initdb had finished) worked. ~135s covers a cold init.
    for _ in range(90):
        try:
            conn = await _pg_connect()
            await conn.close()
            break
        except Exception:  # noqa: BLE001
            await asyncio.sleep(1.5)
    return await _check_compose()


async def _ensure_pgvector() -> dict:
    try:
        conn = await _pg_connect()
    except Exception as exc:  # noqa: BLE001
        return _step("pgvector", "pgvector extension", "fail",
                     detail="Database not reachable", hint=str(exc))
    try:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except Exception as exc:  # noqa: BLE001
        return _step("pgvector", "pgvector extension", "fail", detail=str(exc),
                     hint="Could not create the extension — ensure the "
                          "pgvector/pgvector image is used for Postgres.")
    finally:
        await conn.close()
    return await _check_pgvector()


async def _run_migrations() -> dict:
    from app.api.routes_settings import _apply_db_changes
    from storage import bootstrap as _bs

    await _apply_db_changes()                      # reinit engine + migrate (background)
    for _ in range(120):                           # poll up to ~60s
        if _bs.MIGRATION_STATE == "ready":
            break
        if _bs.MIGRATION_STATE == "error":
            return _step("migrations", "Database schema", "fail",
                         detail=_bs.MIGRATION_ERROR or "migration failed")
        await asyncio.sleep(0.5)
    return await _check_migrations()


# --------------------------------------------------------------------------- #
# Zero-touch auto-provision (runs automatically on startup)
# --------------------------------------------------------------------------- #
# Surfaced by /checks so the UI can show progress and know when to stop polling.
PROVISION_STATE: dict = {"running": False, "done": False, "error": None, "phase": "idle"}


def _docker_desktop_exe() -> str | None:
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramW6432"),
                 r"C:\Program Files"):
        if base:
            p = os.path.join(base, "Docker", "Docker", "Docker Desktop.exe")
            if os.path.exists(p):
                return p
    return None


async def _ensure_docker_running(max_wait_s: float = 120.0) -> bool:
    """True if the Docker daemon is reachable. If Docker Desktop is installed but
    not running, launch it and wait (up to max_wait_s) for the daemon."""
    code, _, _ = await _sh("docker", "info", timeout=10)
    if code == 0:
        return True
    exe = _docker_desktop_exe()
    if not exe:
        return False
    try:
        subprocess.Popen([exe], creationflags=_NO_WINDOW, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:  # noqa: BLE001
        return False
    waited = 0.0
    while waited < max_wait_s:
        await asyncio.sleep(3.0)
        waited += 3.0
        code, _, _ = await _sh("docker", "info", timeout=8)
        if code == 0:
            return True
    return False


async def auto_provision(migration_task=None) -> None:
    """Zero-touch environment setup, run automatically on backend startup.

    If Docker is available: bring up the bundled stack, repoint config at it,
    create pgvector, run migrations — every step idempotent / a no-op when
    already satisfied. Failures are captured in PROVISION_STATE (surfaced by
    /checks), never raised — so a missing or stopped Docker just leaves the UI
    on the setup screen with a clear message instead of crashing.
    """
    import logging

    log = logging.getLogger(__name__)
    if PROVISION_STATE["running"]:
        return
    PROVISION_STATE.update(running=True, error=None, phase="check")
    try:
        # FAST PATH — if the CONFIGURED database is already reachable + migrated,
        # there is nothing to provision: skip the whole Docker dance. This avoids
        # the `docker` subprocess calls (which can wedge on a frozen Windows app),
        # re-pointing config.yaml on every boot, and a duplicate alembic run — the
        # cause of the "stuck on startup" wedge on a normal restart.
        import storage.bootstrap as _bs

        if migration_task is not None:
            try:
                await migration_task  # the startup migration against the cfg DB
            except Exception:  # noqa: BLE001
                pass
        if _bs.POSTGRES_READY and _bs.MIGRATION_STATE == "ready":
            PROVISION_STATE.update(done=True, phase="done", error=None)
            return

        # Configured DB is NOT up → bring up the bundled Docker stack.
        PROVISION_STATE["phase"] = "docker"
        code, _, _ = await _sh("docker", "--version", timeout=15)
        if code != 0:
            PROVISION_STATE["error"] = "Docker is not installed"
            return
        if not await _ensure_docker_running():
            PROVISION_STATE["error"] = "Docker is installed but not running"
            return
        PROVISION_STATE["phase"] = "compose"
        if (await _check_compose())["status"] != "ok":
            up = await _compose_up()
            if up["status"] != "ok":
                PROVISION_STATE["error"] = up.get("detail") or "could not start the database"
                return
        else:
            point_app_at_docker_db()
        # pgvector + migrations, with a few automatic retries. On a cold first
        # boot the Postgres container can accept connections a beat before it's
        # ready for DDL, so the first pgvector/migration attempt occasionally
        # fails — which is exactly why a manual Retry "just worked". Retrying
        # here makes it succeed on FIRST entry with no user action.
        db_ready = False
        last_err = "could not run database migrations"
        for attempt in range(4):
            PROVISION_STATE["phase"] = "pgvector"
            if (await _ensure_pgvector())["status"] != "ok":
                last_err = "could not enable pgvector"
                await asyncio.sleep(4.0)
                continue
            PROVISION_STATE["phase"] = "migrations"
            if (await _run_migrations())["status"] != "ok":
                last_err = "could not run database migrations"
                await asyncio.sleep(4.0)
                continue
            db_ready = True
            break
        if not db_ready:
            PROVISION_STATE["error"] = last_err
            return
        PROVISION_STATE.update(done=True, phase="done")
    except Exception as exc:  # noqa: BLE001
        PROVISION_STATE["error"] = str(exc)
        log.exception("auto_provision failed")
    finally:
        PROVISION_STATE["running"] = False


# --------------------------------------------------------------------------- #
# routes
# --------------------------------------------------------------------------- #
async def _bounded(coro, fallback: dict, *, timeout: float = 25.0) -> dict:
    """Await a check, but never let it hang the endpoint. If a worker thread
    (e.g. a wedged `docker` subprocess in the frozen exe) doesn't return in
    `timeout`s, abandon it and report the fallback step so the UI keeps moving."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except Exception:  # noqa: BLE001 — TimeoutError or anything the check raised
        return fallback


@router.get("/checks")
async def checks() -> dict:
    """Snapshot every gate (cheap, safe to poll). Later gates stay 'pending'
    until their prerequisite is green so the UI advances in order.

    Every gate is hard-bounded: a single hung check can never stall the whole
    response (which would strand the onboarding screen on "Checking environment").
    """
    docker = await _bounded(
        _check_docker(),
        _step("docker", "Docker", "fail", detail="Check timed out",
              hint="Is Docker Desktop running? Start it, then Re-check."),
    )
    compose = (await _bounded(
        _check_compose(),
        _step("compose", "Database services", "pending", "Check timed out — Re-check"),
    ) if docker["status"] == "ok"
               else _step("compose", "Database services", "pending", "Waiting for Docker"))
    cache = (await _bounded(
        _check_cache(),
        _step("cache", "Cache (Dragonfly)", "pending", "Check timed out — Re-check"),
    ) if compose["status"] == "ok"
             else _step("cache", "Cache (Dragonfly)", "pending",
                        "Waiting for services"))
    db = (await _bounded(
        _check_db(),
        _step("db", "PostgreSQL 17+", "pending", "Check timed out — Re-check"),
    ) if compose["status"] == "ok"
          else _step("db", "PostgreSQL 17+", "pending", "Waiting for the database"))
    pgv = (await _bounded(
        _check_pgvector(),
        _step("pgvector", "pgvector extension", "pending", "Check timed out — Re-check"),
    ) if db["status"] == "ok"
           else _step("pgvector", "pgvector extension", "pending", "Waiting for the database"))
    mig = (await _bounded(
        _check_migrations(),
        _step("migrations", "Database schema", "pending", "Check timed out — Re-check"),
    ) if pgv["status"] == "ok"
           else _step("migrations", "Database schema", "pending", "Waiting for pgvector"))
    steps = [docker, compose, cache, db, pgv, mig]
    return {
        "steps": steps,
        # The cache row is informational — the app degrades gracefully without
        # it, so a slow/crashing Dragonfly must never trap the user at setup.
        "ready": all(s["status"] == "ok" for s in steps if s["id"] != "cache"),
        "provisioning": PROVISION_STATE["running"],
        "provision_phase": PROVISION_STATE["phase"],
        "provision_error": PROVISION_STATE["error"],
    }


@router.post("/auto")
async def auto() -> dict:
    """(Re)run zero-touch provisioning, then return the latest gate snapshot.
    Used by the UI's 'Retry' affordance; startup runs it automatically."""
    await auto_provision()
    return await checks()


@router.post("/compose-up")
async def compose_up() -> dict:
    """Bring up the bundled docker-compose services and point config at them."""
    return await _compose_up()


@router.post("/pgvector")
async def pgvector() -> dict:
    """CREATE EXTENSION vector (idempotent), then verify."""
    return await _ensure_pgvector()


@router.post("/migrate")
async def migrate() -> dict:
    """Run database migrations, then verify the schema."""
    return await _run_migrations()
