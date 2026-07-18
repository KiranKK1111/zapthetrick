"""Entry point for the FROZEN backend (the bundled ``ZapTheTrickBackend.exe``).

PyInstaller packages this script; the Flutter app spawns the resulting exe as a
child process on launch. Responsibilities:

  1. Resolve a *writable* config path. When installed under Program Files the
     working dir isn't writable, so config lives under ``%APPDATA%\\ZapTheTrick``,
     seeded from the bundled ``config.example.yaml`` on first run.
  2. Watch the parent (the Flutter app) and self-terminate if it dies — so the
     backend can never linger as an orphaned background process, even if the UI
     crashes. (The Flutter side also kills us on a clean exit; this is the
     belt-and-suspenders.)
  3. Start uvicorn serving ``app.main:app``.

Run standalone for testing:  ``python server_main.py``
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time


def _bundled_path(name: str) -> str:
    """Path to a file bundled by PyInstaller (extracted to ``sys._MEIPASS`` at
    runtime), or next to this script when running from source."""
    base = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, name)


def _set_writable_cwd() -> None:
    """Frozen builds install under Program Files, which is READ-ONLY without
    admin. The app writes several relative paths (``./data/blobs``,
    ``./data/vectors``, ``./data/backups``) that would resolve there and fail
    with ``[WinError 5] Access is denied`` — stalling setup. Redirect the
    process CWD to a per-user writable data root so every ``./data/*`` lands
    under %LOCALAPPDATA%. Bundled reads use absolute _MEIPASS paths and alembic
    pins an absolute script_location, so nothing bundled is affected.
    """
    if not getattr(sys, "frozen", False):
        return
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") \
        or os.path.expanduser("~")
    # chdir to the app root so relative "./data/*" resolves to
    # %LOCALAPPDATA%\ZapTheTrickAI\data\* (next to the runtime pack).
    app_root = os.path.join(base, "ZapTheTrickAI")
    try:
        os.makedirs(os.path.join(app_root, "data"), exist_ok=True)
        os.chdir(app_root)
    except Exception:  # noqa: BLE001 — never block boot on this
        pass


def _resolve_config_path() -> str:
    """A writable config.yaml path, seeded from the bundled example on first run."""
    env = os.environ.get("ZAPTHETRICK_CONFIG_PATH")
    if env:
        return env
    # When running from source, prefer the repo's config.yaml if present.
    local = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    if not getattr(sys, "frozen", False) and os.path.exists(local):
        return local
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    cfg_dir = os.path.join(base, "ZapTheTrick")
    os.makedirs(cfg_dir, exist_ok=True)
    cfg = os.path.join(cfg_dir, "config.yaml")
    if not os.path.exists(cfg):
        example = _bundled_path("config.example.yaml")
        try:
            shutil.copyfile(example, cfg)
        except Exception:
            # No example bundled — the app's first-run wizard writes config anyway.
            pass
    return cfg


def _parent_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k = ctypes.windll.kernel32
        h = k.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _watch_parent() -> None:
    """If the Flutter app (parent) dies, exit so we don't orphan."""
    raw = os.environ.get("ZAPTHETRICK_PARENT_PID")
    if not raw or not raw.isdigit():
        return
    pid = int(raw)

    def loop() -> None:
        while True:
            time.sleep(2.0)
            if not _parent_alive(pid):
                os._exit(0)

    threading.Thread(target=loop, name="parent-watcher", daemon=True).start()


def _redirect_logs_to_file() -> None:
    """With the windowless build (console=False) the frozen exe has no console,
    so send stdout/stderr to a log file next to the config — the app stays a
    single, window-less child process while remaining debuggable."""
    if not getattr(sys, "frozen", False):
        return
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    log_dir = os.path.join(base, "ZapTheTrick")
    try:
        os.makedirs(log_dir, exist_ok=True)
        f = open(os.path.join(log_dir, "backend.log"), "a", buffering=1, encoding="utf-8")
        sys.stdout = f
        sys.stderr = f
    except Exception:
        pass


def main() -> None:
    _redirect_logs_to_file()  # before uvicorn configures its stderr handler
    _set_writable_cwd()       # Program Files is read-only → writable data root
    os.environ["ZAPTHETRICK_CONFIG_PATH"] = _resolve_config_path()
    _watch_parent()

    # LITE build support: the AI runtime pack (torch/onnxruntime, downloaded
    # once to %LOCALAPPDATA%) must be on sys.path BEFORE anything imports the
    # app. A full build (torch bundled) makes this a harmless no-op.
    try:
        from app.runtime_pack import activate as _rp_activate
        _rp_activate()
    except Exception:  # noqa: BLE001 — never block boot on the pack
        pass

    host = os.environ.get("ZAPTHETRICK_HOST", "127.0.0.1")
    port = int(os.environ.get("ZAPTHETRICK_PORT", "8000"))

    import uvicorn

    # Import the app AFTER the config path is set so the loader picks it up.
    from app.main import app

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
