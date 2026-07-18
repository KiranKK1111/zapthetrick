"""Settings API: read and update the live config.

GET  /api/settings                  -> full current config as JSON
POST /api/settings                  -> deep-merge a partial update; persists to config.yaml
GET  /api/settings/schema           -> describes available knobs + choices for UI
GET  /api/settings/llm              -> probes the configured LLM provider
GET  /api/settings/llm/models       -> lists models the provider offers
GET  /api/settings/database         -> current DB stack status snapshot
POST /api/settings/database/test    -> dry-run a Postgres connection with overrides
POST /api/settings/database/apply   -> reinit engine + run migrations after a save

When the `database.postgres` section of `POST /api/settings` changes,
the route automatically calls `_apply_db_changes` so the next request
hits the new database (no restart required). The schema is created if
missing — the Settings UI's "Schema" field is fire-and-forget.
"""
import logging

from fastapi import APIRouter, HTTPException

from app.core.config_loader import get_config, update_config
from app.core.llm_client import LLMError, llm  # noqa: F401 — surfaced via list_models


log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/settings")


def _redact(cfg_dict: dict) -> dict:
    """Strip the Postgres password before sending the config to the UI.

    The Settings screen renders the password field as masked input
    and only sends the new value when the user explicitly types one.
    """
    pg = (cfg_dict.get("database") or {}).get("postgres") or {}
    if "password" in pg:
        pg["password"] = "********" if pg["password"] else ""
    return cfg_dict


@router.get("")
async def read_settings() -> dict:
    """Return the full current configuration (password redacted)."""
    return _redact(get_config().model_dump())


@router.post("")
async def write_settings(updates: dict) -> dict:
    """Deep-merge `updates` into the config, persist to config.yaml.

    Example body: `{"llm": {"model": "qwen2.5:7b-instruct"}}`
    Only the provided keys change; everything else is preserved.

    After persisting, the route publishes one [ConfigEvent] per
    top-level section that changed so live subscribers (LLM client,
    embedder, vector store, …) can rebuild themselves in-process.

    When `database.postgres` is in `updates`, the route additionally
    reinits the DB engine after the save so subsequent requests use
    the new credentials. The schema is created if missing.
    """
    if not isinstance(updates, dict):
        raise HTTPException(status_code=400, detail="Body must be a JSON object.")

    # Drop any masked-password placeholder so we don't overwrite the
    # real value with literal asterisks.
    pg_update = (updates.get("database") or {}).get("postgres") or {}
    if pg_update.get("password") in ("********", "***"):
        pg_update.pop("password", None)

    db_changed = "database" in updates and "postgres" in (updates.get("database") or {})

    # Snapshot the BEFORE state so we can compute a real diff.
    old_cfg = get_config().model_dump()

    try:
        new_cfg = update_config(updates)
    except Exception as exc:  # noqa: BLE001 — surface validation errors verbatim
        raise HTTPException(status_code=400, detail=f"Invalid update: {exc}")

    # Publish per-section diffs to the live-config bus. Each subscriber
    # (LLM client, embedder, reranker, vector store, …) rebuilds in
    # place; no process restart required.
    new_dict = new_cfg.model_dump()
    try:
        from app.settings.bus import bus, diff_paths

        for section, diff in diff_paths(old_cfg, new_dict).items():
            await bus.publish(section, diff, new_dict)
    except Exception as exc:  # noqa: BLE001 — never let bus errors break a save
        log.warning("config bus publish failed: %s", exc)

    if db_changed:
        await _apply_db_changes()

    return _redact(new_dict)


@router.get("/schema")
async def read_schema() -> dict:
    """Describe the user-facing knobs.

    The Flutter Settings screen reads this to render every form. New
    fields show up the next time the screen loads; no rebuild needed.
    `secret: true` flags a password input.
    """
    return {
        "llm": {
            "provider": {
                "type": "enum",
                "choices": ["auto", "ollama", "openrouter", "nvidia"],
                "implemented": ["auto", "ollama", "openrouter", "nvidia"],
                "notes": (
                    "auto = multi-provider fallback engine. Add provider keys "
                    "under Settings → Providers; routing picks the best "
                    "available model and falls back on rate limits/outages."
                ),
            },
            "model": {"type": "string", "label": "Model"},
            "code_model": {"type": "string", "label": "Code model"},
            "vision_model": {"type": "string", "label": "Vision model"},
            "classifier_model": {"type": "string", "label": "Classifier model"},
            "base_url": {
                "type": "string",
                "label": "Ollama base URL",
                "provider": "ollama",
            },
            "openrouter_api_key": {
                "type": "string",
                "label": "OpenRouter API key",
                "secret": True,
                "provider": "openrouter",
            },
            "openrouter_base_url": {
                "type": "string",
                "label": "OpenRouter base URL",
                "provider": "openrouter",
            },
            "nvidia_api_key": {
                "type": "string",
                "label": "NVIDIA API key",
                "secret": True,
                "provider": "nvidia",
            },
            "nvidia_base_url": {
                "type": "string",
                "label": "NVIDIA base URL",
                "provider": "nvidia",
            },
            "temperature": {"type": "float", "min": 0.0, "max": 2.0},
            "max_tokens": {"type": "int", "min": 1, "max": 8192},
            "timeout_seconds": {"type": "float", "min": 1.0, "max": 600.0},
        },
        "app": {
            "theme_default": {
                "type": "enum",
                "choices": ["dark", "light", "system"],
            },
            "language": {"type": "string"},
        },
        # The Database section drives the new UI form. Same shape as
        # the LLM section so the Settings screen can render it via the
        # same generic field widget.
        "database": {
            "postgres": {
                "host": {"type": "string", "label": "Host"},
                "port": {"type": "int", "label": "Port", "min": 1, "max": 65535},
                "db": {
                    "type": "string",
                    "label": "Database",
                    "notes": "The Postgres database name.",
                },
                "schema_name": {
                    "type": "string",
                    "label": "Schema",
                    "notes": "Created automatically if it doesn't exist. Defaults to `public`.",
                },
                "user": {"type": "string", "label": "User"},
                "password": {
                    "type": "string",
                    "label": "Password",
                    "secret": True,
                },
                "enable_age": {
                    "type": "bool",
                    "label": "Apache AGE graph extension",
                },
            },
        },
    }


@router.get("/llm")
async def llm_health() -> dict:
    """Probe the configured LLM provider. Used by the UI status indicator."""
    return await llm.health()


@router.get("/llm/models")
async def list_llm_models() -> dict:
    """List every model the configured provider exposes."""
    try:
        models = await llm.list_models()
    except Exception:  # noqa: BLE001 -- never blow up the UI over this
        models = []
    return {
        "provider": get_config().llm.provider,
        "models": [{"name": m.get("name", ""), "size": m.get("size", 0)} for m in models],
    }


# ---- Database surface ---------------------------------------------------
@router.get("/database")
async def database_status() -> dict:
    """Quick status snapshot the Settings screen displays.

    Doesn't open a fresh connection — reports what bootstrap and the
    background migration task flagged. Use `/database/test` to
    actively probe a candidate connection.

    `state` mirrors the migration lifecycle so the UI can show a
    progress chip while a Save is mid-flight:
        idle | migrating | ready | error
    """
    from storage import bootstrap as _bs

    # Self-heal a stale boot-time flag: if the DB came up after the backend
    # (or a transient outage recovered), flip POSTGRES_READY back on so the
    # badge doesn't read "degraded" while the DB is actually usable.
    if not _bs.POSTGRES_READY:
        try:
            await _bs.recheck_postgres()
        except Exception:  # noqa: BLE001 — status must never fail
            pass

    cfg = get_config()
    pg = cfg.database.postgres
    return {
        "ready": _bs.POSTGRES_READY,
        "state": _bs.MIGRATION_STATE,
        "error": _bs.MIGRATION_ERROR,
        "postgres": {
            "host": pg.host,
            "port": pg.port,
            "db": pg.db,
            "schema_name": pg.schema_name,
            "user": pg.user,
        },
    }


@router.post("/database/test")
async def database_test(body: dict | None = None) -> dict:
    """Try a one-shot Postgres connection with optional overrides.

    Body (all optional — overrides the current cfg for the probe):
      {host, port, db, schema_name, user, password}

    Returns `{ok, host, port, version?, schema?, schema_exists?, error?}`.
    """
    from storage.db import test_connection

    overrides = body or {}
    if not isinstance(overrides, dict):
        raise HTTPException(400, detail="Body must be a JSON object.")
    # Strip the masked-password placeholder.
    if overrides.get("password") in ("********", "***"):
        overrides.pop("password", None)
    return await test_connection(postgres_overrides=overrides)


@router.post("/database/apply")
async def database_apply() -> dict:
    """Manually re-run the schema-create + migrate + reinit cycle.

    `POST /api/settings` does this automatically when the database
    section changes; this endpoint exists for the UI's "Retry" button
    after a failed startup probe.
    """
    return await _apply_db_changes()


async def _apply_db_changes() -> dict:
    """Kick off engine-reinit + migrations in the background.

    Returns immediately with `{state: "migrating"}` so the Save button
    in the UI doesn't block on a slow migration. The UI polls
    `/api/settings/database` to watch `state` transition through
    `migrating` → `ready` (or `error`).
    """
    import asyncio

    from storage import bootstrap as _bs
    from storage.bootstrap import bootstrap_storage, run_migrations_in_background
    from storage.db import reinit_engine

    # Mark migrating so polls see the transition before the engine swap
    # has even started.
    _bs.POSTGRES_READY = False
    _bs.MIGRATION_STATE = "migrating"
    _bs.MIGRATION_ERROR = None

    async def _run():
        try:
            await reinit_engine()
            await bootstrap_storage()
            await run_migrations_in_background()
        except Exception as exc:  # noqa: BLE001
            _bs.POSTGRES_READY = False
            _bs.MIGRATION_STATE = "error"
            _bs.MIGRATION_ERROR = str(exc)
            log.error("DB reinit failed: %s", exc)

    asyncio.create_task(_run(), name="settings-apply-db")
    return {"state": _bs.MIGRATION_STATE, "ready": False}
