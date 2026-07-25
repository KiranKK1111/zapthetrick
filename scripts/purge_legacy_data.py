"""Purge pre-account (global/legacy) data from Postgres (vNext §10.1c).

Removes everything that isn't owned by a real account, so a multi-user deploy
starts clean:
  • NULL-owned rows (the old shared/global data) in every user-owned table, and
  • "device"/anonymous users (users with NO email) + all their data (cascade).

KEEPS: real accounts (users with an email) + their data, and the shared on-pod
`local` model floor.

DRY-RUN by default — prints what it WOULD delete. Pass --yes to actually delete.

    ./.venv/Scripts/python.exe scripts/purge_legacy_data.py          # preview
    ./.venv/Scripts/python.exe scripts/purge_legacy_data.py --yes    # delete
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys

# Allow running as a plain script (add the backend root to the import path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# User-owned tables keyed by a nullable `user_id`. Children (messages,
# agent_steps, resume_chunks, generated_documents, llm_fallback_config …)
# cascade via their FK ON DELETE, so we only target these parents.
_NULL_TABLES = [
    "llm_api_keys", "sessions", "resumes", "projects",
    "model_usage", "episodes", "solve_sessions", "skills",
]


def _make_engine():
    from storage.db import _build_url, _search_path
    return create_async_engine(
        _build_url(),
        connect_args={"server_settings": {"search_path": _search_path()}})


async def _count(conn, sql: str) -> int:
    return int((await conn.execute(text(sql))).scalar() or 0)


async def run(execute: bool) -> None:
    # (label, WHERE clause, table) — count + delete share the WHERE.
    plan: list[tuple[str, str, str]] = [
        (f"{t} (unowned)", "user_id IS NULL", t) for t in _NULL_TABLES
    ]
    plan.append((
        "llm_models (unowned cloud; local kept)",
        "user_id IS NULL AND platform <> 'local'", "llm_models"))
    plan.append((
        "users without an email (device/anon) + their data",
        "email IS NULL", "users"))

    eng = _make_engine()
    try:
        async with eng.begin() as conn:
            print("\nLegacy / global data to remove:")
            print("-" * 56)
            total = 0
            for label, where, table in plan:
                cnt = await _count(
                    conn, f"SELECT count(*) FROM {table} WHERE {where}")
                total += cnt
                print(f"  {label:<48} {cnt:>6}")
            print("-" * 56)
            print(f"  {'TOTAL rows directly targeted':<48} {total:>6}")
            print("  (children cascade: messages, agent_steps, resume_chunks,")
            print("   generated_documents, llm_fallback_config, ...)\n")

            keep = await _count(
                conn, "SELECT count(*) FROM users WHERE email IS NOT NULL")
            local = await _count(
                conn, "SELECT count(*) FROM llm_models WHERE platform = 'local'")
            print(f"  Real accounts kept (users with an email): {keep}")
            print(f"  Shared local model floor kept:            {local}\n")

            if not execute:
                print("DRY RUN — nothing deleted. Re-run with --yes to delete.\n")
                return

            for _label, where, table in plan:
                await conn.execute(text(f"DELETE FROM {table} WHERE {where}"))
        # Committed by the `eng.begin()` context on clean exit.
        print("Done: legacy/global data removed.\n")
    finally:
        await eng.dispose()


def main() -> None:
    ap = argparse.ArgumentParser(description="Purge pre-account/global data.")
    ap.add_argument("--yes", action="store_true",
                    help="actually delete (default is a dry-run preview)")
    args = ap.parse_args()
    try:
        asyncio.run(run(args.yes))
    except Exception as exc:  # noqa: BLE001
        print(f"purge failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
