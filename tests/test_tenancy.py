"""Per-user tenancy scoping + RLS SQL (vNext §10.2) — pure mechanism."""
from __future__ import annotations

import uuid

import pytest

from storage import tenancy as T

_UID = "11111111-2222-3333-4444-555555555555"


def test_app_user_guc_emits_set_local():
    assert T.app_user_guc(_UID) == f"SET LOCAL app.user_id = '{_UID}'"
    # Accepts a UUID object too, re-serialized canonically.
    assert T.app_user_guc(uuid.UUID(_UID)) == f"SET LOCAL app.user_id = '{_UID}'"


def test_app_user_guc_is_injection_safe():
    # Anything that isn't a UUID is rejected — arbitrary text can never reach the
    # SQL, so `'; DROP TABLE users; --` can't be smuggled into SET LOCAL.
    for evil in ["'; DROP TABLE users; --", "1 OR 1=1", "", "not-a-uuid",
                 "11111111-2222-3333-4444"]:
        with pytest.raises((ValueError, TypeError)):
            T.app_user_guc(evil)


def test_rls_policy_statements_are_strict_and_fail_safe():
    stmts = T.rls_policy_statements("resumes")
    joined = "\n".join(stmts)
    assert 'ENABLE ROW LEVEL SECURITY' in joined
    assert 'FORCE ROW LEVEL SECURITY' in joined          # even the owner is scoped
    assert 'DROP POLICY IF EXISTS resumes_tenant_isolation' in joined  # idempotent
    # The policy matches user_id to the request's app.user_id; unset/empty →
    # NULL → no rows (the fail-safe: a forgotten scope leaks nothing).
    assert "current_setting('app.user_id', true)" in joined
    assert "NULLIF(" in joined
    assert "user_id =" in joined


def test_rls_only_for_known_user_owned_tables():
    with pytest.raises(ValueError):
        T.rls_policy_statements("llm_models")   # provider config, not user-owned


def test_all_rls_statements_cover_every_table():
    all_sql = "\n".join(T.all_rls_statements())
    for t in T.RLS_TABLES:
        assert f'"{t}"' in all_sql


def test_clear_scope_resets():
    assert T.clear_app_user_guc() == "RESET app.user_id"
