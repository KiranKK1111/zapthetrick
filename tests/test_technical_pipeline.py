"""Phase 7 — architecture depth (#19/#40/#41/#88/#142).

Domain classification + the architecture-grade structured output (domain
sections + Assumptions/Pattern/Trade-offs/Governance) with a checklist verifier.
The LLM is faked so tests are offline/deterministic.
"""
from __future__ import annotations

import asyncio

from app.technical_pipeline import structured
from app.technical_pipeline.dispatcher import DOMAINS, classify_domain
from app.technical_pipeline.structured import (
    ARCHITECTURE_SECTIONS,
    DomainSpec,
    check_missing,
)


# ── classification ──────────────────────────────────────────────────────────
def test_classify_security():
    assert classify_domain("how do I prevent SQL injection and XSS") == "security"
    assert classify_domain("design an OAuth2 + JWT auth flow") == "security"


def test_classify_backend():
    assert classify_domain("design a REST api with pagination") == "backend"
    assert classify_domain("add a graphql endpoint with rate limiting") == "backend"


def test_classify_other_domains():
    assert classify_domain("design a scalable system with sharding") == "system_design"
    assert classify_domain("which index for this query plan in postgres") == "databases"
    assert classify_domain("set up a github actions ci/cd pipeline") == "devops"
    assert classify_domain("which aws managed service for s3 events") == "cloud"
    assert classify_domain("react ssr hydration and web vitals") == "frontend"
    assert classify_domain("tell me a joke") == "generic"


def test_new_domains_registered():
    assert "backend" in DOMAINS and "security" in DOMAINS


# ── checklist verifier ──────────────────────────────────────────────────────
def test_check_missing_detects_absent_sections():
    checklist = [("authn", ["oauth", "jwt"]), ("crypto", ["encryption"])]
    assert check_missing("we use jwt tokens", checklist) == ["crypto"]
    assert check_missing("nothing relevant", checklist) == ["authn", "crypto"]
    assert check_missing("jwt and encryption", checklist) == []


def test_architecture_sections_present():
    headings = [s[0] for s in ARCHITECTURE_SECTIONS]
    assert headings == ["Assumptions", "Recommended Pattern(s)",
                        "Trade-offs", "Governance & Operability"]


# ── structured run (LLM faked) ──────────────────────────────────────────────
def _events(gen):
    async def go():
        return [e async for e in gen]
    return asyncio.run(go())


_FULL = (
    "## Threat Model\nAn attacker (STRIDE) could target the login.\n"
    "## Authentication & Authorization\nUse OAuth2 + JWT with RBAC.\n"
    "## Data Protection\nEncryption at rest, TLS in transit.\n"
    "## Input Validation & Hardening\nParameterized queries stop injection; escape XSS.\n"
    "## Detection & Response\nAudit logging + alerting on anomalies.\n"
    "## Assumptions\nWe assume a public web app.\n"
    "## Recommended Pattern(s)\nThe pattern is defense-in-depth.\n"
    "## Trade-offs\nPros and cons vs an alternative session model.\n"
    "## Governance & Operability\nSecurity reviews, observability, cost.\n"
)


def _fake_llm(text):
    async def _complete(messages, **kw):
        return text
    return _complete


def test_security_run_full_answer_passes(monkeypatch):
    from app.technical_pipeline import security
    monkeypatch.setattr(structured.llm, "complete", _fake_llm(_FULL))
    evts = _events(security.run("secure my login"))
    kinds = [e["kind"] for e in evts]
    assert kinds[0] == "stage" and evts[0]["data"]["domain"] == "security"
    assert "markdown" in kinds
    assert kinds[-1] == "done"
    # All sections present → no verify-failure event.
    assert "verify" not in kinds
    assert evts[-1]["data"]["missing"] == []


def test_run_flags_missing_sections(monkeypatch):
    from app.technical_pipeline import security
    monkeypatch.setattr(structured.llm, "complete",
                        _fake_llm("Just use HTTPS and call it a day."))
    evts = _events(security.run("secure my login"))
    verify = [e for e in evts if e["kind"] == "verify"]
    assert verify and verify[0]["failed"] > 0
    # The cross-cutting architecture sections are part of what's checked.
    done = evts[-1]
    assert "Assumptions" in done["data"]["missing"]
    assert "Trade-offs" in done["data"]["missing"]


def test_backend_run_emits_expected_shape(monkeypatch):
    from app.technical_pipeline import backend
    monkeypatch.setattr(structured.llm, "complete", _fake_llm(_FULL))
    evts = _events(backend.run("design a REST api"))
    assert evts[0]["data"]["domain"] == "backend"
    assert any(e["kind"] == "markdown" for e in evts)


def test_databases_lint_flags_dangerous_sql(monkeypatch):
    from app.technical_pipeline import databases
    sql_answer = _FULL + "\n```sql\nDELETE FROM users;\n```\n"
    monkeypatch.setattr(structured.llm, "complete", _fake_llm(sql_answer))
    evts = _events(databases.run("how to delete users"))
    verify = [e for e in evts if e["kind"] == "verify"]
    assert verify
    assert any("DELETE without WHERE" in e for e in verify[0]["errors"])


def test_devops_spec_emits_artifacts():
    from app.technical_pipeline.devops import _SPEC
    from app.response_arch import Shape
    assert _SPEC.emit_artifacts is True
    assert _SPEC.shape == Shape.ARTIFACT_SET


def test_llm_failure_is_graceful(monkeypatch):
    from app.core.llm_client import LLMError
    from app.technical_pipeline import backend

    async def _boom(*a, **k):
        raise LLMError("provider down")
    monkeypatch.setattr(structured.llm, "complete", _boom)
    evts = _events(backend.run("anything"))
    assert evts[-1]["kind"] == "done"
    assert "warning" in evts[-1]["data"]
