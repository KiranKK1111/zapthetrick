"""Background research follow-up (perceived-speed R15, task 17.3).

Pins Property 14: disabled → none, material findings → marked follow-up,
immaterial → none, error → none (non-blocking is structural — runs post-answer).
"""
from __future__ import annotations

import asyncio

from app.core.config_loader import cfg
from app.perceived import research as R


def _enable(monkeypatch, on=True):
    monkeypatch.setattr(cfg.perceived, "background_research", on, raising=False)


async def _research_hit(_q, _a):
    return "Redis 7.4 adds hash field TTLs."


async def _research_none(_q, _a):
    return ""


def test_disabled_returns_none(monkeypatch):
    _enable(monkeypatch, on=False)
    out = asyncio.run(R.maybe_followup(
        "q", "a", research=_research_hit, is_material=lambda f, a: True))
    assert out is None


def test_material_findings_become_followup(monkeypatch):
    _enable(monkeypatch, on=True)
    out = asyncio.run(R.maybe_followup(
        "q", "a", research=_research_hit, is_material=lambda f, a: True))
    assert out is not None
    assert R.FOLLOWUP_HEADER in out
    assert "Redis" in out


def test_immaterial_findings_no_followup(monkeypatch):
    _enable(monkeypatch, on=True)
    out = asyncio.run(R.maybe_followup(
        "q", "a", research=_research_hit, is_material=lambda f, a: False))
    assert out is None


def test_empty_findings_no_followup(monkeypatch):
    _enable(monkeypatch, on=True)
    out = asyncio.run(R.maybe_followup(
        "q", "a", research=_research_none, is_material=lambda f, a: True))
    assert out is None


def test_research_error_is_swallowed(monkeypatch):
    _enable(monkeypatch, on=True)

    async def _boom(_q, _a):
        raise RuntimeError("search down")

    out = asyncio.run(R.maybe_followup(
        "q", "a", research=_boom, is_material=lambda f, a: True))
    assert out is None
