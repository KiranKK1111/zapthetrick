"""A6 — precompute full answers for predicted follow-ups into the prepared store
so a matching next question serves instantly. Uses a stub generator + a temp
resume id (no real LLM / no network)."""
import asyncio

import pytest

from app.core.config_loader import cfg
from app.live import prepared, predict


async def _gen_ok(q):
    return (f"A thorough precomputed answer about {q} that comfortably exceeds "
            "the twenty-five word minimum required for it to be cached into the "
            "prepared store for instant serving on a later matching question.")


async def _gen_short(q):
    return "too short"


async def _gen_boom(q):
    raise RuntimeError("no LLM route: providers exhausted")


@pytest.fixture(autouse=True)
def _enable_prepared(monkeypatch):
    monkeypatch.setattr(cfg.live, "prepared_answers", True, raising=False)
    yield
    prepared.drop("test-a6")


def test_precompute_caches_top_k():
    async def run():
        prepared.drop("test-a6")
        preds = ["How do partitions work", "What are consumer groups", "Rebalancing"]
        n = await predict.precompute_followups("test-a6", preds, _gen_ok, limit=2)
        assert n == 2
        # idempotent — same questions add nothing new.
        n2 = await predict.precompute_followups("test-a6", preds[:2], _gen_ok, limit=2)
        assert n2 == 0
    asyncio.run(run())


def test_precompute_skips_too_short():
    async def run():
        prepared.drop("test-a6")
        n = await predict.precompute_followups("test-a6", ["Q one", "Q two"], _gen_short)
        assert n == 0
    asyncio.run(run())


def test_precompute_aborts_on_exhaustion_never_raises():
    async def run():
        prepared.drop("test-a6")
        n = await predict.precompute_followups("test-a6", ["Q one", "Q two"], _gen_boom)
        assert n == 0            # generation failed → nothing cached, no raise
    asyncio.run(run())


def test_precompute_noop_without_resume():
    async def run():
        assert await predict.precompute_followups("", ["Q"], _gen_ok) == 0
    asyncio.run(run())
