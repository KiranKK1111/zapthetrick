"""Phase 1b — Document Job Manager (queue / priority / concurrency / cancel /
timeout / retry / progress / cleanup)."""
from __future__ import annotations

import asyncio
import time

import pytest

from app.documents.jobs import DocumentJobManager, JobStatus


def _ok_render(content, fmt, title):
    return (f"{title}:{content}".encode(), "text/plain", fmt or "txt")


async def _mgr(**kw):
    m = DocumentJobManager(track=False, **kw)
    await m.start()
    return m


def test_basic_render():
    async def run():
        async with DocumentJobManager(track=False, render_fn=_ok_render) as m:
            job = await m.submit_and_wait("hello", "txt", "T")
            assert job.status == JobStatus.DONE
            assert job.result == b"T:hello" and job.ext == "txt"
            assert job.progress == 1.0 and job.attempts == 1
    asyncio.run(run())


def test_bounded_concurrency():
    live = {"now": 0, "max": 0}

    def slow(content, fmt, title):
        live["now"] += 1
        live["max"] = max(live["max"], live["now"])
        time.sleep(0.05)
        live["now"] -= 1
        return (b"x", "text/plain", "txt")

    async def run():
        async with DocumentJobManager(track=False, render_fn=slow,
                                      max_concurrent=2) as m:
            ids = [m.submit("c", "txt") for _ in range(6)]
            await asyncio.gather(*(m.wait(i) for i in ids))
        assert live["max"] <= 2   # never more than the worker cap in flight
    asyncio.run(run())


def test_priority_ordering():
    order: list[int] = []

    def rec(content, fmt, title):
        order.append(int(content))
        time.sleep(0.01)
        return (b"x", "text/plain", "txt")

    async def run():
        # One worker → strict ordering. Submit low prio first, then high.
        async with DocumentJobManager(track=False, render_fn=rec,
                                      max_concurrent=1) as m:
            a = m.submit("1", "txt", priority=0)
            b = m.submit("2", "txt", priority=0)
            c = m.submit("3", "txt", priority=10)   # jumps the queue
            await asyncio.gather(m.wait(a), m.wait(b), m.wait(c))
        # First job may already be running; the high-priority one must beat the
        # remaining low-priority one.
        assert order.index(3) < order.index(2)
    asyncio.run(run())


def test_cancel_queued_job():
    def slow(content, fmt, title):
        time.sleep(0.1)
        return (b"x", "text/plain", "txt")

    async def run():
        async with DocumentJobManager(track=False, render_fn=slow,
                                      max_concurrent=1) as m:
            first = m.submit("a", "txt")     # occupies the single worker
            queued = m.submit("b", "txt")    # waits behind it
            assert m.cancel(queued) is True
            j = await m.wait(queued)
            assert j.status == JobStatus.CANCELLED and j.result is None
            await m.wait(first)
    asyncio.run(run())


def test_timeout_then_no_retry():
    def hang(content, fmt, title):
        time.sleep(0.3)
        return (b"x", "text/plain", "txt")

    async def run():
        async with DocumentJobManager(track=False, render_fn=hang,
                                      timeout_s=0.05, max_retries=0) as m:
            j = await m.submit_and_wait("a", "txt")
            assert j.status == JobStatus.TIMEOUT and j.attempts == 1
    asyncio.run(run())


def test_retry_recovers():
    calls = {"n": 0}

    def flaky(content, fmt, title):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return (b"ok", "text/plain", "txt")

    async def run():
        async with DocumentJobManager(track=False, render_fn=flaky,
                                      max_retries=1) as m:
            j = await m.submit_and_wait("a", "txt")
            assert j.status == JobStatus.DONE and j.attempts == 2
            assert j.result == b"ok"
    asyncio.run(run())


def test_failure_after_retries():
    def boom(content, fmt, title):
        raise ValueError("nope")

    async def run():
        async with DocumentJobManager(track=False, render_fn=boom,
                                      max_retries=1) as m:
            j = await m.submit_and_wait("a", "txt")
            assert j.status == JobStatus.FAILED and "nope" in j.error
            assert j.attempts == 2
    asyncio.run(run())


class TestBoundedRetry:
    """BUG 2 — retry is bounded, backed off, transient-only, cancellable."""

    def test_backoff_waits_between_attempts(self):
        stamps: list[float] = []

        def flaky(content, fmt, title):
            stamps.append(time.monotonic())
            if len(stamps) == 1:
                raise OSError("transient disk hiccup")
            return (b"ok", "text/plain", "txt")

        async def run():
            async with DocumentJobManager(track=False, render_fn=flaky,
                                          max_retries=1,
                                          retry_backoff_s=0.15) as m:
                j = await m.submit_and_wait("a", "txt")
                assert j.status == JobStatus.DONE and j.attempts == 2
            # It actually backed off instead of retrying instantly. Compared
            # against a floor, not the exact 0.15: the backoff waits on
            # `asyncio.wait_for`, and the Windows event-loop timer has ~15.6ms
            # granularity, so it can wake a hair early. An instant retry would
            # be ~0.0001s, so this floor still fails loudly if the sleep is
            # skipped — it only tolerates clock granularity, not a missing wait.
            assert stamps[1] - stamps[0] >= 0.10
        asyncio.run(run())

    def test_deterministic_failure_is_not_retried(self):
        from app.documents.generators import UnsupportedFormat
        calls = {"n": 0}

        def bad_format(content, fmt, title):
            calls["n"] += 1
            raise UnsupportedFormat("Unsupported format 'xyz'")

        async def run():
            async with DocumentJobManager(track=False, render_fn=bad_format,
                                          max_retries=3,
                                          retry_backoff_s=0.01) as m:
                j = await m.submit_and_wait("a", "xyz")
                assert j.status == JobStatus.FAILED
                assert j.attempts == 1          # failed fast, no wasted retries
            assert calls["n"] == 1
        asyncio.run(run())

    def test_malformed_input_typeerror_is_not_retried(self):
        calls = {"n": 0}

        def bad_input(content, fmt, title):
            calls["n"] += 1
            raise TypeError("expected str, got dict")

        async def run():
            async with DocumentJobManager(track=False, render_fn=bad_input,
                                          max_retries=2,
                                          retry_backoff_s=0.01) as m:
                j = await m.submit_and_wait("a", "txt")
                assert j.status == JobStatus.FAILED and j.attempts == 1
            assert calls["n"] == 1
        asyncio.run(run())

    def test_retry_is_bounded(self):
        calls = {"n": 0}

        def always_transient(content, fmt, title):
            calls["n"] += 1
            raise RuntimeError("still flaky")

        async def run():
            async with DocumentJobManager(track=False,
                                          render_fn=always_transient,
                                          max_retries=2,
                                          retry_backoff_s=0.01) as m:
                j = await m.submit_and_wait("a", "txt")
                assert j.status == JobStatus.FAILED and j.attempts == 3
            assert calls["n"] == 3               # 1 try + 2 retries, no more
        asyncio.run(run())

    def test_cancel_during_backoff_stops_the_retry(self):
        calls = {"n": 0}

        def flaky(content, fmt, title):
            calls["n"] += 1
            raise RuntimeError("transient")

        async def run():
            async with DocumentJobManager(track=False, render_fn=flaky,
                                          max_concurrent=1, max_retries=3,
                                          retry_backoff_s=0.3) as m:
                jid = m.submit("a", "txt")
                await asyncio.sleep(0.1)         # first attempt has failed
                assert m.cancel(jid) is True     # cancel mid-backoff
                j = await m.wait(jid)
                assert j.status == JobStatus.CANCELLED
            assert calls["n"] == 1               # never re-rendered
        asyncio.run(run())

    def test_timeout_is_retried_then_gives_up(self):
        def hang(content, fmt, title):
            time.sleep(0.3)
            return (b"x", "text/plain", "txt")

        async def run():
            async with DocumentJobManager(track=False, render_fn=hang,
                                          timeout_s=0.05, max_retries=1,
                                          retry_backoff_s=0.01) as m:
                j = await m.submit_and_wait("a", "txt")
                assert j.status == JobStatus.TIMEOUT and j.attempts == 2
        asyncio.run(run())

    def test_retries_come_from_config(self, monkeypatch):
        import app.documents.jobs as jobs_mod

        class _Docs:
            export_concurrency = 3
            export_timeout_s = 42.0
            export_max_retries = 0
            export_retry_backoff_s = 0.0

        class _Cfg:
            documents = _Docs()

        monkeypatch.setattr(jobs_mod, "_MANAGER", None, raising=False)
        monkeypatch.setattr("app.core.config_loader.get_config",
                            lambda: _Cfg(), raising=False)

        async def run():
            m = await jobs_mod.get_manager()
            assert m._max_retries == 0 and m._backoff == 0.0
            assert m._max == 3 and m._timeout == 42.0
            await m.aclose()
        asyncio.run(run())
        # monkeypatch restores `_MANAGER`; a stale one is recreated on the next
        # loop change anyway (see get_manager).


def test_progress_events():
    seen: list[tuple[str, float]] = []

    async def run():
        async with DocumentJobManager(track=False, render_fn=_ok_render) as m:
            m.submit("a", "txt", on_progress=lambda j: seen.append(
                (j.status.value, j.progress)))
            await asyncio.sleep(0.05)
        statuses = [s for s, _ in seen]
        assert "running" in statuses and "done" in statuses
        assert seen[-1] == ("done", 1.0)
    asyncio.run(run())


def test_cleanup_drops_old_finished():
    clock = {"t": 1000.0}

    async def run():
        m = DocumentJobManager(track=False, render_fn=_ok_render,
                               clock=lambda: clock["t"])
        await m.start()
        await m.submit_and_wait("a", "txt")
        assert m.cleanup(ttl_s=100.0) == 0     # just finished, not stale
        clock["t"] += 200.0
        assert m.cleanup(ttl_s=100.0) == 1     # now older than the TTL
        assert m.stats()["total"] == 0
        await m.aclose()
    asyncio.run(run())


def test_real_render_integration():
    # No injected render_fn → uses the real render_document (markdown → html).
    async def run():
        async with DocumentJobManager(track=False) as m:
            j = await m.submit_and_wait("# Hi\n\nbody", "html", "Doc")
            assert j.status == JobStatus.DONE
            assert j.result.decode().startswith("<!doctype html>")
            assert j.mime.startswith("text/html")
    asyncio.run(run())
