"""Stage-6 §9.1 — GPU admission & scheduling plane: lanes, ledger, sessions, degrade."""
from __future__ import annotations

import pytest

from app.gpu import scheduler as G
from app.gpu.scheduler import GpuScheduler


def _sched(vram=10_000, cap=2, headroom=1_000):
    return GpuScheduler(total_vram_mb=vram, max_live_sessions=cap,
                        bg_headroom_mb=headroom)


class TestVramLedger:
    def test_reserve_and_free(self):
        s = _sched(vram=10_000)
        assert s.free_mb == 10_000
        s.admit(G.INTERACTIVE, "vlm", est_mb=4_000)
        assert s.reserved_mb == 4_000 and s.free_mb == 6_000
        s.release("vlm")
        assert s.free_mb == 10_000

    def test_reserved_never_negative_after_double_release(self):
        s = _sched()
        s.admit(G.INTERACTIVE, "x", est_mb=1_000)
        s.release("x")
        s.release("x")                      # idempotent
        assert s.reserved_mb == 0


class TestRealtimeLane:
    def test_realtime_always_runs(self):
        s = _sched(vram=1_000)
        # Even when VRAM is nearly gone, realtime is never rejected.
        s.admit(G.INTERACTIVE, "vlm", est_mb=900)
        a = s.admit(G.REALTIME, "stt", est_mb=500)
        assert a.admit is True and a.action == G.RUN


class TestInteractiveLane:
    def test_runs_when_it_fits(self):
        s = _sched(vram=10_000)
        a = s.admit(G.INTERACTIVE, "vlm", est_mb=4_000)
        assert a.action == G.RUN

    def test_sheds_to_cloud_when_it_does_not_fit(self):
        s = _sched(vram=5_000)
        s.admit(G.INTERACTIVE, "a", est_mb=4_000)     # 1000 free
        a = s.admit(G.INTERACTIVE, "b", est_mb=3_000)  # doesn't fit
        assert a.admit is False and a.action == G.SHED_CLOUD
        assert s.reserved_mb == 4_000                  # b not reserved


class TestBackgroundLane:
    def test_runs_when_idle_with_headroom(self):
        s = _sched(vram=10_000, headroom=1_000)
        a = s.admit(G.BACKGROUND, "raster", est_mb=500)
        assert a.action == G.RUN

    def test_defers_when_interactive_active(self):
        s = _sched(vram=10_000)
        s.admit(G.INTERACTIVE, "vlm", est_mb=2_000)
        a = s.admit(G.BACKGROUND, "raster", est_mb=500)
        assert a.admit is False and a.action == G.DEFER

    def test_defers_when_realtime_active(self):
        s = _sched(vram=10_000)
        s.admit(G.REALTIME, "stt", est_mb=300)
        assert s.admit(G.BACKGROUND, "raster", est_mb=500).action == G.DEFER

    def test_defers_without_headroom(self):
        s = _sched(vram=2_000, headroom=1_000)
        # idle, but 1600 + 1000 headroom > 2000 free → defer.
        a = s.admit(G.BACKGROUND, "raster", est_mb=1_600)
        assert a.action == G.DEFER

    def test_runs_after_interactive_releases(self):
        s = _sched(vram=10_000)
        s.admit(G.INTERACTIVE, "vlm", est_mb=2_000)
        s.release("vlm")
        assert s.admit(G.BACKGROUND, "raster", est_mb=500).action == G.RUN


class TestSessions:
    def test_cap_enforced(self):
        s = _sched(cap=2)
        assert s.open_session("a") is True
        assert s.open_session("b") is True
        assert s.open_session("c") is False       # over cap
        assert s.live_sessions == 2

    def test_reopen_is_idempotent(self):
        s = _sched(cap=1)
        assert s.open_session("a") is True
        assert s.open_session("a") is True        # same session, still fine
        assert s.live_sessions == 1

    def test_close_frees_a_slot(self):
        s = _sched(cap=1)
        s.open_session("a")
        assert s.open_session("b") is False
        s.close_session("a")
        assert s.open_session("b") is True


class TestDegradeOrder:
    def test_sheds_cheapest_first(self):
        s = _sched()
        assert s.next_to_shed(["vlm", "screen", "pre_answer"]) == "pre_answer"
        assert s.next_to_shed(["vlm", "screen"]) == "screen"
        assert s.next_to_shed(["vlm", "speculation_top1"]) == "speculation_top1"
        assert s.next_to_shed(["vlm"]) == "vlm"      # last resort

    def test_nothing_sheddable(self):
        assert _sched().next_to_shed([]) is None
        assert _sched().next_to_shed(["something_else"]) is None


class TestStatsAndFlag:
    def test_stats_shape(self):
        s = _sched()
        s.admit(G.INTERACTIVE, "vlm", est_mb=1_000)
        s.open_session("a")
        st = s.stats()
        assert st["reserved_mb"] == 1_000 and st["live_sessions"] == 1
        assert st["per_lane"].get("interactive") == 1

    def test_enabled_default_off(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.gpu, "scheduler", False, raising=False)
        assert G.enabled() is False

    def test_singleton_built_from_config(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.gpu, "total_vram_mb", 8_000, raising=False)
        G.reset_for_tests()
        assert G.scheduler().total_vram_mb == 8_000
        G.reset_for_tests()
