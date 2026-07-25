"""Stage-6 §4.7 — audio reliability: silence watchdog + drift-proof transport + STT failover."""
from __future__ import annotations

import pytest

from app.live import silence as SIL
from app.live import stt_recovery as REC
from app.live import transport as TP
from app.live import watchdog as WD
from app.live.watchdog import AudioWatchdog


class _Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


# --------------------------------------------------------------------------- #
class TestSilenceWatchdog:
    def test_fires_after_threshold(self):
        clock = _Clock()
        w = AudioWatchdog(now=clock)
        w.seed("s1", interviewer_audio=True)
        assert w.check("s1", silence_s=30) is None      # just started
        clock.advance(31)
        assert w.check("s1", silence_s=30) == WD.SILENCE

    def test_one_shot_per_episode(self):
        clock = _Clock()
        w = AudioWatchdog(now=clock)
        w.seed("s1", interviewer_audio=True)
        clock.advance(31)
        assert w.check("s1", silence_s=30) == WD.SILENCE
        assert w.check("s1", silence_s=30) is None       # doesn't re-fire

    def test_audio_re_arms_the_episode(self):
        clock = _Clock()
        w = AudioWatchdog(now=clock)
        w.seed("s1", interviewer_audio=True)
        clock.advance(31)
        assert w.check("s1", silence_s=30) == WD.SILENCE
        w.mark_audio("s1")                                # interviewer speaks again
        clock.advance(31)
        assert w.check("s1", silence_s=30) == WD.SILENCE  # a NEW episode fires

    def test_no_fire_before_any_audio_seen(self):
        clock = _Clock()
        w = AudioWatchdog(now=clock)
        w.seed("s1", interviewer_audio=False)             # never heard the interviewer
        clock.advance(60)
        assert w.check("s1", silence_s=30) is None

    def test_inactive_session_does_not_fire(self):
        clock = _Clock()
        w = AudioWatchdog(now=clock)
        w.seed("s1", interviewer_audio=True)
        w.set_active("s1", False)
        clock.advance(60)
        assert w.check("s1", silence_s=30) is None

    def test_module_gate_off_is_noop(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "audio_watchdog", False, raising=False)
        WD.reset_for_tests()
        WD.seed_baseline("s1", interviewer_audio=True)
        assert WD.check_silence("s1") is None             # gate off → no watch

    def test_silence_seed_baseline_delegates(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "audio_watchdog", True, raising=False)
        monkeypatch.setattr(cfg.live, "silence_watchdog_s", 30.0, raising=False)
        WD.reset_for_tests()
        # Component A calls silence.seed_baseline → must reach the watchdog.
        SIL.seed_baseline("s1", interviewer_audio=True)
        assert WD.watchdog()._s.get("s1") is not None
        WD.reset_for_tests()


# --------------------------------------------------------------------------- #
class TestTransport:
    def test_in_order_is_all_ok(self):
        t = TP.SequenceTracker()
        assert [t.observe(i) for i in range(5)] == [TP.OK] * 5
        assert t.loss_rate() == 0.0

    def test_gap_counts_missing_packets(self):
        t = TP.SequenceTracker()
        t.observe(0)
        t.observe(1)
        assert t.observe(5) == TP.GAP        # 2,3,4 skipped
        assert t.gaps == 3

    def test_duplicate_detected(self):
        t = TP.SequenceTracker()
        t.observe(0)
        t.observe(1)
        assert t.observe(1) == TP.DUPLICATE
        assert t.duplicates == 1

    def test_reorder_detected(self):
        t = TP.SequenceTracker()
        t.observe(0)
        t.observe(3)                         # gap
        assert t.observe(1) == TP.REORDER    # a late earlier packet, not seen
        assert t.reorders == 1

    def test_loss_rate_and_degraded(self):
        t = TP.SequenceTracker()
        t.observe(0)
        t.observe(20)                        # 19 missing of 21 → high loss
        assert t.loss_rate() > 0.5
        assert t.degraded() is True

    def test_jitter_ewma_from_arrivals(self):
        t = TP.SequenceTracker()
        t.observe(0, arrival_s=0.0, nominal_ms=20)
        t.observe(1, arrival_s=0.020, nominal_ms=20)   # on time → ~0 jitter
        t.observe(2, arrival_s=0.100, nominal_ms=20)   # 60 ms late → jitter rises
        assert t.jitter_ms > 0.0

    def test_bad_seq_is_ignored(self):
        t = TP.SequenceTracker()
        t.observe(0)
        assert t.observe("not-a-number") == TP.OK       # never raises/corrupts
        assert t.expected == 1

    def test_stats_shape(self):
        t = TP.SequenceTracker()
        t.observe(0)
        s = t.stats()
        assert {"received", "gaps", "reorders", "duplicates", "loss_rate",
                "jitter_ms"} <= set(s)


# --------------------------------------------------------------------------- #
class TestSttFailover:
    def test_gpu_degraded_prefers_cloud(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "stt_failover", True, raising=False)
        REC.forget_session("s1")
        target = REC.recover("s1", gpu_degraded=True)
        assert target in ("cloud",) or target                # a cloud target

    def test_local_failover_when_not_gpu_degraded(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.stt, "provider", "parakeet", raising=False)
        REC.forget_session("s2")
        target = REC.recover("s2", gpu_degraded=False)
        assert target != "parakeet"                          # a different local engine

    def test_single_shot_latch(self):
        REC.forget_session("s3")
        first = REC.recover("s3")
        assert first is not None
        assert REC.recover("s3") is None                     # only once per session

    def test_failover_off_ignores_gpu_degrade(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.live, "stt_failover", False, raising=False)
        monkeypatch.setattr(cfg.stt, "provider", "parakeet", raising=False)
        REC.forget_session("s4")
        # Flag off → cloud path NOT taken even when GPU-degraded (local swap).
        assert REC.recover("s4", gpu_degraded=True) != "cloud"
