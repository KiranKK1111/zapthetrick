"""Tests for continuous screen context (vNext §9.4, Stage 10 Component E)."""
from __future__ import annotations

import app.live.screen_context as S


def _on(monkeypatch):
    monkeypatch.setattr(S, "enabled", lambda: True)


# ---- luma-diff gate -------------------------------------------------------
def test_first_frame_always_processes(monkeypatch):
    _on(monkeypatch)
    d = S.should_process([0.2, 0.3, 0.5])
    assert d.changed and d.reason == "first frame"


def test_static_screen_is_skipped(monkeypatch):
    _on(monkeypatch)
    d = S.should_process([0.2, 0.3, 0.5], [0.2, 0.3, 0.5])
    assert not d.changed                          # zero upstream on a static IDE


def test_changed_screen_processes(monkeypatch):
    _on(monkeypatch)
    d = S.should_process([0.6, 0.2, 0.2], [0.2, 0.3, 0.5])
    assert d.changed and d.delta > 0


def test_tiny_change_below_threshold_skipped(monkeypatch):
    _on(monkeypatch)
    d = S.should_process([0.21, 0.30, 0.49], [0.2, 0.3, 0.5])
    assert not d.changed


def test_scalar_luma_means(monkeypatch):
    _on(monkeypatch)
    assert not S.should_process(0.5, 0.5).changed
    assert S.should_process(0.9, 0.3).changed


def test_disabled_never_processes(monkeypatch):
    monkeypatch.setattr(S, "enabled", lambda: False)
    assert not S.should_process([0.9], [0.1]).changed


def test_luma_delta_never_raises(monkeypatch):
    _on(monkeypatch)
    assert 0.0 <= S.luma_delta(None, None) <= 1.0


# ---- downscale ------------------------------------------------------------
def test_downscale_4k_to_1280_edge():
    assert S.downscale_target(3840, 2160) == (1280, 720)


def test_downscale_leaves_small_frames():
    assert S.downscale_target(800, 600) == (800, 600)


def test_downscale_portrait():
    assert S.downscale_target(1080, 1920) == (720, 1280)


# ---- privacy invariant (cloud refuses ambient) ---------------------------
def test_ambient_only_local():
    assert S.allow_vision_target("local", ambient=True)
    assert not S.allow_vision_target("cloud", ambient=True)   # cloud refuses ambient


def test_user_upload_may_use_cloud():
    assert S.allow_vision_target("cloud", ambient=False)
    assert S.allow_vision_target("local", ambient=False)


def test_unknown_target_refused():
    assert not S.allow_vision_target("elsewhere", ambient=False)


# ---- rolling ScreenState --------------------------------------------------
def test_state_update_and_hint():
    st = S.ScreenState()
    st.update(S.ScreenRead(app="vscode", code_language="python", has_error=True,
                           frame_ts=100))
    hint = st.context_hint()
    assert "vscode" in hint and "python" in hint and "error" in hint
    assert st.to_dict()["has_error"] is True


def test_state_staleness():
    st = S.ScreenState()
    st.update(S.ScreenRead(app="vscode", frame_ts=100))
    assert not st.is_stale(now=110, max_age_s=30)
    assert st.is_stale(now=200, max_age_s=30)


def test_empty_state_is_stale_and_hintless():
    st = S.ScreenState()
    assert st.is_stale(now=100)
    assert st.context_hint() == ""


def test_state_history_bounded():
    st = S.ScreenState()
    for i in range(30):
        st.update(S.ScreenRead(app="x", frame_ts=i))
    assert len(st.history) <= 20


def test_for_tracker_is_stable():
    class T:
        pass
    t = T()
    assert S.for_tracker(t) is S.for_tracker(t)


# ---- live-coding delta verification --------------------------------------
def test_delta_hint_with_shared_technical_token_verified():
    assert S.verify_delta_hint("now runs kubectl apply",
                               'os.system("kubectl apply -f x")')


def test_hallucinated_delta_rejected():
    assert not S.verify_delta_hint("added a websocket handler", "def f(): return 1")


def test_delta_hint_empty_rejected():
    assert not S.verify_delta_hint("", "some code")
    assert not S.verify_delta_hint("something", "")


def test_delta_hint_semantic_seam():
    assert S.verify_delta_hint("anything", "code", verify_fn=lambda h, s: True)
    assert not S.verify_delta_hint("kubectl apply", "kubectl apply here",
                                   verify_fn=lambda h, s: False)


def test_verify_never_raises():
    assert S.verify_delta_hint(None, None) is False   # type: ignore[arg-type]
