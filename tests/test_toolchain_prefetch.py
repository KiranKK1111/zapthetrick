"""Stage-4 §3.5 — toolchain prefetch: fence-language detection + gated firing.

The `warm_toolchain`/`prefetch_toolchain` primitives are exercised by the pool
suite; here we test the streaming-path helpers this component adds — the first
opening-fence language detector and the enable gate.
"""
from __future__ import annotations

from app.sandbox import pool as P


class TestFirstFenceLanguage:
    def test_detects_opening_fence_tag(self):
        assert P.first_fence_language("Here:\n```python\nprint(1)\n```") == "python"

    def test_none_when_no_fence(self):
        assert P.first_fence_language("just prose, no code") is None

    def test_none_for_bare_fence_without_tag(self):
        assert P.first_fence_language("```\nplain\n```") is None

    def test_first_of_several_wins(self):
        text = "```java\n// a\n```\nthen\n```python\n# b\n```"
        assert P.first_fence_language(text) == "java"

    def test_partial_buffer_mid_stream(self):
        # A fence that has opened but not yet closed still resolves.
        assert P.first_fence_language("intro\n```rust\nfn main() {") == "rust"

    def test_lowercased_and_trimmed(self):
        assert P.first_fence_language("```CPP  \ncode") == "cpp"

    def test_start_of_string_fence(self):
        assert P.first_fence_language("```go\npackage main") == "go"


class TestEnableGate:
    def test_default_off(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.code_solver, "toolchain_prefetch", False,
                            raising=False)
        assert P.prefetch_enabled() is False

    def test_on_when_flagged(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.code_solver, "toolchain_prefetch", True,
                            raising=False)
        assert P.prefetch_enabled() is True


class TestPrefetchIsBestEffort:
    def test_prefetch_unknown_language_is_noop(self):
        # An un-probed language must never raise (fire-and-forget contract).
        P.prefetch_toolchain("cobol-9000")
        P.prefetch_toolchain("")

    def test_prefetch_fires_thread_for_known_lang(self, monkeypatch):
        fired: list[str] = []
        monkeypatch.setattr(P, "warm_toolchain", lambda lang: fired.append(lang))
        # Reset the one-shot guard so this lang can fire.
        with P._warm_lock:
            P._warmed.discard("python")
        P.prefetch_toolchain("python")
        # The daemon thread calls the (stubbed) warm_toolchain.
        import time
        for _ in range(50):
            if fired:
                break
            time.sleep(0.01)
        assert fired == ["python"]
