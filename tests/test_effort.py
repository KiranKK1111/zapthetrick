"""Stage-7 §8.2 — adaptive reasoning effort dial."""
from __future__ import annotations

import pytest

from app.llm import effort as E


@pytest.fixture
def _on(monkeypatch):
    from app.core.config_loader import cfg
    monkeypatch.setattr(cfg.llm, "effort_dial", True, raising=False)


class TestBaseProfiles:
    def test_effort_scales_with_difficulty(self, _on):
        t = E.effort_for("trivial")
        x = E.effort_for("expert")
        assert t.thinking_budget < x.thinking_budget
        assert t.best_of_n <= x.best_of_n
        assert t.reasoning is False and x.reasoning is True

    def test_trivial_spends_nothing(self, _on):
        p = E.effort_for("trivial")
        assert p.tier == "fast" and p.thinking_budget == 0 and p.best_of_n == 1
        assert p.use_judge is False

    def test_hard_uses_best_of_n_and_judge(self, _on):
        p = E.effort_for("hard")
        assert p.best_of_n >= 2 and p.use_judge is True

    def test_expert_routes_to_reasoning(self, _on):
        assert E.effort_for("expert").reasoning is True

    def test_unknown_difficulty_defaults_standard(self, _on):
        assert E.effort_for("wizard").tier == "standard"


class TestModeShading:
    def test_thorough_shifts_up(self, _on):
        # hard + thorough → the expert band (reasoning tier).
        assert E.effort_for("hard", mode="thorough").tier == "reasoning"

    def test_fast_shifts_down(self, _on):
        assert E.effort_for("hard", mode="fast").tier == "standard"

    def test_balanced_is_neutral(self, _on):
        assert E.effort_for("hard", mode="balanced").tier \
            == E.effort_for("hard").tier

    def test_shift_clamps_at_ends(self, _on):
        assert E.effort_for("expert", mode="thorough").tier == "reasoning"
        assert E.effort_for("trivial", mode="fast").tier == "fast"


class TestEscalation:
    def test_repair_escalates_to_reasoning(self, _on):
        p = E.effort_for("standard", repair_stage=1)
        assert p.tier == "reasoning" and p.reasoning is True and p.escalate is True

    def test_dsa_optimal_routes_to_reasoning(self, _on):
        assert E.effort_for("standard", dsa_optimal=True).reasoning is True


class TestFlag:
    def test_disabled_returns_base_no_shading(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.llm, "effort_dial", False, raising=False)
        # Off → base profile; mode AND repair are ignored (byte-identical).
        assert E.effort_for("hard", mode="fast").tier == "hard"
        assert E.effort_for("standard", repair_stage=2).escalate is False

    def test_thinking_summary_directive(self):
        d = E.thinking_summary_directive()
        assert "considering" in d.lower() and "chain-of-thought" in d.lower()

    def test_never_raises(self, _on):
        E.effort_for(None)                     # type: ignore[arg-type]
