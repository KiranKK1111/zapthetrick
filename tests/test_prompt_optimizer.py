"""Autonomous prompt optimization (P7 #5): benchmark → pick best → gated promote."""
from __future__ import annotations

from app.eval.prompt_eval import PromptCase, PromptVariant
from app.eval.prompt_optimizer import active_version, optimize, reset, set_active
from app.eval.scoring import contains_all


def _cases():
    # A gate that wants the word "concise" in the output.
    return [PromptCase(inputs={"q": "explain X"},
                       gates=[contains_all("concise")], name="c1")]


def _gen(prompt: str) -> str:
    # The generator echoes the prompt — so a variant whose TEMPLATE contains
    # "concise" produces a passing output; one that doesn't, fails the gate.
    return prompt


def test_first_run_sets_a_champion():
    reset()
    v1 = PromptVariant("answer", "v1", "Answer: {q}")            # no 'concise'
    v2 = PromptVariant("answer", "v2", "Answer concise: {q}")    # has 'concise'
    out = optimize("answer", [v1, v2], _cases(), _gen)
    assert out["promoted"] is True
    assert out["active"] == "v2"                # best variant wins first slot
    assert active_version("answer") == "v2"


def test_better_variant_is_promoted_over_champion():
    reset()
    set_active("answer", "v1")                  # champion is the weaker one
    v1 = PromptVariant("answer", "v1", "Answer: {q}")
    v2 = PromptVariant("answer", "v2", "Answer concise: {q}")
    out = optimize("answer", [v1, v2], _cases(), _gen)
    assert out["promoted"] is True and out["active"] == "v2"
    assert out["delta"] is not None and out["delta"] > 0


def test_champion_kept_when_no_candidate_beats_it():
    reset()
    set_active("answer", "v2")                  # champion already best
    v1 = PromptVariant("answer", "v1", "Answer: {q}")
    v2 = PromptVariant("answer", "v2", "Answer concise: {q}")
    out = optimize("answer", [v1, v2], _cases(), _gen)
    assert out["promoted"] is False and out["active"] == "v2"


def test_default_optimization_invoker_runs_end_to_end():
    """P7 #5: the reachable no-arg invoker the roadmap said was missing."""
    from app.eval.prompt_optimizer import run_default_optimization
    reset()
    out = run_default_optimization()
    # First run always names a champion; the best variant (v3, most qualities)
    # should win with the echo generator.
    assert out["promoted"] is True
    assert out["active"] in ("v2", "v3")
    assert "scores" in out


def test_default_optimization_is_fail_open():
    from app.eval.prompt_optimizer import run_default_optimization
    reset()

    def boom(_prompt: str) -> str:
        raise RuntimeError("generator down")

    out = run_default_optimization(boom)
    # A generator that raises scores every case 0 → no crash, decision returned.
    assert "promoted" in out
