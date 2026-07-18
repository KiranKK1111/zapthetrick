"""Tests for the 95%-push modules:
  #7  Shadow Execution         (app/eval/shadow.py)          — Phase 7
  #13 Mid-stream Quality Ctrl  (app/quality/stream_controller.py) — Phase 5
  #19 Contextual chunking      (app/rag/contextual.py)       — Phase 3
  #5  Prompt Compiler          (app/core/prompt_compiler.py) — Phase 8
All deterministic + offline.
"""
from __future__ import annotations

from app.core.prompt_compiler import PromptSpec, compile_prompt
from app.eval import shadow
from app.quality import stream_controller as qc
from app.rag import contextual


# ── Shadow Execution (#7) ───────────────────────────────────────────────────
def test_shadow_promotes_better_candidate():
    cases = [1, 2, 3]
    base = lambda c: 0.5
    cand = lambda c: 0.8
    r = shadow.run_shadow(cases, base, cand)
    assert r.delta > 0 and r.improved_cases == 3 and r.regressed_cases == 0
    assert shadow.should_promote(r)


def test_shadow_rejects_regressing_candidate():
    cases = [1, 2, 3]
    base = lambda c: 0.8
    cand = lambda c: 0.9 if c == 1 else 0.5  # improves one, regresses two
    r = shadow.run_shadow(cases, base, cand)
    assert r.regressed_cases == 2
    assert not shadow.should_promote(r)  # regressions block promotion


def test_shadow_is_fail_open_per_case():
    def boom(_c):
        raise RuntimeError("x")
    r = shadow.run_shadow([1], boom, lambda c: 1.0)
    assert r.baseline_mean == 0.0 and r.candidate_mean == 1.0


# ── Mid-stream Quality Controller (#13) ─────────────────────────────────────
def test_quality_flags_refusal_leak():
    v = qc.assess_partial("I can't help with that request.")
    assert v.action in (qc.FLAG, qc.REGENERATE)
    assert "refusal_leak" in v.reasons


def test_quality_allows_refusal_when_expected():
    v = qc.assess_partial("I can't help with that.", expect_refusal_ok=True)
    assert "refusal_leak" not in v.reasons


def test_quality_flags_degenerate_repetition():
    v = qc.assess_partial("the the the the the the the the the the")
    assert v.action == qc.REGENERATE
    assert any("degenerate" in r for r in v.reasons)


def test_quality_passes_good_answer():
    v = qc.assess_partial("Kafka partitions distribute load across brokers; "
                          "each partition is an ordered, replicated log.")
    assert v.action == qc.CONTINUE and v.score >= 0.75


# ── Contextual chunking (#19) ───────────────────────────────────────────────
def test_contextualize_prepends_header():
    out = contextual.contextualize("Improved throughput by 40%.",
                                   doc_title="Jane Doe Resume", section="Experience")
    assert out.startswith("[Jane Doe Resume — Experience]")
    assert "Improved throughput by 40%." in out


def test_contextualize_noop_without_context():
    assert contextual.contextualize("bare chunk") == "bare chunk"


def test_contextualize_all():
    outs = contextual.contextualize_all(["a", "b"], doc_title="Doc")
    assert all(o.startswith("[Doc]") for o in outs)


# ── Prompt Compiler (#5) ────────────────────────────────────────────────────
def test_compile_orders_and_labels_sections():
    spec = PromptSpec(system="Be precise.", intent="coding",
                      evidence=["Java 21", "Spring Boot 3.5"],
                      constraints=["include tests"], task="write a REST controller")
    out = compile_prompt(spec)
    assert out.index("# SYSTEM") < out.index("# EVIDENCE") < out.index("# TASK")
    assert "- Java 21" in out and "- include tests" in out


def test_compile_dedupes_and_omits_empty():
    spec = PromptSpec(context=["x", "x", "y"], task="do it")
    out = compile_prompt(spec)
    assert out.count("- x") == 1
    assert "# EVIDENCE" not in out  # empty section omitted


def test_compile_fail_open_returns_task():
    assert "do it" in compile_prompt(PromptSpec(task="do it"))
