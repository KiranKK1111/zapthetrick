"""Phase-3 (ArchitectureVerdict.md): declarative policy engine.

Contract under test:
  * Builtin rules replicate the legacy final-gate cascade EXACTLY — for a
    corpus of prompts, engine-on and engine-off produce identical decisions.
  * Config rules overlay builtins by id (add / override / disable), scoring
    (not first-match) picks the winner, safety priority dominates.
  * Every decision carries an audit record (rules fired + scores).
  * Broken rules are skipped; a broken engine falls back to the legacy gate.
"""
from __future__ import annotations

import pytest

from app.clarify import intent_pipeline as ip
from app.policy.engine import (ACTION_ANSWER, ACTION_CLARIFY, ACTION_DEFER,
                               PolicyRule, decide, load_rules)

# A behavioral corpus spanning the three legacy outcomes.
_CORPUS = [
    "what is a hash map?",
    "write a login api",                       # clarify: language missing
    "write a login api in python",             # answer
    "compare kafka and rabbitmq",
    "build me an app",                         # clarify: stack missing
    "explain this code",
    "document this",                           # clarify: doc format
    "compress this project",                   # clarify: archive format
    "hey there",
    "asdf qwerty zxcv",                        # unknown → defer
]


class TestBuiltinEquivalence:
    @pytest.mark.parametrize("prompt", _CORPUS)
    def test_engine_matches_legacy_cascade(self, prompt, monkeypatch):
        from app.core.config_loader import cfg
        on = ip.assess(prompt)
        monkeypatch.setattr(cfg.policy, "enabled", False)
        off = ip.assess(prompt)
        assert on.decision == off.decision, prompt
        assert on.strategy == off.strategy, prompt
        # Engine-on carries the audit record; engine-off doesn't.
        assert on.policy is not None
        assert off.policy is None

    def test_record_shape(self):
        a = ip.assess("write a login api")
        assert a.policy["action"] in ("ANSWER", "CLARIFY", "DEFER")
        assert a.policy["rule_id"]
        assert isinstance(a.policy["fired"], list) and a.policy["fired"]
        assert all({"id", "action", "score"} <= set(f) for f in a.policy["fired"])


class TestEngine:
    def _ctx(self, **kw) -> dict:
        base = {"intent": "knowledge", "answerable": True,
                "missing_required": [], "missing_optional": [],
                "ambiguity": 0.1, "confidence": 0.9,
                "clarification_gain": 0.0, "clarification_cost": 0.6,
                "has_artifact": False, "slots": {}}
        base.update(kw)
        return base

    def test_answer_when_complete(self):
        d = decide(self._ctx())
        assert d.action == ACTION_ANSWER and d.rule_id == "answer_direct"

    def test_clarify_when_required_missing(self):
        d = decide(self._ctx(missing_required=["language"], answerable=False))
        assert d.action == ACTION_CLARIFY
        assert d.rule_id == "clarify_missing_required"

    def test_defer_on_unknown_intent(self):
        d = decide(self._ctx(intent="unknown"))
        assert d.action == ACTION_DEFER and d.rule_id == "defer_to_gate"

    def test_config_rule_overrides_by_priority(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.policy, "rules", [{
            "id": "always_clarify_comparisons",
            "action": "CLARIFY",
            "priority": 200,
            "reason": "test override",
            "when": [{"field": "intent", "op": "eq", "value": "comparison"}],
        }])
        d = decide(self._ctx(intent="comparison"))
        assert d.action == ACTION_CLARIFY
        assert d.rule_id == "always_clarify_comparisons"
        # Other intents are untouched by the scoped rule.
        d2 = decide(self._ctx(intent="knowledge"))
        assert d2.action == ACTION_ANSWER

    def test_config_can_disable_builtin(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.policy, "rules", [
            {"id": "answer_direct", "enabled": False},
        ])
        d = decide(self._ctx())
        assert d.rule_id == "defer_to_gate"      # answer rule removed → fallback

    def test_scoring_beats_first_match(self):
        rules = [
            PolicyRule(id="low", action=ACTION_DEFER, priority=10),
            PolicyRule(id="high", action=ACTION_ANSWER, priority=99,
                       benefit=1.0, weight=5.0),
        ]
        d = decide(self._ctx(), rules=rules)
        assert d.rule_id == "high"
        assert {f["id"] for f in d.fired} == {"low", "high"}

    def test_condition_operators(self):
        r = PolicyRule(id="x", action=ACTION_CLARIFY, when=[
            {"field": "confidence", "op": "lt", "value": 0.5},
            {"field": "missing_required", "op": "not_empty"},
        ])
        assert r.applies({"confidence": 0.3, "missing_required": ["language"]})
        assert not r.applies({"confidence": 0.9, "missing_required": ["language"]})
        assert not r.applies({"confidence": 0.3, "missing_required": []})

    def test_malformed_rule_is_skipped(self, monkeypatch):
        from app.core.config_loader import cfg
        monkeypatch.setattr(cfg.policy, "rules", [
            {"nonsense": True},                       # no id → skipped
            {"id": "bad_action", "action": "EXPLODE"},   # bad action → skipped
        ])
        ids = {r.id for r in load_rules()}
        assert ids == {"answer_direct", "clarify_missing_required",
                       "defer_to_gate"}

    def test_broken_condition_never_fires(self):
        r = PolicyRule(id="boom", action=ACTION_ANSWER,
                       when=lambda c: 1 / 0)          # raises inside
        assert not r.applies({})
