"""Declarative decision-policy engine (SeveralFeatures.md: Policy Engine).

The design doc's central architectural ask: instead of hardcoded `if` chains
deciding execute-vs-clarify, express the decision as POLICIES — declarative
rules with conditions, priority, and cost/benefit scoring — evaluated
uniformly, with every decision recorded (which rules fired, scores, why).

    policy:
      enabled: true
      rules:
        - id: always_clarify_deploys
          priority: 200
          action: CLARIFY
          when:
            - {field: intent, op: eq, value: deploy}

Design constraints honored here:
  * The BUILTIN rules replicate the pre-gate's legacy final gate EXACTLY
    (answer-direct / clarify-missing-required / defer) — enabling the engine
    changes zero decisions until someone adds or overrides rules in config.
  * Scoring, not first-match: every applicable rule gets
    `priority + weight * (benefit - cost)`; the highest score wins. Safety-
    class rules use high priority so they dominate optimization rules
    (meta-policy: safety > optimization).
  * Config rules OVERLAY builtins by id (same id replaces; `enabled: false`
    disables), so policies are versionable/testable without code changes.
  * Every decision returns a [PolicyDecision] record for the trace.
  * Fail-open: a broken rule is skipped; a broken engine falls back to the
    legacy cascade in the caller.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger(__name__)

ACTION_ANSWER = "ANSWER"
ACTION_CLARIFY = "CLARIFY"
ACTION_DEFER = "DEFER"
_ACTIONS = {ACTION_ANSWER, ACTION_CLARIFY, ACTION_DEFER}

# Condition operators for declarative (config-authored) rules.
_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
    "in": lambda a, b: a in (b or []),
    "not_in": lambda a, b: a not in (b or []),
    "gte": lambda a, b: _num(a) >= _num(b),
    "lte": lambda a, b: _num(a) <= _num(b),
    "gt": lambda a, b: _num(a) > _num(b),
    "lt": lambda a, b: _num(a) < _num(b),
    "truthy": lambda a, b: bool(a),
    "falsy": lambda a, b: not a,
    "contains": lambda a, b: b in (a or []),
    "empty": lambda a, b: not a,
    "not_empty": lambda a, b: bool(a),
}


def _num(x: Any) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class PolicyRule:
    """One decision policy. `when` is either a callable(ctx)->bool (builtin
    rules) or a list of {field, op, value} condition dicts (config rules),
    AND-combined."""
    id: str
    action: str
    priority: float = 50.0
    weight: float = 1.0
    cost: float = 0.0                # friction of the action (interruption…)
    benefit: float = 0.0             # expected value of the action
    when: Callable[[dict], bool] | list[dict] | None = None
    reason: str = ""
    source: str = "builtin"          # builtin | config
    enabled: bool = True

    def applies(self, ctx: dict) -> bool:
        try:
            if not self.enabled:
                return False
            if self.when is None:
                return True
            if callable(self.when):
                return bool(self.when(ctx))
            for cond in self.when:
                op = _OPS.get(str(cond.get("op", "eq")).lower())
                if op is None:
                    return False
                if not op(ctx.get(str(cond.get("field", ""))),
                          cond.get("value")):
                    return False
            return True
        except Exception:  # noqa: BLE001 — a broken rule never fires
            return False

    def score(self) -> float:
        return self.priority + self.weight * (self.benefit - self.cost)


@dataclass
class PolicyDecision:
    """The chosen action + full audit trail for the trace."""
    action: str
    rule_id: str
    reason: str = ""
    fired: list[dict] = field(default_factory=list)   # every applicable rule

    def as_dict(self) -> dict:
        return {"action": self.action, "rule_id": self.rule_id,
                "reason": self.reason, "fired": list(self.fired)}


# ---- builtin rules: EXACT replica of the legacy pre-gate final cascade -----
# Legacy (intent_pipeline.assess):
#   if answerable and not missing_req and gain < cost and intent != unknown
#       → ANSWER
#   elif missing_req → CLARIFY
#   else → DEFER
def _builtin_rules() -> list[PolicyRule]:
    return [
        PolicyRule(
            id="clarify_missing_required",
            action=ACTION_CLARIFY,
            priority=90.0,
            reason="A required detail is missing.",
            when=lambda c: bool(c.get("missing_required")),
        ),
        PolicyRule(
            id="answer_direct",
            action=ACTION_ANSWER,
            priority=80.0,
            reason="Request is specific enough to answer directly.",
            when=lambda c: (bool(c.get("answerable"))
                            and not c.get("missing_required")
                            and _num(c.get("clarification_gain"))
                            < _num(c.get("clarification_cost"))
                            and c.get("intent") != "unknown"),
        ),
        PolicyRule(
            id="defer_to_gate",
            action=ACTION_DEFER,
            priority=10.0,
            reason="Likely answerable; defer wording to the gate.",
            when=None,                       # always applicable (fallback)
        ),
    ]


def _config_rules() -> list[PolicyRule]:
    """Rules authored in `config policy.rules` (declarative overlay)."""
    out: list[PolicyRule] = []
    try:
        from app.core.config_loader import cfg
        raw = list(getattr(cfg.policy, "rules", None) or [])
    except Exception:  # noqa: BLE001
        return out
    for r in raw:
        try:
            if not isinstance(r, dict) or not r.get("id"):
                continue
            action = str(r.get("action", ACTION_DEFER)).upper()
            if action not in _ACTIONS:
                continue
            out.append(PolicyRule(
                id=str(r["id"]),
                action=action,
                priority=_num(r.get("priority", 50.0)),
                weight=_num(r.get("weight", 1.0)),
                cost=_num(r.get("cost", 0.0)),
                benefit=_num(r.get("benefit", 0.0)),
                when=list(r.get("when") or []) or None,
                reason=str(r.get("reason", "")),
                source="config",
                enabled=bool(r.get("enabled", True)),
            ))
        except Exception:  # noqa: BLE001 — one bad rule never sinks the rest
            log.info("policy: skipping malformed rule %r", r)
    return out


def load_rules() -> list[PolicyRule]:
    """Builtin rules overlaid by config rules (same id replaces/disables)."""
    rules = {r.id: r for r in _builtin_rules()}
    for r in _config_rules():
        if not r.enabled:
            rules.pop(r.id, None)
        else:
            rules[r.id] = r
    return list(rules.values())


def decide(ctx: dict, rules: list[PolicyRule] | None = None) -> PolicyDecision:
    """Evaluate all policies against the decision context and pick the
    highest-scoring applicable action. `ctx` is the flat decision state
    (intent, answerable, missing_required, ambiguity, confidence, risk_level,
    clarification_gain/cost, has_artifact, mode, …).
    """
    rules = rules if rules is not None else load_rules()
    fired: list[dict] = []
    best: PolicyRule | None = None
    for r in rules:
        if not r.applies(ctx):
            continue
        s = r.score()
        fired.append({"id": r.id, "action": r.action,
                      "score": round(s, 3), "source": r.source})
        if best is None or s > best.score():
            best = r
    if best is None:      # can't happen with the defer fallback, but be safe
        return PolicyDecision(action=ACTION_DEFER, rule_id="fallback",
                              reason="No policy applied.", fired=fired)
    return PolicyDecision(action=best.action, rule_id=best.id,
                          reason=best.reason, fired=fired)


__all__ = ["PolicyRule", "PolicyDecision", "decide", "load_rules",
           "ACTION_ANSWER", "ACTION_CLARIFY", "ACTION_DEFER"]
