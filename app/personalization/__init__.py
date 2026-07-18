"""Personalization + governance (personalization-and-governance spec).

A per-user model (expertise / verbosity / communication style / frustration), a
deterministic topic-risk policy gate that ADDS caution on sensitive domains, and
a read-only analytics/audit view — all by extending existing signals
(`User.preferences`, the answer-depth mechanic, the safety guards, the latency
observatory). Additive + flag-gated + fail-open: neutral user + general topic =
today's behavior; no second blocking LLM call; safety guards keep precedence.
"""
from .user_model import UserModel, infer, load_user_model, save_user_model
from .frustration import update_frustration, FrustrationState
from .policy import classify, strategy_for, TopicRisk
from .analytics import summary

__all__ = [
    "UserModel", "infer", "load_user_model", "save_user_model",
    "update_frustration", "FrustrationState",
    "classify", "strategy_for", "TopicRisk",
    "summary",
]
