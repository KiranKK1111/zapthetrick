"""Evaluation + reliability meta-layer (evaluation-and-reliability spec).

Runtime reliability components — all flag-gated and fail-open so flags off =
today's behavior byte-for-byte:

  • ``confidence`` — per-subsystem ``SubsystemConfidence`` + an ``aggregate`` that
    extends ``app/chat/trust.py`` and gates proceed / defer-to-clarifier / judgment.
  • ``governor`` — fast vs deep pipeline selection from the existing difficulty.
  • ``degrade`` — a guard that swaps a failed subsystem for a safe fallback.
  • ``critic`` — a deterministic, non-blocking post-answer quality check.

The offline measurement face (scenario matrix + baseline) lives in ``app/eval``.
"""
from .confidence import SubsystemConfidence, aggregate, gate

__all__ = ["SubsystemConfidence", "aggregate", "gate"]
