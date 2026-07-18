"""Failure Prediction / pre-execution risk (roadmap Phase 4 #18).

Before running a task, flag the failure classes it's *likely* to hit from cheap
signals (needs network offline, huge input, unsupported language, missing SDK…),
so the planner can pick a safer strategy up front instead of failing then
repairing. Returns taxonomy-aligned predictions. Deterministic + fail-open.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.obs import failure_taxonomy as ft


@dataclass(frozen=True)
class RiskPrediction:
    failure_id: str
    likelihood: float      # 0..1 heuristic
    reason: str


@dataclass
class PreflightReport:
    predictions: list[RiskPrediction] = field(default_factory=list)

    @property
    def risky(self) -> bool:
        return any(p.likelihood >= 0.5 for p in self.predictions)

    def top(self) -> RiskPrediction | None:
        return max(self.predictions, key=lambda p: p.likelihood, default=None)


def predict(
    *,
    needs_network: bool = False,
    network_available: bool = True,
    input_chars: int = 0,
    language: str = "",
    supported_languages: set[str] | None = None,
    needs_sdk: str = "",
    available_sdks: set[str] | None = None,
    max_input_chars: int = 200_000,
) -> PreflightReport:
    """Predict likely failures from pre-execution signals. All args optional;
    each fires an independent, taxonomy-aligned prediction."""
    out: list[RiskPrediction] = []
    try:
        if needs_network and not network_available:
            out.append(RiskPrediction("network_error", 0.95,
                                      "task needs network but it's unavailable"))
        if input_chars and input_chars > max_input_chars:
            out.append(RiskPrediction("generation_timeout",
                                      min(0.9, 0.5 + input_chars / (max_input_chars * 10)),
                                      f"input {input_chars} chars exceeds budget {max_input_chars}"))
        if language and supported_languages is not None and language not in supported_languages:
            out.append(RiskPrediction("stt_unavailable", 0.7,
                                      f"language {language!r} not in supported set"))
        if needs_sdk and available_sdks is not None and needs_sdk not in available_sdks:
            out.append(RiskPrediction("verification_failed", 0.8,
                                      f"required SDK {needs_sdk!r} not available in sandbox"))
        # Only surface known taxonomy classes.
        out = [p for p in out if ft.get(p.failure_id) is not None]
    except Exception:  # noqa: BLE001
        out = []
    return PreflightReport(out)


__all__ = ["RiskPrediction", "PreflightReport", "predict"]
