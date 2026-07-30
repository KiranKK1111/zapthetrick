"""Post-render quality score (MermaidDiagramVisualizations.md #4).

    Generated → Validator → Quality Score 92% → Pass → Display

The doc's point is that a binary "did it parse?" is too weak a gate. A diagram
can render and still be dense, mislabelled, backwards or invisible to a screen
reader. A score turns the validator findings into ONE number a caller can gate on
— and, more usefully, four sub-scores that say *where* the quality went.

Scoring model, deliberately simple and explainable:
  * each category starts at 100 and loses points per finding, weighted by
    severity (error 34, warn 12, info 4) — so one error can't be out-weighted by
    a pile of nits, and three warnings still leave a passing category;
  * the overall score is a weighted mean: syntax and semantics matter most
    (a wrong diagram is worse than an ugly one), then style, then accessibility;
  * any `error` finding forces `passed = False` regardless of the number, because
    "92% but it points the wrong way" is not a pass.

Pure and deterministic — the same findings always give the same score.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.diagrams import validators as V
from app.diagrams.ir import DiagramIR

# Points removed from a category per finding.
_PENALTY = {V.ERROR: 34.0, V.WARN: 12.0, V.INFO: 4.0}
# Category weights in the overall mean.
_WEIGHTS = {V.SYNTAX: 0.35, V.SEMANTIC: 0.30, V.STYLE: 0.20,
            V.ACCESSIBILITY: 0.15}
# Below this a diagram is worth regenerating rather than showing.
PASS_THRESHOLD = 70.0

_GRADES = ((90.0, "excellent"), (80.0, "good"), (70.0, "fair"),
           (50.0, "poor"), (0.0, "unusable"))


@dataclass
class QualityScore:
    overall: float = 100.0
    passed: bool = True
    grade: str = "excellent"
    subscores: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    top_issues: list[dict] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict:
        return {"overall": round(self.overall, 1), "passed": self.passed,
                "grade": self.grade,
                "subscores": {k: round(v, 1) for k, v in self.subscores.items()},
                "counts": dict(self.counts), "top_issues": list(self.top_issues),
                "summary": self.summary}


def _grade(value: float) -> str:
    for floor, name in _GRADES:
        if value >= floor:
            return name
    return "unusable"


def score_findings(findings: list[V.Finding]) -> QualityScore:
    """Turn validator findings into a :class:`QualityScore`. Never raises."""
    result = QualityScore()
    try:
        subscores: dict[str, float] = {}
        for category in _WEIGHTS:
            penalty = sum(
                _PENALTY.get(f.severity, 0.0)
                for f in findings if f.category == category)
            subscores[category] = max(0.0, 100.0 - penalty)
        result.subscores = subscores
        result.overall = sum(subscores[c] * w for c, w in _WEIGHTS.items())

        counts = {V.ERROR: 0, V.WARN: 0, V.INFO: 0}
        for finding in findings:
            if finding.severity in counts:
                counts[finding.severity] += 1
        result.counts = counts

        result.passed = counts[V.ERROR] == 0 and result.overall >= PASS_THRESHOLD
        result.grade = _grade(result.overall)

        # The issues actually worth surfacing: errors first, then warnings,
        # capped so a UI can show them without a scroll.
        ordered = sorted(
            findings,
            key=lambda f: {V.ERROR: 0, V.WARN: 1, V.INFO: 2}.get(f.severity, 3))
        result.top_issues = [f.to_dict() for f in ordered[:6]]
        result.summary = _summary(result, counts)
    except Exception:  # noqa: BLE001
        pass
    return result


def _summary(result: QualityScore, counts: dict[str, int]) -> str:
    if counts[V.ERROR]:
        return (f"{counts[V.ERROR]} blocking issue(s) — the diagram is "
                f"{result.grade} and should be repaired before it is shown")
    if counts[V.WARN]:
        weakest = min(result.subscores, key=lambda c: result.subscores[c]) \
            if result.subscores else ""
        return (f"renders cleanly; {counts[V.WARN]} readability warning(s), "
                f"weakest area: {weakest or 'n/a'}")
    if counts[V.INFO]:
        return "renders cleanly with minor suggestions"
    return "clean across syntax, semantics, style and accessibility"


def score(ir: DiagramIR, *, source: str = "") -> tuple[QualityScore, V.ValidationReport]:
    """Validate `ir` and score it → `(QualityScore, ValidationReport)`."""
    report = V.validate(ir, source=source)
    return score_findings(report.findings), report


def score_source(source: str) -> tuple[QualityScore, V.ValidationReport]:
    """Validate + score raw Mermaid (lifted through the IR)."""
    report = V.validate_source(source)
    return score_findings(report.findings), report


__all__ = ["QualityScore", "PASS_THRESHOLD", "score", "score_source",
           "score_findings"]
