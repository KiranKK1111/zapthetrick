"""Goal-completion detector — Phase 6 of the Document Generation roadmap.

DocuementGeneration.md #13 (goal completion) + #15 (intelligent next-step
suggestions): after answering, ask "was the user's objective actually
completed?" — detect which deliverables a project already has and suggest the
concrete ones that are missing ("Spring Boot API done → add unit tests, a
Dockerfile, Swagger docs, a CI pipeline") rather than a generic menu.

Deterministic + content-based (no LLM): it inspects the conversation/artifact
text for deliverable signals, classifies the project type, and diffs against the
deliverables that type usually needs. Fail-safe: an unknown project → the
generic deliverable set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Deliverable → (label, detection regex). Order is the suggestion order.
_DELIVERABLES: list[tuple[str, str, re.Pattern]] = [
    ("source_code", "Source code", re.compile(r"```[a-z0-9+#]", re.I)),
    ("readme", "README", re.compile(
        r"\breadme\b|^#\s|\bgetting started\b", re.I | re.M)),
    ("tests", "Unit tests", re.compile(
        r"\bdef test_|\b@Test\b|\bunittest\b|\bpytest\b|\bdescribe\(|\bit\(|"
        r"\bassert\b|\bjunit\b|\bexpect\(|\btest(?:s|ing)?\b", re.I)),
    ("dockerfile", "Dockerfile", re.compile(
        r"^\s*FROM\s+\w|\bdockerfile\b|\bdocker build\b|\bdocker-compose\b",
        re.I | re.M)),
    ("ci_cd", "CI/CD pipeline", re.compile(
        r"\.github/workflows|\bgitlab-ci\b|\bjenkinsfile\b|\bci/cd\b|"
        r"\bgithub actions\b|\bcircleci\b", re.I)),
    ("api_docs", "API docs (Swagger/OpenAPI)", re.compile(
        r"\bswagger\b|\bopenapi\b|\bapi docs?\b|\bpostman\b", re.I)),
    ("deps", "Dependency manifest", re.compile(
        r"\brequirements\.txt\b|\bpackage\.json\b|\bpom\.xml\b|"
        r"\bbuild\.gradle\b|\bpyproject\.toml\b|\bgo\.mod\b|\bCargo\.toml\b",
        re.I)),
    ("deployment", "Deployment manifest", re.compile(
        r"\bkubernetes\b|\bk8s\b|\bhelm\b|\bdeployment\.yaml\b|\bterraform\b",
        re.I)),
]

# Project type → the deliverables it usually needs.
_PROJECT_TYPES: list[tuple[str, re.Pattern, list[str]]] = [
    ("rest_api", re.compile(
        r"\brest api\b|\bendpoint\b|@RestController|\bspring boot\b|\bfastapi\b|"
        r"\bflask\b|\bexpress\b|\brouter\b|@app\.(?:get|post|route)", re.I),
     ["source_code", "readme", "tests", "api_docs", "deps", "dockerfile"]),
    ("web_app", re.compile(
        r"\breact\b|\bvue\b|\bangular\b|\bnext\.js\b|\bfront-?end\b|\bweb app\b",
        re.I),
     ["source_code", "readme", "tests", "deps", "dockerfile"]),
    ("cli", re.compile(
        r"\bcli\b|\bcommand-?line\b|\bargparse\b|\bargv\b|\bclick\b", re.I),
     ["source_code", "readme", "tests", "deps"]),
    ("library", re.compile(
        r"\blibrary\b|\bpackage\b|\bsdk\b|\bpublish to (?:pypi|npm)\b", re.I),
     ["source_code", "readme", "tests", "deps", "api_docs"]),
    ("data_ml", re.compile(
        r"\bmodel training\b|\bdataset\b|\bml pipeline\b|\bnotebook\b", re.I),
     ["source_code", "readme", "tests", "deps"]),
]
_GENERIC = ["source_code", "readme", "tests"]
_LABELS = {k: label for k, label, _ in _DELIVERABLES}
# "Why produce this" — used in the suggestion.
_WHY = {
    "readme": "so others can run it",
    "tests": "to lock in behavior + catch regressions",
    "dockerfile": "for a reproducible run environment",
    "ci_cd": "to build/test automatically on every change",
    "api_docs": "so consumers know the contract",
    "deps": "to pin the dependencies",
    "deployment": "to ship it",
    "source_code": "the implementation itself",
}


@dataclass
class Suggestion:
    key: str
    label: str
    reason: str

    def as_dict(self) -> dict:
        return {"key": self.key, "label": self.label, "reason": self.reason}


def detect_deliverables(text: str) -> set[str]:
    """The deliverable types present in the conversation/artifact text."""
    t = text or ""
    return {key for key, _label, pat in _DELIVERABLES if pat.search(t)}


_PROJECT_TYPE_EXEMPLARS: dict[str, list[str]] = {
    "rest_api": ["a rest api with endpoints", "a spring boot service",
                 "a fastapi backend", "an express api server"],
    "web_app": ["a react front-end", "a vue web app", "an angular application"],
    "cli": ["a command-line tool", "a cli using argparse", "a terminal app"],
    "library": ["a reusable library", "an sdk to publish", "a python package"],
    "data_ml": ["a model training pipeline", "an ml notebook",
                "a dataset processing job"],
    "generic": ["a small script", "explain this code", "a general program"],
}


def detect_project_type(text: str) -> str:
    """Project type SEMANTICALLY (embedding nearest-class); regex cues are the
    embedder-down fallback."""
    try:
        from app.semantics.gates import classify
        cls = classify(text, _PROJECT_TYPE_EXEMPLARS,
                       cache_key="project_type", threshold=0.42)
        if cls is not None:
            return cls
    except Exception:  # noqa: BLE001
        pass
    for name, pat, _ in _PROJECT_TYPES:  # fallback: deterministic cues
        if pat.search(text or ""):
            return name
    return "generic"


def _expected(project_type: str) -> list[str]:
    for name, _pat, expected in _PROJECT_TYPES:
        if name == project_type:
            return expected
    return _GENERIC


def suggest_next_deliverables(text: str, *, limit: int = 5) -> list[Suggestion]:
    """Given a project's conversation/artifacts, suggest the concrete missing
    deliverables — but only once there's actually a project (some source code):
    a chat with no code shouldn't nag about a Dockerfile."""
    present = detect_deliverables(text)
    if "source_code" not in present:
        return []
    project_type = detect_project_type(text)
    missing = [k for k in _expected(project_type) if k not in present]
    return [Suggestion(key=k, label=_LABELS.get(k, k),
                       reason=_WHY.get(k, ""))
            for k in missing][:limit]


def completion_report(text: str) -> dict:
    """A full picture: project type, present + missing deliverables + a
    completion percentage — the answer to 'is the objective done?'."""
    present = detect_deliverables(text)
    project_type = detect_project_type(text)
    expected = _expected(project_type)
    have = [k for k in expected if k in present]
    missing = [k for k in expected if k not in present]
    pct = round(100 * len(have) / len(expected)) if expected else 100
    return {
        "project_type": project_type,
        "present": have,
        "missing": missing,
        "completion_pct": pct,
        "suggestions": [s.as_dict() for s in suggest_next_deliverables(text)],
    }


__all__ = [
    "Suggestion", "detect_deliverables", "detect_project_type",
    "suggest_next_deliverables", "completion_report",
]
