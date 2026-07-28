"""Addressee veto: interviewer speech aimed at a CO-PANELIST or at logistics
must not be answered as the candidate's turn, while questions TO the candidate
must never be vetoed (a false veto = a missed answer, the worse failure).
Embedder-gated like the floor veto; fail-open without it."""
import pytest

from app.semantics import gates


def _ready() -> bool:
    try:
        from app.rag import embedder
        embedder.embed(["warm"])
        return bool(embedder.is_ready())
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _ready(), reason="embedder unavailable")

ELSEWHERE = [
    "Do you have any questions for the candidate?",
    "Can you pull up the candidate's resume?",
    "Let me hand it over to my colleague now.",
    "Do we have time for one more question?",
    "We can discuss his answer after the call.",
]

TO_CANDIDATE = [
    "Can you walk me through your resume?",
    "Do you have any questions for us?",
    "How would you scale this system?",
    "Tell me about your current project.",
    "Do you have experience with Kafka?",
    "Can you share your screen?",
    "What is the difference between a process and a thread?",
]


@pytest.mark.parametrize("text", ELSEWHERE)
def test_panel_aside_is_vetoed(text):
    assert gates.matches("addressed_elsewhere", text) is True, text


@pytest.mark.parametrize("text", TO_CANDIDATE)
def test_candidate_directed_is_never_vetoed(text):
    assert gates.matches("addressed_elsewhere", text) is not True, text


def test_fail_open_without_gates(monkeypatch):
    monkeypatch.setattr(gates, "_enabled", lambda: False)
    assert gates.matches("addressed_elsewhere", "do we have time") is None
